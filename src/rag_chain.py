import argparse
import time
from typing import Dict, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from .config import PROMPT_TEMPLATE, GEMINI_API_KEY
from .vector_store import search_similar_documents


_LLM: Optional[ChatGoogleGenerativeAI] = None


def build_prompt(context: str, question: str) -> str:
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    return prompt_template.format(context=context, question=question)


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a singleton LLM client for the current process."""
    global _LLM
    if _LLM is None:
        _LLM = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            project="bot-hajster",
            location="europe-west1",
            temperature=0.2,
            max_tokens=None,
            api_key=GEMINI_API_KEY
        )
    return _LLM


def generate_answer(
    query_text: str, 
    top_k: int = 8,
    memory: Optional[any] = None,
    use_query_expansion = None,
    use_image_search = None,
    validate_answer = None
) -> Dict:
    """
    Run vector retrieval and generate an LLM answer with chat history context.
    
    Args:
        query_text: Current user question
        top_k: Number of retrieval results
        memory: LangChain ConversationBufferWindowMemory instance for history
    
    Returns a dictionary: {'answer': str, 'sources': List, 'results': List[Dict], 'context_chunks': List}
    """
    t0 = time.perf_counter()

    # 1. Adaptive top_k based on query complexity
    query_lower = query_text.lower()
    
    # Keywords that usually require more context
    complex_keywords = [
        "setup", "configure", "configuration",
        "connection", "connect", "installation", "mounting",
        "how it works", "working principle", "all", "full",
        "specifications", "parameters", "characteristics"
    ]
    
    if any(keyword in query_lower for keyword in complex_keywords):
        top_k = max(top_k, 8)
        print(f"[DEBUG] Increased top_k to {top_k} for a complex query")

    # 2. Vector store retrieval
    t_retrieval_start = time.perf_counter()
    results = search_similar_documents(query_text, top_k=top_k)
    retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000
    print(f"[DEBUG] Search results count: {len(results)} for query: {query_text}")

    if not results:
        return {
            "answer": "Sorry, I could not find relevant information in the manual.",
            "sources": [],
            "results": []
        }
    
    # 3. Filter by minimum relevance threshold
    # Threshold 0.3 is low enough to keep partially relevant results
    min_score = 0.3
    filtered_results = [r for r in results if r.get('score', 0) >= min_score]
    
    if not filtered_results:
        # If all results are below threshold, keep at least top-3
        filtered_results = results[:3]
        print("[DEBUG] All results below threshold, using top-3")
    else:
        print(f"[DEBUG] Filtered: {len(filtered_results)}/{len(results)} results with score >= {min_score}")
    
    results = filtered_results

    # Build numbered context blocks for better LLM grounding
    context_parts = []
    for idx, item in enumerate(results, 1):
        if item.get("text"):
            score = item.get("score", 0)
            page = item.get("page", "?")
            chunk_id = item.get("chunk_id", f"p{page}_c?")
            context_parts.append(
                f"[Fragment {idx}, Page {page}, Chunk {chunk_id}, Relevance: {score:.2f}]\n{item['text']}"
            )
    
    context_text = "\n\n---\n\n".join(context_parts)
    
    # Build question with chat history from LangChain memory
    if memory is not None:
        # Load history from memory
        chat_history = memory.load_memory_variables({})
        history_text = chat_history.get("history", "")
        
        if history_text:
            enhanced_question = f"""PREVIOUS CONVERSATION:
{history_text}

CURRENT QUESTION:
{query_text}

Answer the current question while considering the prior conversation context."""
        else:
            enhanced_question = query_text
    else:
        enhanced_question = query_text
    
    prompt = build_prompt(context=context_text, question=enhanced_question)

    llm = get_llm()
    t_llm_start = time.perf_counter()
    response = llm.invoke(prompt)
    llm_ms = (time.perf_counter() - t_llm_start) * 1000
    answer_text = response.content if hasattr(response, "content") else str(response)
    
    # Save exchange to memory
    if memory is not None:
        memory.save_context({"input": query_text}, {"output": answer_text})
    
    # Build structured chunks for UI output (text + images)
    # Group by page to avoid duplicate images
    from collections import defaultdict
    page_groups = defaultdict(
        lambda: {"texts": [], "images": [], "source": None, "score": 0, "chunk_ids": [], "chunk_files": []}
    )
    
    for item in results:
        page = item.get("page")
        text = item.get("text", "")
        source = item.get("source")
        score = item.get("score", 0)
        chunk_id = item.get("chunk_id")
        chunk_file = item.get("chunk_file")
        
        # Extract images
        images = item.get("images", []) if page is not None else []
        
        if page is not None:
            page_groups[page]["texts"].append(text)
            page_groups[page]["source"] = source
            page_groups[page]["score"] = max(page_groups[page]["score"], score)
            if chunk_id and chunk_id not in page_groups[page]["chunk_ids"]:
                page_groups[page]["chunk_ids"].append(chunk_id)
            if chunk_file and chunk_file not in page_groups[page]["chunk_files"]:
                page_groups[page]["chunk_files"].append(chunk_file)
            if images:
                # Add images without duplicates
                for img in images:
                    if img not in page_groups[page]["images"]:
                        page_groups[page]["images"].append(img)
    
    # Build context_chunks with unique images
    context_chunks = []
    sources = []
    
    for page in sorted(page_groups.keys()):
        group = page_groups[page]
        # Combine all texts from this page
        combined_text = "\n\n".join(group["texts"])
        
        # Sort images
        images = sorted(group["images"], key=lambda x: (
            int(x.split('page')[1].split('_')[0]) if 'page' in x else 0,
            int(x.split('img')[1].split('.')[0]) if 'img' in x else 0
        ))
        
        context_chunks.append({
            "text": combined_text,
            "images": images,
            "page": page,
            "source": group["source"],
            "score": group["score"],
            "chunk_ids": group["chunk_ids"],
            "chunk_files": group["chunk_files"]
        })
        
        sources.append({
            "page": page,
            "source": group["source"],
            "images": images,
            "chunk_ids": group["chunk_ids"],
            "chunk_files": group["chunk_files"]
        })

    total_ms = (time.perf_counter() - t0) * 1000
    debug_metrics = {
        "retrieval_ms": round(retrieval_ms, 1),
        "llm_ms": round(llm_ms, 1),
        "total_ms": round(total_ms, 1)
    }
    print(f"[PERF] retrieval={debug_metrics['retrieval_ms']}ms llm={debug_metrics['llm_ms']}ms total={debug_metrics['total_ms']}ms")

    return {
        "answer": answer_text,
        "sources": sources,
        "results": results,
        "context_chunks": context_chunks,  # Detailed chunks for the UI
        "debug_metrics": debug_metrics
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--top_k", type=int, default=3, help="How many results to retrieve")
    args = parser.parse_args()

    output = generate_answer(args.query_text, top_k=args.top_k)

    print("\n=== Answer ===")
    print(output["answer"])
    if output["sources"]:
        print("\nSources:")
        for src in output["sources"]:
            print(f"- page: {src.get('page')}, source: {src.get('source')}")


if __name__ == "__main__":
    main()
