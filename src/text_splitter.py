from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


def split_documents(documents, chunk_size=500):
    """
    Split documents into smaller chunks while preserving context.
    
    Args:
        documents: List of Document objects
        chunk_size: Chunk size in characters (reduced for embedding model compatibility)
        
    Returns:
        List of Document chunks with metadata
    """
    # Use improved separators for technical documentation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=90,  # Reduced overlap to avoid overly long sequences
        length_function=len,
        separators=[
            "\n\n\n",  # Sections
            "\n\n",    # Paragraphs
            "\n",      # Lines
            ". ",      # Sentences
            ", ",      # Phrases
            " ",       # Words
            ""         # Characters
        ],
        keep_separator=True  # Keep separators for better context continuity
    )

    chunks = text_splitter.split_documents(documents)
    
    # Add extra metadata for better retrieval
    page_chunk_counters = {}

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = int(chunk.metadata.get("page", 0))

        page_chunk_index = page_chunk_counters.get(page, 0)
        page_chunk_counters[page] = page_chunk_index + 1

        # Add chunk indexes
        chunk.metadata['chunk_index'] = i
        chunk.metadata['chunk_page_index'] = page_chunk_index

        # Virtual "small file" identifier for a specific fragment
        chunk_id = f"p{page}_c{page_chunk_index}"
        chunk.metadata['chunk_id'] = chunk_id
        chunk.metadata['chunk_file'] = f"{source}::page_{page}/chunk_{page_chunk_index}.md"
        
        # Add positional hints inside the document
        if i > 0:
            chunk.metadata['has_previous'] = True
        if i < len(chunks) - 1:
            chunk.metadata['has_next'] = True
        
        # Use first sentence as chunk title
        first_sentence = chunk.page_content.split('.')[0][:100]
        chunk.metadata['chunk_title'] = first_sentence
    
    if chunks:
        print(f"\n[TextSplitter] Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n--- Chunk {i} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Page: {chunk.metadata.get('page')}")
            print(f"Length: {len(chunk.page_content)}")
            print(f"Title: {chunk.metadata.get('chunk_title', 'N/A')}")
            print(f"Preview: {chunk.page_content[:150]}...")

        if len(chunks) > 3:
            print(f"\n... and {len(chunks) - 3} more chunks")

    return chunks


def merge_related_chunks(chunks: List[Document], similarity_threshold: float = 0.8) -> List[Document]:
    """
    Merge highly similar chunks to reduce duplicate information.
    
    Args:
        chunks: List of chunks
        similarity_threshold: Similarity threshold for merging
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return chunks
    
    merged = []
    skip_indices = set()
    
    for i, chunk in enumerate(chunks):
        if i in skip_indices:
            continue
        
        current_content = chunk.page_content
        current_page = chunk.metadata.get('page')
        
        # Look ahead for chunks from the same page that can be merged
        for j in range(i + 1, min(i + 3, len(chunks))):
            if j in skip_indices:
                continue
            
            next_chunk = chunks[j]
            next_page = next_chunk.metadata.get('page')
            
            # Merge only chunks from the same page
            if current_page == next_page:
                # Check overlap
                overlap = _calculate_text_overlap(current_content, next_chunk.page_content)
                
                if overlap > similarity_threshold:
                    # Merge chunks
                    current_content = _merge_texts(current_content, next_chunk.page_content)
                    skip_indices.add(j)
        
        # Create a new chunk with merged content
        merged_chunk = Document(
            page_content=current_content,
            metadata=chunk.metadata.copy()
        )
        merged.append(merged_chunk)
    
    print(f"[TextSplitter] Merged chunks: {len(chunks)} -> {len(merged)}")
    return merged


def _calculate_text_overlap(text1: str, text2: str) -> float:
    """Compute overlap between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0


def _merge_texts(text1: str, text2: str) -> str:
    """Merge two texts while removing duplicates."""
    # Simple approach: add text2 only if it does not duplicate text1 suffix
    if text2.strip() in text1:
        return text1
    
    # Find overlap between text1 suffix and text2 prefix
    overlap_size = 0
    for i in range(min(len(text1), len(text2)), 0, -1):
        if text1[-i:] == text2[:i]:
            overlap_size = i
            break
    
    if overlap_size > 0:
        return text1 + text2[overlap_size:]
    else:
        return text1 + "\n" + text2

