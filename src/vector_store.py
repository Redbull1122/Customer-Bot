from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from langchain_core.documents import Document
import hashlib

from .config import PINECONE_API_KEY, PINECONE_INDEX


_pc: Optional[Pinecone] = None
_model: Optional[SentenceTransformer] = None
_index = None
INDEX_NAME = PINECONE_INDEX


def get_embedding_model() -> SentenceTransformer:
    """Get or create embedding model (supports Ukrainian text)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("intfloat/multilingual-e5-large")
    return _model


def get_pinecone_client() -> Optional[Pinecone]:
    global _pc
    if _pc is None and PINECONE_API_KEY:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
    return _pc


def get_or_create_index(index_name: str = INDEX_NAME):
    """Get or create a Pinecone index."""
    global _index
    if _index is not None:
        return _index

    pc = get_pinecone_client()
    
    if not pc:
        print("WARNING: PINECONE_API_KEY is not set. Falling back to local behavior.")
        return None
    
    try:
        # Check whether index already exists
        try:
            existing_indexes = [idx.name for idx in pc.list_indexes()]
        except (AttributeError, TypeError):
            # Alternative path for different API versions
            indexes_list = pc.list_indexes()
            if hasattr(indexes_list, 'indexes'):
                existing_indexes = [idx['name'] for idx in indexes_list.indexes]
            else:
                existing_indexes = []
        
        if index_name not in existing_indexes:
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec={
                    "serverless": {
                        "cloud": "aws",
                        "region": "us-east-1"
                    }
                }
            )
            print(f"Index {index_name} created successfully.")
        
        _index = pc.Index(index_name)
        return _index
    except Exception as e:
        print(f"Pinecone error: {e}")
        return None


def generate_chunk_id(text: str, page: int, chunk_index: int) -> str:
    """Generate a unique ID for a chunk."""
    content_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"chunk_{page}_{chunk_index}_{content_hash}"


def add_documents_to_vector_store(
    documents: List[Document],
    index_name: str = INDEX_NAME,
    batch_size: int = 100
) -> bool:
    """
    Upload documents into Pinecone vector database.
    
    Args:
        documents: List of Document objects to index
        index_name: Target Pinecone index name
        batch_size: Upload batch size
    
    Returns:
        True if successful, False otherwise
    """
    index = get_or_create_index(index_name)
    if not index:
        print("Cannot upload documents: index is unavailable")
        return False
    
    model = get_embedding_model()
    
    # Prepare vectors for upload
    vectors_to_upsert = []
    
    for i, doc in enumerate(documents):
        # Build embedding for text
        text = doc.page_content
        if not text.strip():
            continue
            
        embedding = model.encode(text).tolist()
        
        # Generate unique ID
        chunk_id = generate_chunk_id(text, doc.metadata.get('page', 0), i)

        # Prepare metadata
        metadata = {
            "text": text,
            "page": doc.metadata.get('page', 0),
            "source": doc.metadata.get('source', 'unknown'),
            "chunk_id": doc.metadata.get('chunk_id', chunk_id),
            "chunk_index": doc.metadata.get('chunk_index', i),
            "chunk_page_index": doc.metadata.get('chunk_page_index', 0),
            "chunk_file": doc.metadata.get('chunk_file', f"chunk_{doc.metadata.get('page', 0)}_{i}.md"),
            "chunk_title": doc.metadata.get('chunk_title', '')
        }
        
        # Add image info when available
        if 'images' in doc.metadata and doc.metadata['images']:
            # Pinecone does not support lists directly, store as JSON string
            import json
            metadata['images'] = json.dumps(doc.metadata['images'])
        
        vectors_to_upsert.append({
            "id": chunk_id,
            "values": embedding,
            "metadata": metadata
        })
        
        # Upload in batches
        if len(vectors_to_upsert) >= batch_size:
            try:
                index.upsert(vectors=vectors_to_upsert)
                print(f"Uploaded {len(vectors_to_upsert)} chunks to Pinecone")
                vectors_to_upsert = []
            except Exception as e:
                print(f"Batch upload error: {e}")
                return False
    
    # Upload final batch
    if vectors_to_upsert:
        try:
            index.upsert(vectors=vectors_to_upsert)
            print(f"Uploaded final {len(vectors_to_upsert)} chunks to Pinecone")
        except Exception as e:
            print(f"Final batch upload error: {e}")
            return False
    
    print(f"Uploaded {len(documents)} documents to the vector store")
    return True


def search_similar_documents(
    query: str,
    top_k: int = 8,
    index_name: str = INDEX_NAME,
    include_metadata: bool = True
) -> List[Dict]:
    """
    Search for the most relevant documents for a query.
    
    Args:
        query: Query text
        top_k: Number of top relevant results
        index_name: Pinecone index name
        include_metadata: Whether to include metadata in results
    
    Returns:
        List of result dictionaries, each containing:
        - 'text': chunk text
        - 'score': relevance score (0-1, where 1 is most relevant)
        - 'page': page number
        - 'source': document source
    """
    index = get_or_create_index(index_name)
    if not index:
        print("WARNING: Pinecone is unavailable. Returning empty list.")
        return []
    
    model = get_embedding_model()
    
    # Build query embedding
    query_embedding = model.encode(query).tolist()
    
    try:
        # Run search
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata
        )
        
        # Normalize result format (API can return different structures)
        formatted_results = []
        
        # Get matches (attribute or dict key depending on SDK version)
        if hasattr(results, 'matches'):
            matches = results.matches
        elif isinstance(results, dict) and 'matches' in results:
            matches = results['matches']
        else:
            matches = []
        
        for match in matches:
            # Handle different match object formats
            if hasattr(match, 'metadata'):
                metadata = match.metadata
                score = getattr(match, 'score', 0.0)
            elif isinstance(match, dict):
                metadata = match.get('metadata', {})
                score = match.get('score', 0.0)
            else:
                continue
            
            result = {
                'text': metadata.get('text', '') if metadata else '',
                'score': float(score),
                'page': metadata.get('page', None) if metadata else None,
                'source': metadata.get('source', None) if metadata else None,
                'chunk_id': metadata.get('chunk_id', None) if metadata else None,
                'chunk_index': metadata.get('chunk_index', None) if metadata else None,
                'chunk_page_index': metadata.get('chunk_page_index', None) if metadata else None,
                'chunk_file': metadata.get('chunk_file', None) if metadata else None,
                'chunk_title': metadata.get('chunk_title', None) if metadata else None
            }
            
            # Parse images from metadata if present
            if metadata and 'images' in metadata:
                import json
                try:
                    result['images'] = json.loads(metadata['images'])
                except (json.JSONDecodeError, TypeError):
                    result['images'] = []
            
            formatted_results.append(result)
        
        return formatted_results
    
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return []


def clear_index(index_name: str = INDEX_NAME) -> bool:
    """
    Delete the full index (use with caution).
    
    Args:
        index_name: Index name to delete
    
    Returns:
        True if successful
    """
    pc = get_pinecone_client()
    if not pc:
        return False
    
    try:
        pc.delete_index(index_name)
        print(f"Deleted index {index_name}")
        global _index
        _index = None
        return True
    except Exception as e:
        print(f"Index deletion error: {e}")
        return False



#if PINECONE_API_KEY:
 #   get_or_create_index()

