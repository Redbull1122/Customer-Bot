"""
Semantic image search module based on text queries.
Uses a CLIP model to find the most relevant images.
"""

from typing import List, Dict, Tuple
from pathlib import Path
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os



def _calculate_text_relevance(query: str, context: str) -> float:
    """
    Compute query-to-context text relevance.
    Uses shared keyword overlap.
    
    Args:
        query: User query
        context: Text context around an image
        
    Returns:
        Relevance score from 0 to 1
    """
    # Stop words
    stop_words = {
        'how', 'what', 'where', 'when', 'why', 'which',
        'in', 'on', 'from', 'to', 'for', 'about', 'at', 'under',
        'and', 'or', 'but', 'is', 'was', 'will', 'has', 'have',
        'this', 'that', 'these', 'those', 'my', 'your', 'our', 'their',
        'show', 'looks', 'look', 'can', 'with'
    }
    
    # Extract keywords from query
    query_words = set(
        word.lower().strip('.,!?;:')
        for word in query.split()
        if word.lower() not in stop_words and len(word) > 2
    )
    
    # Extract words from context
    context_words = set(
        word.lower().strip('.,!?;:')
        for word in context.split()
        if word.lower() not in stop_words and len(word) > 2
    )
    
    if not query_words or not context_words:
        return 0.0
    
    # Compute Jaccard similarity
    intersection = len(query_words.intersection(context_words))
    union = len(query_words.union(context_words))
    
    if union == 0:
        return 0.0
    
    jaccard = intersection / union
    
    # Also include query-word coverage in context
    coverage = intersection / len(query_words) if query_words else 0.0
    
    # Combine both metrics
    relevance = (jaccard * 0.4 + coverage * 0.6)
    
    return min(relevance, 1.0)


def _calculate_size_penalty(image_path: str) -> Tuple[float, str]:
    """
    Calculate image-size penalty.
    Instead of removing images, lower score for tiny/decorative ones.
    
    Args:
        image_path: Image path
        
    Returns:
        Tuple (penalty_multiplier, reason)
        penalty_multiplier: 0.0-1.0 where 1.0 = no penalty, 0.0 = max penalty
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # Very small images (likely icons)
        if width < 30 or height < 30:
            return 0.1, f"very small ({width}x{height})"
        
        # Small images (possibly icons, but can still be relevant)
        if width < 50 or height < 50:
            return 0.5, f"small ({width}x{height})"
        
        # Very extreme aspect ratio (likely lines/decor)
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > 20:
            return 0.2, f"line/decor (ratio {aspect_ratio:.1f})"
        
        # Narrow aspect ratio (possibly decorative)
        if aspect_ratio > 10:
            return 0.6, f"narrow (ratio {aspect_ratio:.1f})"
        
        # Normal image, no penalty
        return 1.0, "OK"
        
    except Exception as e:
        return 0.5, f"error: {e}"


class ImageSearchEngine:
    """CLIP-based image search engine."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model for image search.
        
        Args:
            model_name: CLIP model name from HuggingFace
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[ImageSearch] Using device: {self.device}")
        
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        
        # Cache for image embeddings
        self._image_embeddings_cache: Dict[str, torch.Tensor] = {}
    
    def _get_image_embedding(self, image_path: str) -> torch.Tensor:
        """
        Get image embedding.
        
        Args:
            image_path: Image path
            
        Returns:
            Tensor with image embedding
        """
        # Check cache
        if image_path in self._image_embeddings_cache:
            return self._image_embeddings_cache[image_path]
        
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Store in cache
            self._image_embeddings_cache[image_path] = image_features
            return image_features
            
        except Exception as e:
            print(f"[ImageSearch] Error processing image {image_path}: {e}")
            return None
    
    def _get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Get text embedding.
        
        Args:
            text: Text query
            
        Returns:
            Tensor with text embedding
        """
        try:
            # CLIP has a 77-token limit, so trim text
            # Approx. 4 chars per token with margin
            max_chars = 250
            if len(text) > max_chars:
                text = text[:max_chars]
            
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                # Normalize for cosine similarity
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features
            
        except Exception as e:
            print(f"[ImageSearch] Error processing text: {e}")
            return None
    
    def search_images_by_text(
        self, 
        query: str, 
        image_paths: List[str], 
        top_k: int = 7,
        min_score: float = 0.3
    ) -> List[Dict[str, any]]:
        """
        Search for the most relevant images for a text query.
        
        Args:
            query: Text query
            image_paths: List of image paths to search
            top_k: Number of top results
            min_score: Minimum similarity threshold (0-1)
            
        Returns:
            List of result dictionaries: [{'path': str, 'score': float}, ...]
        """
        if not image_paths:
            return []
        
        # Get query embedding
        text_embedding = self._get_text_embedding(query)
        if text_embedding is None:
            return []
        
        # Compute similarity for each image
        results = []
        for img_path in image_paths:
            if not os.path.exists(img_path):
                continue
                
            img_embedding = self._get_image_embedding(img_path)
            if img_embedding is None:
                continue
            
            # Cosine similarity (vectors are normalized)
            similarity = (text_embedding @ img_embedding.T).item()
            
            if similarity >= min_score:
                results.append({
                    'path': img_path,
                    'score': similarity
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
    
    def search_images_by_context(
        self,
        context_text: str,
        page_images: Dict[int, List[str]],
        top_k: int = 5
    ) -> List[Dict[str, any]]:
        """
        Search images based on document context.
        
        Args:
            context_text: Document context text
            page_images: Dict {page_num: [image_paths]}
            top_k: Number of top results
            
        Returns:
            List of most relevant images
        """
        # Collect all images
        all_images = []
        for page_num, images in page_images.items():
            all_images.extend(images)
        
        if not all_images:
            return []
        
        # Run search
        return self.search_images_by_text(context_text, all_images, top_k=top_k)


# Global reusable instance
_image_search_engine = None


def get_image_search_engine() -> ImageSearchEngine:
    """Get or create global ImageSearchEngine instance."""
    global _image_search_engine
    if _image_search_engine is None:
        _image_search_engine = ImageSearchEngine()
    return _image_search_engine


def find_relevant_images(
    query: str,
    context_chunks: List[Dict],
    top_k: int = 5
) -> List[Dict[str, any]]:
    """
    Find the most relevant images for a user query.
    Uses CLIP semantic search and applies size penalties.
    
    Args:
        query: User query
        context_chunks: List of context chunks with images (sorted by relevance)
        top_k: Number of images to return
        
    Returns:
        List of relevant images with scores and descriptions
    """
    # Collect images with their associated text context
    image_contexts = {}  # {image_path: {'text': str, 'page': int, 'chunk_score': float, 'size_penalty': float}}
    
    for chunk in context_chunks:
        if 'images' in chunk and chunk['images']:
            chunk_text = chunk.get('text', '')
            page = chunk.get('page', 'N/A')
            chunk_score = chunk.get('score', 0)
            
            for img_path in chunk['images']:
                if not Path(img_path).exists():
                    continue
                
                # Compute size penalty (do not remove, reduce score)
                size_penalty, size_reason = _calculate_size_penalty(img_path)
                
                # Store context per image
                # If image appears in multiple chunks, merge context
                if img_path in image_contexts:
                    # Append text and keep the higher chunk score
                    image_contexts[img_path]['text'] += '\n' + chunk_text
                    image_contexts[img_path]['chunk_score'] = max(
                        image_contexts[img_path]['chunk_score'],
                        chunk_score
                    )
                    # Keep the smallest penalty (best-case)
                    image_contexts[img_path]['size_penalty'] = min(
                        image_contexts[img_path]['size_penalty'],
                        size_penalty
                    )
                else:
                    image_contexts[img_path] = {
                        'text': chunk_text,
                        'page': page,
                        'chunk_score': chunk_score,
                        'size_penalty': size_penalty,
                        'size_reason': size_reason
                    }
    
    if not image_contexts:
        return []
    
    # Log size-penalty statistics
    size_stats = {}
    for ctx in image_contexts.values():
        reason = ctx.get('size_reason', 'OK')
        if reason not in size_stats:
            size_stats[reason] = 0
        size_stats[reason] += 1
    
    if len(size_stats) > 1 or 'OK' not in size_stats:
        print(f"[ImageSearch] Size penalties: {size_stats}")
    
    # Use CLIP for semantic image search
    search_engine = get_image_search_engine()
    image_paths = list(image_contexts.keys())
    
    # Run semantic search for the query
    clip_results = search_engine.search_images_by_text(
        query=query,
        image_paths=image_paths,
        top_k=top_k * 3,  # Retrieve more for post-filtering
        min_score=0.22  # Slightly higher threshold for quality
    )
    
    # Build final results with context-based descriptions
    results = []
    for clip_result in clip_results:
        img_path = clip_result['path']
        clip_score = clip_result['score']
        context = image_contexts[img_path]
        
        # Compute text relevance between query and context
        text_relevance = _calculate_text_relevance(query, context['text'])
        
        # Combine three factors for final ranking:
        # 50% CLIP score (image semantic similarity)
        # 30% text_relevance (how well context matches query)
        # 20% chunk score (overall chunk relevance)
        combined_score = (
            clip_score * 0.50 +
            text_relevance * 0.30 +
            context['chunk_score'] * 0.20
        )
        
        # Apply size penalty (reduce score, do not remove)
        size_penalty = context.get('size_penalty', 1.0)
        final_score = combined_score * size_penalty
        
        # Generate description based on surrounding context
        description = _generate_image_description(
            context['text'][:500],
            context['page'],
            query
        )
        
        results.append({
            'path': img_path,
            'score': final_score,  # Use final_score after penalty
            'clip_score': clip_score,
            'text_relevance': text_relevance,
            'text_score': context['chunk_score'],
            'size_penalty': size_penalty,
            'size_reason': context.get('size_reason', 'OK'),
            'description': description,
            'page': context['page'],
            'context': context['text'][:300]  # Include short context snippet
        })
    
    # Sort by final score
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter out results with very low final score
    # "How it looks" queries need high visual relevance
    min_combined_score = 0.30
    filtered_results = [r for r in results if r['score'] >= min_combined_score]
    
    if filtered_results:
        print(f"[ImageSearch] Filtered out {len(results) - len(filtered_results)} low-score images")
        results = filtered_results
    
    return results[:top_k]


def _generate_image_description(context: str, page: int, query: str) -> str:
    """
    Generate a short image description based on context.
    
    Args:
        context: Text context around the image
        page: Page number
        query: User query
        
    Returns:
        Short image description
    """
    if not context or len(context.strip()) < 10:
        return f"Image from page {page}"
    
    # Normalize whitespace in context
    context = ' '.join(context.split())
    
    # Find the most relevant sentence containing query keywords
    query_words = set(query.lower().split())
    sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 15]
    
    # Select sentence with maximum query-word overlap
    best_sentence = None
    max_matches = 0
    
    for sentence in sentences[:5]:  # Check first 5 sentences
        sentence_words = set(sentence.lower().split())
        matches = len(query_words.intersection(sentence_words))
        if matches > max_matches:
            max_matches = matches
            best_sentence = sentence
    
    # Use best matching sentence when available
    if best_sentence and max_matches > 0:
        description = best_sentence
    elif sentences:
        # Otherwise use first sentence
        description = sentences[0]
    else:
        # If no sentences found, use first 120 chars
        description = context[:120]
    
    # Limit description length
    if len(description) > 120:
        description = description[:117] + "..."
    
    # Add page number
    description = f"{description} (p. {page})"
    
    return description
