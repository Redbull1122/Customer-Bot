import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from langchain_core.documents import Document

from .config import PDF_PATH, IMAGES_DIR

# Docling imports for PDF processing
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode

# PyMuPDF import for fallback
import fitz

logger = logging.getLogger(__name__)


def _extract_text_with_fitz(pdf_path: str) -> List[Document]:
    """Fallback: extract page-by-page text using PyMuPDF."""
    documents: List[Document] = []
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        for page_num, page in enumerate(doc):
            page_text = page.get_text("text") or ""
            metadata = {
                "source": pdf_path,
                "page": page_num,
                "total_pages": total_pages
            }
            documents.append(Document(page_content=page_text, metadata=metadata))
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF fallback text extraction error: {e}")
    return documents


def _get_page_markdown(page_entry) -> str:
    """
    Return page markdown text across different Docling page entry formats.
    Supports: page object or tuple/list containing a page object.
    """
    # Most common case: page object
    if hasattr(page_entry, "export_to_markdown"):
        return page_entry.export_to_markdown() or ""

    # Compatibility for variants where Docling returns tuple/list
    if isinstance(page_entry, (tuple, list)):
        for item in page_entry:
            if hasattr(item, "export_to_markdown"):
                return item.export_to_markdown() or ""
        # If no markdown object exists, return first text element
        for item in page_entry:
            if isinstance(item, str):
                return item

    return ""

def _extract_images_with_fitz(pdf_path: str, images_dir: Path) -> Dict[int, List[str]]:
    """
    Extract images from PDF using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        images_dir: Directory to save extracted images
    
    Returns:
        Dict {page_number: [list_of_image_paths]}
    """
    
    os.makedirs(images_dir, exist_ok=True)
    page_images = {}
    
    try:
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            images_on_page = []
            for img_index, img in enumerate(page.get_images(full=True)):
                pix = None
                pix_rgb = None
                try:
                    xref = img[0]
                    image_filename = f"page{page_num}_img{img_index}.png"
                    image_path = images_dir / image_filename
                    
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n == 4:  # CMYK
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        pix_rgb.save(image_path)
                    elif pix.n == 1:  # Grayscale
                        pix.save(image_path)
                    elif pix.n == 3:  # RGB
                        pix.save(image_path)
                    else:
                        pix_rgb = fitz.Pixmap(fitz.csRGB, pix)
                        pix_rgb.save(image_path)
                    
                    images_on_page.append(str(image_path))
                except Exception as e:
                    logger.warning(f"Image extraction error on page {page_num}, img {img_index}: {e}")
                finally:
                    if pix_rgb:
                        pix_rgb = None
                    if pix:
                        pix = None
            
            if images_on_page:
                page_images[page_num] = images_on_page
        
        doc.close()
    except Exception as e:
        logger.error(f"PyMuPDF image extraction error: {e}")
    
    return page_images


def _extract_images_with_docling(pdf_path: str, images_dir: Path) -> Dict[int, List[str]]:
    """
    Extract images from PDF using Docling (recommended method).
    
    Args:
        pdf_path: Path to PDF file
        images_dir: Directory to save extracted images
    
    Returns:
        Dict {page_number: [list_of_image_paths]}
    """
    os.makedirs(images_dir, exist_ok=True)
    page_images = {}
    
    # Return empty dict because images are loaded on demand
    logger.info("Extracting images with Docling")
    
    return page_images


def load_pdf_with_images(pdf_path: str = str(PDF_PATH), images_dir: Path = None) -> Tuple[List[Document], Dict[int, List[str]]]:
    """
    Load PDF with text and image extraction via Docling and PyMuPDF fallback.
    
    Args:
        pdf_path: Path to PDF file
        images_dir: Image output directory (if None, uses data/images/extracted)
    
    Returns:
        Tuple of:
        - List of Document objects (text per page)
        - Dict {page_number: [list_of_image_paths]}
    """
    if images_dir is None:
        images_dir = IMAGES_DIR
    else:
        images_dir = Path(images_dir)
    
    os.makedirs(images_dir, exist_ok=True)
    documents = []
    page_images = {}
    
    logger.info(f"Using Docling to process PDF: {pdf_path}")
    
    # Processing options
    pdf_options = PdfPipelineOptions(
        images_scale=2,
        generate_page_images=True
    )
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_options)
        }
    )
    
    try:
        result = converter.convert(pdf_path)
        doc = result.document
        page_numbers = sorted(getattr(doc, "pages", {}).keys())

        # Extract text page-by-page (correct for DoclingDocument)
        total_pages = len(page_numbers)
        for idx, page_no in enumerate(page_numbers):
            page_text = ""
            if hasattr(doc, "export_to_markdown"):
                page_text = doc.export_to_markdown(page_no=page_no) or ""
            else:
                # Fallback branch for non-standard object shape
                page_entry = getattr(doc, "pages", {}).get(page_no)
                page_text = _get_page_markdown(page_entry)

            # Per-page fallback when markdown is unavailable
            if not page_text.strip():
                logger.warning(f"Empty Docling text for page {idx}, falling back to PyMuPDF.")
                try:
                    fitz_doc = fitz.open(pdf_path)
                    page_text = fitz_doc[idx].get_text("text") or ""
                    fitz_doc.close()
                except Exception as e:
                    logger.warning(f"Fallback text extraction error for page {idx}: {e}")

            metadata = {
                "source": pdf_path,
                "page": idx,
                "total_pages": total_pages
            }
            documents.append(Document(page_content=page_text, metadata=metadata))
    except Exception as e:
        logger.error(f"Docling conversion error, using PyMuPDF fallback: {e}")
        documents = _extract_text_with_fitz(pdf_path)
    
    # Extract images with fallback method
    page_images = _extract_images_with_fitz(pdf_path, images_dir)
    
    return documents, page_images
