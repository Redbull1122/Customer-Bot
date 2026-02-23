import re
from src.pdf_loader import load_pdf_with_images
from src.text_splitter import split_documents
from src.vector_store import clear_index, add_documents_to_vector_store, get_or_create_index


def clean_text_content(text):
    """Clean text from PDF extraction artifacts."""
    if not text:
        return ""

    # 0. Fix /uniXXXX encoding artifacts (e.g., /uni041F -> P)
    # IMPORTANT: do this BEFORE other processing steps
    def replace_unicode(match):
        try:
            code = int(match.group(1), 16)
            return chr(code)
        except (ValueError, OverflowError):
            return match.group(0)  # Keep original token if conversion fails

    text = re.sub(r'/uni([0-9A-Fa-f]{4})', replace_unicode, text)

    # Also handle the no-slash variant (uni041F)
    text = re.sub(r'\buni([0-9A-Fa-f]{4})\b', replace_unicode, text)

    # 1. Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    text = text.replace('\u00a0', ' ')

    # 2. Collapse excessive newlines (keep at most 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 3. Remove consecutive duplicate lines
    lines = text.split('\n')
    deduped_lines = []
    prev_line = None

    for line in lines:
        stripped = line.strip()
        # Skip line if it duplicates the previous non-empty line
        if stripped and stripped == prev_line:
            continue

        if stripped:
            prev_line = stripped

        deduped_lines.append(line)

    text = '\n'.join(deduped_lines)

    # 4. Collapse extra spaces (more than two in a row)
    text = re.sub(r' {2,}', ' ', text)

    # 5. Trim spaces at line boundaries
    lines = text.split('\n')
    text = '\n'.join(line.strip() for line in lines)

    return text.strip()


def main():
    print("Starting PDF reindex...")

    # Step 1: Verify index availability
    print("\nStep 1: Checking index...")
    index = get_or_create_index()
    if not index:
        print("Failed to create Pinecone index. Check your API key.")
        return

    # Step 2: Clear existing index
    print("\nStep 2: Clearing existing index...")
    clear_index()

    # Step 3: Re-create index
    print("\nStep 3: Creating index...")
    index = get_or_create_index()
    if not index:
        print("Failed to create Pinecone index.")
        return

    print("\nStep 4: Loading PDF and extracting images...")
    try:
        documents, images = load_pdf_with_images()
        print(f"Loaded {len(documents)} pages")
        print(f"Found images on {len(images)} pages")
    except Exception as e:
        print(f"Error while loading PDF: {e}")
        return

    # Step 5: Create chunks
    print("\nStep 5: Creating text chunks...")
    try:
        chunks = split_documents(documents, chunk_size=500)
        if not chunks:
            print("Failed to create chunks")
            return
        print(f"Created {len(chunks)} chunks")
    except Exception as e:
        print(f"Error while creating chunks: {e}")
        return

    # Step 6: Clean text and attach metadata
    print("\nStep 6: Cleaning text and attaching metadata...")
    chunks_with_images = 0

    # Show sample BEFORE cleanup
    if chunks:
        print("\nSample text BEFORE cleanup:")
        print(chunks[0].page_content[:200])

    for chunk in chunks:
        # Clean text
        original = chunk.page_content
        cleaned = clean_text_content(original)
        chunk.page_content = cleaned

        # Attach images to metadata
        page_num = chunk.metadata.get('page')
        if page_num is not None and page_num in images:
            chunk.metadata['images'] = images[page_num]
            chunks_with_images += 1

    # Show sample AFTER cleanup
    if chunks:
        print("\nSample text AFTER cleanup:")
        print(chunks[0].page_content[:200])

    print(f"\nProcessed {len(chunks)} chunks")
    print(f"{chunks_with_images} chunks contain images")

    # Step 7: Upload to Pinecone
    print("\nStep 7: Uploading data to Pinecone...")
    try:
        success = add_documents_to_vector_store(chunks)
        if success:
            print("\nReindex completed successfully.")
            print(f"Uploaded {len(chunks)} chunks")
            print(f"{chunks_with_images} chunks contain images")
        else:
            print("Error uploading to Pinecone")
    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()
