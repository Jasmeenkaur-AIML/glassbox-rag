import re
import pdfplumber
import chromadb

PDF_PATH = "data/notes.pdf"
CHUNK_SIZE = 500


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=1.5, y_tolerance=3)
            if page_text:
                text += page_text + "\n"
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE):
    # Split into sentences first, so we never cut mid-word or mid-sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def main():
    print("Extracting text from PDF...")
    text = extract_text(PDF_PATH)
    print(f"Extracted {len(text)} characters.")

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks.")

    print("Setting up Chroma and storing chunks...")
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("notes")
    except Exception:
        pass

    collection = client.create_collection("notes")

    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    print(f"Done! Stored {len(chunks)} chunks in Chroma collection 'notes'.")


if __name__ == "__main__":
    main()