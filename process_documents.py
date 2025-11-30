import os
import time
import fitz  # PyMuPDF
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# --- SETUP ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
DATA_DIR = "course_materials"
INDEX_FILE = "faiss_index.bin"
METADATA_FILE = "text_metadata.pkl"

# Configure AI for the "Fallback" transcription
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    # Using Flash because it's fast and handles large docs easily
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("⚠️ Warning: No GOOGLE_API_KEY found. Image-based PDFs cannot be processed.")
    model = None


# --- HELPERS ---

def transcribe_with_gemini(file_path):
    """
    Uploads a 'scanned' or image-heavy PDF to Gemini to extract text.
    """
    if not model: return ""

    print(f"   🤖 AI Transcription engaged for: {os.path.basename(file_path)}...")
    try:
        # 1. Upload
        file = genai.upload_file(file_path, mime_type="application/pdf")

        # 2. Wait for processing (usually instant for docs)
        # (Gemini handles this internally mostly, but good to be safe)
        time.sleep(2)

        # 3. Prompt
        response = model.generate_content([file,
                                           "Transcribe all the text from this document exactly as it appears. Include text from tables and diagrams."])

        return response.text
    except Exception as e:
        print(f"   ❌ AI Transcription failed: {e}")
        return ""


def read_pdf(file_path):
    """
    Smart Reader:
    1. Tries local text extraction.
    2. If text is suspiciously empty (images), calls Gemini to transcribe.
    """
    # Attempt 1: Fast Local Extraction
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # Check if the PDF was just images (Scanned)
    # Heuristic: If a whole file has < 100 characters of text, it's likely an image scan.
    if len(text.strip()) < 100:
        print(f"   ⚠️ Low text detected in {os.path.basename(file_path)}. Switching to Visual Analysis...")
        ai_text = transcribe_with_gemini(file_path)
        if ai_text:
            return ai_text

    return text


def read_txt(file_path):
    """Extracts text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# --- MAIN BUILDER ---
def process_and_build_db():
    all_texts = []
    text_metadata = []

    print(f"📂 Scanning {DATA_DIR}...")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"⚠️ Created {DATA_DIR} folder. Please put your PDF slides inside and run this again.")
        return

    # Initialize Embedder
    print("🔌 Loading Embedding Model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        text = ""

        try:
            # 1. Check for PDF
            if filename.lower().endswith(".pdf"):
                text = read_pdf(file_path)

            # 2. Check for TXT
            elif filename.lower().endswith(".txt"):
                text = read_txt(file_path)

            # 3. Process the text if we found any
            if text:
                # Split huge text into smaller chunks (approx paragraphs)
                # We filter out very short chunks (len < 50) to avoid noise
                chunks = [c for c in text.split('\n\n') if len(c) > 50]

                for chunk in chunks:
                    all_texts.append(chunk)
                    text_metadata.append({"source": filename, "content": chunk})

                print(f"   ✅ Processed {filename} ({len(chunks)} chunks)")

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

    if not all_texts:
        print("⚠️ No text found. Add PDF slides to 'course_materials' folder.")
        return

    # Create Embeddings
    print("🧠 Generating Vector Embeddings...")
    embeddings = embedder.encode(all_texts, show_progress_bar=True)

    # Save to FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(text_metadata, f)

    print("🎉 Knowledge Base Built! (Visual & Text Support Active)")


if __name__ == "__main__":
    process_and_build_db()