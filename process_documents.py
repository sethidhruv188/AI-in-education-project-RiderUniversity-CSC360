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

# Configure AI for "Fallback" transcription
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("⚠️ Warning: No GOOGLE_API_KEY found. Image-based PDFs cannot be processed.")
    model = None


# --- HELPERS ---
def transcribe_with_gemini(file_path):
    """Uploads a 'scanned' or image-heavy PDF to Gemini to extract text."""
    if not model: return ""
    print(f"  AI Transcription engaged for: {os.path.basename(file_path)}...")
    try:
        file = genai.upload_file(file_path, mime_type="application/pdf")
        time.sleep(2)
        response = model.generate_content([file, "Transcribe all the text from this document exactly as it appears."])
        return response.text
    except Exception as e:
        print(f" ❌ AI Transcription failed: {e}")
        return ""


def read_pdf(file_path):
    """Smart Reader: Tries local extraction, falls back to AI for scans."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()

    # Heuristic: If < 100 chars, it's likely an image scan.
    if len(text.strip()) < 100:
        print(f"   ⚠️ Low text detected in {os.path.basename(file_path)}. Switching to Visual Analysis...")
        ai_text = transcribe_with_gemini(file_path)
        if ai_text: return ai_text

    return text


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# --- MAIN BUILDER ---
def process_and_build_db():
    all_texts = []
    text_metadata = []

    print(f"📂 Scanning {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"⚠️ Created {DATA_DIR} folder. Add slides here.")
        return

    print("🔌 Loading Embedding Model...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                text = read_pdf(file_path)
            elif filename.lower().endswith(".txt"):
                text = read_txt(file_path)

            if text:
                chunks = [c for c in text.split('\n\n') if len(c) > 50]
                for chunk in chunks:
                    all_texts.append(chunk)
                    text_metadata.append({"source": filename, "content": chunk})
                print(f"   ✅ Processed {filename} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

    if not all_texts:
        print("⚠️ No text found. Add slides to 'course_materials'.")
        return

    print("Generating Vector Embeddings...")
    embeddings = embedder.encode(all_texts, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    faiss.write_index(index, INDEX_FILE)
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(text_metadata, f)

    print("Knowledge Base Built!")


# EXECUTE DIRECTLY
process_and_build_db()