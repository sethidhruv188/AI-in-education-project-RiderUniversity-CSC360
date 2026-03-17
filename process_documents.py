import os
import time
import fitz
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    print("Warning: No GOOGLE_API_KEY found.")
    model = None


def transcribe_with_gemini(file_path):
    if not model:
        return ""
    try:
        file = genai.upload_file(file_path, mime_type="application/pdf")
        time.sleep(2)
        response = model.generate_content(
            [file, "Transcribe all the text from this document exactly as it appears."]
        )
        return response.text
    except Exception as e:
        print(f"AI Transcription failed: {e}")
        return ""


def read_pdf(file_path):
    doc = fitz.open(file_path)
    text = "".join([page.get_text() for page in doc])
    if len(text.strip()) < 100:
        ai_text = transcribe_with_gemini(file_path)
        if ai_text:
            return ai_text
    return text


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


# --- KEY CHANGE: DATA_DIR is now a parameter, not hardcoded ---
# Each course gets its own isolated knowledge base stored under:
#   courses/{course_id}/knowledge_base/faiss_index.bin
#   courses/{course_id}/knowledge_base/text_metadata.pkl
#
# This means CSC 130 and CSC 350 never share embeddings — zero cross-contamination.

def build_knowledge_base(course_id: str, materials_dir: str = None):
    """
    Build (or rebuild) the FAISS knowledge base for a specific course.

    Args:
        course_id: unique identifier like "CSC130_Spring2026"
        materials_dir: path to folder containing slides/PDFs for this course.
                       Defaults to courses/{course_id}/materials/
    Returns:
        dict with status and chunk count.
    """
    if materials_dir is None:
        materials_dir = os.path.join("courses", course_id, "materials")

    # Where we save this course's knowledge base
    kb_dir = os.path.join("courses", course_id, "knowledge_base")
    os.makedirs(kb_dir, exist_ok=True)

    index_file = os.path.join(kb_dir, "faiss_index.bin")
    metadata_file = os.path.join(kb_dir, "text_metadata.pkl")

    if not os.path.exists(materials_dir):
        os.makedirs(materials_dir, exist_ok=True)
        return {"status": "error", "message": f"No materials folder found at {materials_dir}"}

    all_texts = []
    text_metadata = []

    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    for filename in os.listdir(materials_dir):
        file_path = os.path.join(materials_dir, filename)
        text = ""
        try:
            if filename.lower().endswith(".pdf"):
                text = read_pdf(file_path)
            elif filename.lower().endswith(".txt"):
                text = read_txt(file_path)

            if text:
                # Split into meaningful chunks — paragraphs or slide sections
                chunks = [c for c in text.split('\n\n') if len(c) > 50]
                for chunk in chunks:
                    all_texts.append(chunk)
                    text_metadata.append({
                        "source": filename,
                        "content": chunk,
                        "course_id": course_id
                    })
                print(f"Processed {filename} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_texts:
        return {"status": "error", "message": "No text found in materials folder"}

    embeddings = embedder.encode(all_texts, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    faiss.write_index(index, index_file)
    with open(metadata_file, "wb") as f:
        pickle.dump(text_metadata, f)

    return {
        "status": "success",
        "course_id": course_id,
        "chunks_indexed": len(all_texts),
        "index_path": index_file
    }


def load_knowledge_base(course_id: str):
    """
    Load an existing FAISS index for a course.
    Returns (index, text_metadata, embedder) or (None, None, None) if not found.
    """
    kb_dir = os.path.join("courses", course_id, "knowledge_base")
    index_file = os.path.join(kb_dir, "faiss_index.bin")
    metadata_file = os.path.join(kb_dir, "text_metadata.pkl")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        return None, None, None

    index = faiss.read_index(index_file)
    with open(metadata_file, "rb") as f:
        text_metadata = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    return index, text_metadata, embedder