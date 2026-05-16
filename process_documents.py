import os
import time
import fitz
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
from gcs_helper import download_from_gcs, upload_to_gcs
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')  # Transcription only — 2.0-flash is sufficient
else:
    print("Warning: No GOOGLE_API_KEY found.")
    model = None

# --- MODULE-LEVEL CACHES ---
# Embedder: ~90MB model, takes 2-3s to load. Cache it once per container lifetime.
_embedder = None

# KB cache: avoids re-downloading FAISS index from GCS on every request.
# Key: "course_id_assignment_id", Value: (index, text_metadata, embedder)
# For batch grading 30 students, this means 1 GCS download instead of 30.
_kb_cache = {}


def get_embedder():
    """Return the shared SentenceTransformer instance, loading it only once."""
    global _embedder
    if _embedder is None:
        print("Loading SentenceTransformer embedder (one-time)...")
        _embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedder


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

def build_knowledge_base(course_id: str, assignment_id: str, materials_dir: str):
    """
    Build (or rebuild) the FAISS knowledge base and upload it to Google Cloud Storage.
    """
    # Define temporary server paths
    tmp_index_file = f"/tmp/{course_id}_{assignment_id}_faiss_index.bin"
    tmp_metadata_file = f"/tmp/{course_id}_{assignment_id}_text_metadata.pkl"

    all_texts = []
    text_metadata = []

    embedder = get_embedder()  # Cached — no reload cost

    for filename in os.listdir(materials_dir):
        file_path = os.path.join(materials_dir, filename)
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
                    text_metadata.append({
                        "source": filename,
                        "content": chunk,
                        "course_id": course_id,
                        "assignment_id": assignment_id
                    })
                print(f"Processed {filename} ({len(chunks)} chunks)")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not all_texts:
        return {"status": "error", "message": "No text found in materials folder"}

    # Build FAISS
    embeddings = embedder.encode(all_texts, show_progress_bar=False)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))

    # Save to /tmp
    faiss.write_index(index, tmp_index_file)
    with open(tmp_metadata_file, "wb") as f:
        pickle.dump(text_metadata, f)

    # UPLOAD THE BRAIN TO GOOGLE CLOUD STORAGE!
    gcs_kb_dir = f"courses/{course_id}/assignments/{assignment_id}/knowledge_base"
    upload_to_gcs(tmp_index_file, f"{gcs_kb_dir}/faiss_index.bin")
    upload_to_gcs(tmp_metadata_file, f"{gcs_kb_dir}/text_metadata.pkl")

    # Clean up server RAM (/tmp)
    os.remove(tmp_index_file)
    os.remove(tmp_metadata_file)
    for filename in os.listdir(materials_dir):
        os.remove(os.path.join(materials_dir, filename))
    os.rmdir(materials_dir)

    return {
        "status": "success",
        "course_id": course_id,
        "assignment_id": assignment_id,
        "chunks_indexed": len(all_texts)
    }

def load_knowledge_base(course_id: str, assignment_id: str):
    """
    Load an existing FAISS index for a SPECIFIC ASSIGNMENT from Google Cloud Storage.

    Results are cached in memory (_kb_cache). For batch grading (e.g. 30 students),
    the index is downloaded from GCS exactly once and reused for every submission.
    """
    cache_key = f"{course_id}_{assignment_id}"
    if cache_key in _kb_cache:
        print(f"KB cache hit for {cache_key} — skipping GCS download")
        return _kb_cache[cache_key]

    # GCS Paths
    gcs_kb_dir = f"courses/{course_id}/assignments/{assignment_id}/knowledge_base"
    gcs_index_file = f"{gcs_kb_dir}/faiss_index.bin"
    gcs_metadata_file = f"{gcs_kb_dir}/text_metadata.pkl"

    # Temporary Server Paths
    tmp_index_file = f"/tmp/{course_id}_{assignment_id}_faiss_index.bin"
    tmp_metadata_file = f"/tmp/{course_id}_{assignment_id}_text_metadata.pkl"

    # Download the brain from the cloud to the server
    index_downloaded = download_from_gcs(gcs_index_file, tmp_index_file)
    metadata_downloaded = download_from_gcs(gcs_metadata_file, tmp_metadata_file)

    if not index_downloaded or not metadata_downloaded:
        return None, None, None

    index = faiss.read_index(tmp_index_file)
    with open(tmp_metadata_file, "rb") as f:
        text_metadata = pickle.load(f)

    embedder = get_embedder()  # Cached — no reload cost

    result = (index, text_metadata, embedder)
    _kb_cache[cache_key] = result  # Store for all future requests this session
    return result