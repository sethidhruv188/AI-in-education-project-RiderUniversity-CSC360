import os
import textwrap
import json
import fitz  # PyMuPDF (Still needed to extract text for RAG search)
import faiss
import pickle
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# *** CHANGE THIS TO SWITCH ASSIGNMENTS ***
TARGET_ASSIGNMENT = "Module_5_Discussion"

# Paths
BASE_DIR = os.path.join("assignments", TARGET_ASSIGNMENT)
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
RUBRIC_FILE = os.path.join(BASE_DIR, "rubric.txt")
SOLUTION_FILE = os.path.join(BASE_DIR, "solution.pdf")

if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    safety_settings=[
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
    ],
generation_config={"temperature": 0.4}
)

# Load Knowledge Base
try:
    index = faiss.read_index("faiss_index.bin")
    with open("text_metadata.pkl", "rb") as f:
        text_metadata = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Knowledge Base Loaded.")
except:
    print("⚠️ Knowledge Base not found. Run 'process_documents.py' first.")
    index = None


# --- 2. HELPER: Text Extractor (For RAG Query ONLY) ---
def extract_text_for_search(file_path):
    """
    We still need to read text to search the slides (RAG).
    If the file is an image-pdf, we return None and will use the Rubric as the search query.
    """
    try:
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            text = "".join([page.get_text() for page in doc])
            return text.strip() if text.strip() else None
    except:
        return None


# --- 3. RAG SEARCH ---
def find_relevant_slides(query, k=3):
    if index is None or not query: return ""
    query_vec = embedder.encode([query]).astype('float32')
    _, indices = index.search(query_vec, k)
    return "\n\n".join([text_metadata[i]['content'] for i in indices[0]])


# --- 4. VISION UPLOADER (The Image Feature) ---
def upload_to_gemini(path, mime_type="application/pdf"):
    """Uploads file to Google so the AI can SEE tables/screenshots."""
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None
    return genai.upload_file(path, mime_type=mime_type)


# --- 5. GRADING PROMPT ---
# --- 5. GRADING PROMPT ---
# --- 6. PROMPT ---
def build_grading_instruction(rubric_text, slides_text):
    return f"""
    You are an expert AI Teaching Assistant. 
    **TASK:** Visually compare the Student Submission to the Instructor Solution.

    **HANDWRITING PROTOCOL (CRITICAL):**
    - Students may submit handwritten tables or code.
    - If a handwritten number is ambiguous (e.g., a '0' looks like a '6' or '1'), look for **context clues** in the surrounding text to confirm.
    - **Benefit of the Doubt:** If the student's written logic is correct but one digit in the final string looks slightly off due to handwriting quality (e.g., '101101' vs '110101'), assume it is a handwriting artifact and **DO NOT deduct points** unless it is clearly a calculation error.
    - Cross-reference the "Length" they calculated with the string they wrote. If they match the solution, the string is likely correct.

    <Rubric>
    {rubric_text}
    </Rubric>

    <Slides Context>
    {slides_text}
    </Slides Context>

    **OUTPUT INSTRUCTIONS:**
    - Use keys "Q1", "Q2", "Q3".
    - "points" must be "X/Y".

    **OUTPUT (JSON ONLY):**
    {{
        "score": "X/10",
        "feedback_summary": "Summary...",
        "detailed_grading": [
            {{ "question": "Q1", "points": "X/Y", "reason": "..." }},
            {{ "question": "Q2", "points": "X/Y", "reason": "..." }},
            {{ "question": "Q3", "points": "X/Y", "reason": "..." }}
        ]
    }}
    """
# --- 6. MAIN PROCESSOR ---
def process_assignment_folder():
    print(f"\n📂 TARGET ASSIGNMENT: {TARGET_ASSIGNMENT}")

    if not os.path.exists(RUBRIC_FILE) or not os.path.exists(SOLUTION_FILE):
        print(f"❌ Missing Rubric or Solution in: {BASE_DIR}")
        return

    # 1. Read Rubric & Upload Solution (Vision)
    with open(RUBRIC_FILE, 'r', encoding='utf-8') as f:
        rubric_text = f.read()
    solution_file_obj = upload_to_gemini(SOLUTION_FILE)

    # 2. Grade Submissions
    for filename in os.listdir(SUBMISSIONS_DIR):
        if not (filename.endswith(".pdf") or filename.endswith(".txt")): continue

        submission_path = os.path.join(SUBMISSIONS_DIR, filename)
        print(f"\n📝 Processing: {filename}")

        # A. RAG Search (Hybrid Strategy)
        # Try to search using student text. If student file is image-only, search using Rubric text.
        search_query = extract_text_for_search(submission_path) or rubric_text
        slides_context = find_relevant_slides(search_query[:500])

        # B. Visual Upload (The Upgrade)
        student_file_obj = upload_to_gemini(submission_path)
        if not student_file_obj: continue

        # C. Call AI (Multimodal)
        try:
            prompt = build_grading_instruction(rubric_text, slides_context)
            response = llm.generate_content([
                prompt,
                "--- INSTRUCTOR SOLUTION ---", solution_file_obj,
                "--- STUDENT SUBMISSION ---", student_file_obj
            ])

            result = json.loads(response.text.replace("```json", "").replace("```", "").strip())

            # --- IMPROVED PRINTING WITH TEXT WRAP ---
            wrapper = textwrap.TextWrapper(width=80, subsequent_indent="     ")

            print("-" * 60)
            print(f"✅ Result: {result.get('student_name', 'Unknown')} | Score: {result.get('score', 'N/A')}")

            # Wrap the Main Summary
            summary = result.get('feedback_summary', 'No summary provided.')
            print("\n📝 Summary:")
            print(textwrap.fill(summary, width=80))
            print("")  # Empty line for spacing

            # Wrap the Detailed Reasons
            for item in result.get('detailed_grading', []):
                q_text = f"• {item.get('question')}: {item.get('points')}"
                reason_text = f"  Reason: {item.get('reason')}"

                print(q_text)
                print(wrapper.fill(reason_text))
                print("")  # Empty line between questions
            print("-" * 60)
        except Exception as e:
            print(f"❌ Failed: {e}")


# EXECUTE DIRECTLY
process_assignment_folder()