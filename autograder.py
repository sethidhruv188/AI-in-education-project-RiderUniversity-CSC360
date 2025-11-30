import os
import json
import faiss
import pickle
import fitz  # PyMuPDF
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    print("❌ Error: GOOGLE_API_KEY not found in .env file")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
llm = genai.GenerativeModel('gemini-2.5-flash')

# Load Knowledge Base
try:
    index = faiss.read_index("faiss_index.bin")
    with open("text_metadata.pkl", "rb") as f:
        text_metadata = pickle.load(f)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Knowledge Base Loaded.")
except:
    print("⚠️ Knowledge Base not found. Please run 'process_documents.py' first.")
    index = None


# --- 2. SMART FILE READER ---
def read_submission(file_path):
    """
    Reads the file based on extension.
    IF .txt -> Reads text
    ELSE IF .pdf -> Extracts text
    """
    if not os.path.exists(file_path):
        return None

    # Logic: If TXT, do this...
    if file_path.lower().endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except:
            return None

    # Logic: Else if PDF, do this...
    elif file_path.lower().endswith('.pdf'):
        try:
            doc = fitz.open(file_path)
            text = ""
            for page in doc:
                text += page.get_text() + "\n"
            return text
        except:
            return None

    else:
        return "Error: Unsupported file format."


# --- 3. RAG SEARCH (Finds Slides) ---
def find_relevant_slides(query, k=3):
    if index is None: return ""
    query_vec = embedder.encode([query]).astype('float32')
    _, indices = index.search(query_vec, k)
    return "\n\n".join([text_metadata[i]['content'] for i in indices[0]])


# --- 4. THE PROMPT LOGIC ---
def build_grading_prompt(student_text, rubric, solution, slides):
    return f"""
    You are an AI Teaching Assistant. Grade this student submission.

    **SOURCES OF TRUTH:**
    1. <Rubric>: How to assign points.
    2. <Instructor Solution>: The factual correct answers.
    3. <Lecture Slides>: Context for definitions.

    **TASK:**
    - Compare <Student Submission> to <Instructor Solution>.
    - If the student contradicts the Solution Key (e.g., wrong calculation), deduct points.
    - If the definition is slightly different but matches the concept in <Lecture Slides>, give credit.

    **INPUT DATA:**

    <Rubric>
    {rubric}
    </Rubric>

    <Instructor Solution>
    {solution}
    </Instructor Solution>

    <Lecture Slides Context>
    {slides}
    </Lecture Slides Context>

    <Student Submission>
    {student_text}
    </Student Submission>

    **OUTPUT FORMAT (JSON ONLY):**
    {{
        "student_name": "extracted name",
        "score": "X/Total",
        "feedback_summary": "Overall comment...",
        "detailed_grading": [
            {{ "question": "Q1", "points": "X/Y", "reason": "..." }},
            {{ "question": "Q2", "points": "X/Y", "reason": "..." }}
        ]
    }}
    """


# --- 5. MAIN GRADER FUNCTION ---
def grade_assignment(submission_path, rubric_path, solution_path):
    print(f"\n📝 Grading: {submission_path}")

    # Read all files
    student_text = read_submission(submission_path)
    rubric_text = read_submission(rubric_path)
    solution_text = read_submission(solution_path)

    if not student_text:
        print("❌ Error reading submission file.")
        return

    # Get relevant slides (Context)
    # We use the first 500 chars of student text to find relevant topics
    relevant_slides = find_relevant_slides(student_text[:500])

    # Build Prompt & Call AI
    prompt = build_grading_prompt(student_text, rubric_text, solution_text, relevant_slides)

    try:
        response = llm.generate_content(prompt)
        # Clean JSON
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean_json)

        # Print Beautiful Output
        print("-" * 40)
        print(f"Student: {result.get('student_name', 'Unknown')}")
        print(f"Score:   {result.get('score', 'N/A')}")
        print("-" * 40)
        for item in result.get('detailed_grading', []):
            print(f"• {item['question']}: {item['points']}")
            print(f"  Reason: {item['reason']}")
        print("-" * 40)

    except Exception as e:
        print(f"❌ Grading Failed: {e}")
        print("Raw Output:", response.text)


# --- 6. RUN THE TEST ---
if __name__ == "__main__":
    # Define your files
    RUBRIC_FILE = "rubric.txt"
    SOLUTION_FILE = "submissions_to_grade/CSC350-Module 5 Discussion Board Solution.pdf"

    # 1. Test a TXT file
    print("\n--- Test 1: PDF Submission ---")
    grade_assignment("submissions_to_grade/Tanmay_Submission_1.pdf", RUBRIC_FILE, SOLUTION_FILE)

    # 2. Test a PDF file (Make sure you have one!)
    print("\n--- Test 2: PDF Submission ---")
    grade_assignment("submissions_to_grade/Dhruv_Submission_2.pdf", RUBRIC_FILE, SOLUTION_FILE)

    print("\n--- Test 3: PDF Submission ---")
    grade_assignment("submissions_to_grade/Nik_Submission_3.pdf", RUBRIC_FILE, SOLUTION_FILE)