import os
import json
import fitz
import google.generativeai as genai
from dotenv import load_dotenv
from gcs_helper import download_from_gcs

# process_documents is now our shared KB loader
from process_documents import load_knowledge_base

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)


def get_llm():
    """
    Returns a configured Gemini model instance.
    Safety filters are relaxed for academic CS content
    (e.g. 'kill process', 'segfault', 'exploit') which would
    otherwise get flagged as harmful content.
    """
    return genai.GenerativeModel(
        model_name='gemini-2.5-flash',
        safety_settings=[
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
        ],
        generation_config={"temperature": 0.4}
    )


# --- HELPERS ---

def extract_text_for_search(file_path: str):
    """Extract plain text from a PDF or TXT for use as a RAG query."""
    try:
        if file_path.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_path.lower().endswith('.pdf'):
            doc = fitz.open(file_path)
            text = "".join([page.get_text() for page in doc])
            return text.strip() if text.strip() else None
    except Exception:
        return None


def requires_vision(file_path: str):
    """
    Checks if a PDF requires the expensive Vision API by scanning for images
    or a lack of selectable text (which implies a scanned document).
    """
    if not file_path.lower().endswith('.pdf'):
        return False  # Text files (.txt) never need vision

    try:
        doc = fitz.open(file_path)
        text_length = 0
        image_count = 0

        for page in doc:
            text_length += len(page.get_text().strip())
            # get_images() returns a list of images on the page
            image_count += len(page.get_images(full=True))

        # If it has images, OR if it has almost no text (e.g., a scanned handwritten page)
        if image_count > 0 or text_length < 100:
            return True

        return False  # It is a pure text PDF!
    except Exception as e:
        print(f"Error scanning {file_path} for vision: {e}")
        return True  # If PyMuPDF fails, default to safety (Vision)

def find_relevant_slides(query: str, index, text_metadata, embedder, k=3):
    """Query the course-specific FAISS index for relevant slide content."""
    if index is None or not query:
        return ""
    query_vec = embedder.encode([query]).astype('float32')
    _, indices = index.search(query_vec, k)
    return "\n\n".join([text_metadata[i]['content'] for i in indices[0]])


def upload_to_gemini(path: str, mime_type="application/pdf"):
    """Upload a file to Gemini's file API so it can visually process tables/screenshots."""
    if not os.path.exists(path):
        return None
    return genai.upload_file(path, mime_type=mime_type)


def build_grading_prompt(rubric_text: str, slides_context: str):
    """
    The master grading prompt. Enforces the hierarchy:
    Rubric > Instructor Solution > Course Slides > Student Submission

    The handwriting protocol handles ambiguous scanned submissions
    by giving benefit of the doubt when surrounding logic is correct.
    """
    return f"""
You are an expert AI Teaching Assistant grading for a university course.

TASK: Compare the Student Submission against the Instructor Solution.
Grade strictly according to the Rubric. Use the Course Slides for context
on course-specific rules (e.g. custom method signatures, data structure constraints).

HANDWRITING PROTOCOL:
- If a handwritten digit is ambiguous, use surrounding context to infer.
- If the student's logic is correct but one symbol looks off due to handwriting,
  do NOT deduct — flag for human review instead.

<Rubric>
{rubric_text}
</Rubric>

<Course Slides Context>
{slides_context}
</Course Slides Context>

OUTPUT: Return ONLY valid JSON. No markdown, no preamble.
{{
    "score": "X/10",
    "feedback_summary": "2-3 sentence overall summary",
    "detailed_grading": [
        {{"question": "Q1", "points": "X/Y", "reason": "..."}},
        {{"question": "Q2", "points": "X/Y", "reason": "..."}},
        {{"question": "Q3", "points": "X/Y", "reason": "..."}}
    ],
    "flagged_for_review": false,
    "flag_reason": ""
}}
"""


def build_feedback_prompt(doc_text: str, rubric_text: str, slides_context: str):
    """
    Pre-submission feedback prompt for the Google Docs 'Get AI Review' button.

    Key constraint: must NOT give answers. Only tells students what is
    correct, what needs improvement, and where to look in course materials.
    This is enforced in the prompt itself.
    """
    return f"""
You are a course-aligned AI reviewer helping a student BEFORE they submit.

Your job is to review their draft and give formative feedback.

STRICT RULES:
1. Do NOT provide answers, solutions, or corrected work.
2. DO point out what looks correct and what may need improvement.
3. For anything that needs improvement, reference the relevant course material
   (e.g. "See Module 3, slide on Big-O notation" or "Review the Stack ADT lecture").
4. Keep feedback concise — 1-2 sentences per item.
5. Do not have a conversation. This is a one-time review.

<Rubric>
{rubric_text}
</Rubric>

<Course Material Context>
{slides_context}
</Course Material Context>

<Student Draft>
{doc_text}
</Student Draft>

OUTPUT: Return ONLY valid JSON. No markdown, no preamble.
{{
    "overall_impression": "Brief 1-sentence overall impression",
    "items": [
        {{
            "section": "Q1 / Introduction / etc.",
            "status": "looks good" | "needs review" | "unclear",
            "comment": "Specific non-answer feedback",
            "reference": "Slide/module reference if applicable, else null"
        }}
    ]
}}
"""


# --- CORE FUNCTIONS called by app.py --

def grade_submission(
        course_id: str,
        assignment_id: str,
        submission_path: str
):
    """
    Grade a single student submission for a given course and assignment using GCS.
    (Note: submission_path is already a /tmp path passed from app.py)
    """
    # Load this course's isolated knowledge base from GCS
    index, text_metadata, embedder = load_knowledge_base(course_id, assignment_id)
    if index is None:
        return {"error": f"No knowledge base found for {course_id} - {assignment_id}."}

    # Define GCS Cloud Paths
    gcs_rubric_file = f"courses/{course_id}/assignments/{assignment_id}/rubric.txt"
    gcs_solution_file = f"courses/{course_id}/assignments/{assignment_id}/solution.pdf"

    # Define /tmp Local Paths
    tmp_rubric_file = f"/tmp/{course_id}_{assignment_id}_rubric.txt"
    tmp_solution_file = f"/tmp/{course_id}_{assignment_id}_solution.pdf"

    # Download Rubric and Solution from GCS to /tmp
    if not download_from_gcs(gcs_rubric_file, tmp_rubric_file):
        return {"error": "Rubric not found in Google Cloud Storage."}

    # We don't error out if solution is missing, some assignments might not have one
    has_solution = download_from_gcs(gcs_solution_file, tmp_solution_file)

    with open(tmp_rubric_file, 'r', encoding='utf-8') as f:
        rubric_text = f.read()

    # RAG: search course slides using student text (or rubric as fallback)
    search_query = extract_text_for_search(submission_path) or rubric_text
    slides_context = find_relevant_slides(search_query[:500], index, text_metadata, embedder)

    # === SMART ROUTING: INSTRUCTOR SOLUTION ===
    solution_content = "No solution provided."
    if has_solution:
        if requires_vision(tmp_solution_file):
            print("Routing Instructor Solution to Expensive Vision Pipeline...")
            solution_content = upload_to_gemini(tmp_solution_file)
        else:
            print("Routing Instructor Solution to Cheap Text Pipeline...")
            solution_content = extract_text_for_search(tmp_solution_file)

    # === SMART ROUTING: STUDENT SUBMISSION ===
    if requires_vision(submission_path):
        print(f"Routing Student Submission to Vision Pipeline...")
        submission_content = upload_to_gemini(submission_path)
    else:
        print(f"Routing Student Submission to Text Pipeline...")
        submission_content = extract_text_for_search(submission_path)

    if not submission_content:
        return {"error": "Failed to process student submission"}

    llm = get_llm()
    prompt = build_grading_prompt(rubric_text, slides_context)

    response = llm.generate_content([
        prompt,
        "--- INSTRUCTOR SOLUTION ---", solution_content,
        "--- STUDENT SUBMISSION ---", submission_content
    ])

    result = json.loads(
        response.text.replace("```json", "").replace("```", "").strip()
    )
    result["course_id"] = course_id
    result["assignment_id"] = assignment_id
    result["submission_file"] = os.path.basename(submission_path)

    # Clean up the /tmp files to save server memory
    os.remove(tmp_rubric_file)
    if has_solution:
        os.remove(tmp_solution_file)

    return result

def get_presubmission_review(course_id: str, assignment_id: str, doc_text: str):
    """Generate pre-submission formative feedback for a Google Doc draft."""
    index, text_metadata, embedder = load_knowledge_base(course_id, assignment_id)
    if index is None:
        return {"error": f"No knowledge base found for course {course_id}"}

    # Define paths
    gcs_rubric_file = f"courses/{course_id}/assignments/{assignment_id}/rubric.txt"
    tmp_rubric_file = f"/tmp/{course_id}_{assignment_id}_review_rubric.txt"

    # Download Rubric from GCS
    if not download_from_gcs(gcs_rubric_file, tmp_rubric_file):
        return {"error": "Rubric not found in Google Cloud Storage."}

    with open(tmp_rubric_file, 'r', encoding='utf-8') as f:
        rubric_text = f.read()

    # Clean up /tmp
    os.remove(tmp_rubric_file)

    slides_context = find_relevant_slides(doc_text[:500], index, text_metadata, embedder)

    llm = get_llm()
    prompt = build_feedback_prompt(doc_text, rubric_text, slides_context)

    response = llm.generate_content([prompt])

    result = json.loads(
        response.text.replace("```json", "").replace("```", "").strip()
    )
    result["course_id"] = course_id
    result["assignment_id"] = assignment_id

    return result

def grade_all_submissions(course_id: str, assignment_id: str):
    """
    Batch grade all submissions in a course/assignment folder.
    Skips already-graded files using a history log.

    Returns:
        list of grading result dicts
    """
    submissions_dir = os.path.join(
        "courses", course_id, "assignments", assignment_id, "submissions"
    )
    history_file = os.path.join(
        "courses", course_id, "assignments", assignment_id, "grading_history.json"
    )

    if not os.path.exists(submissions_dir):
        return [{"error": f"Submissions folder not found: {submissions_dir}"}]

    # Load history to skip already-graded files
    history = {}
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            history = json.load(f)

    results = []
    for filename in os.listdir(submissions_dir):
        if not (filename.endswith(".pdf") or filename.endswith(".txt")):
            continue
        if filename in history:
            print(f"Skipping already graded: {filename}")
            results.append(history[filename])
            continue

        submission_path = os.path.join(submissions_dir, filename)
        print(f"Grading: {filename}")

        try:
            result = grade_submission(course_id, assignment_id, submission_path)
            history[filename] = result
            results.append(result)
        except Exception as e:
            error_result = {"file": filename, "error": str(e)}
            results.append(error_result)

    # Save updated history
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

    return results