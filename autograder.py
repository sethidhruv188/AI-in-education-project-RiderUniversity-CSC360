import os
import json
import fitz
import google.generativeai as genai
from dotenv import load_dotenv

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
    Grade a single student submission for a given course and assignment.

    Args:
        course_id: e.g. "CSC130_Spring2026"
        assignment_id: e.g. "Module_5_Discussion"
        submission_path: full path to student's PDF or TXT file

    Returns:
        dict with score, feedback, and grading details
    """
    # Load this course's isolated knowledge base
    index, text_metadata, embedder = load_knowledge_base(course_id)
    if index is None:
        return {"error": f"No knowledge base found for course {course_id}. Run /build-kb first."}

    # Locate rubric and solution for this assignment
    assignment_dir = os.path.join("courses", course_id, "assignments", assignment_id)
    rubric_file = os.path.join(assignment_dir, "rubric.txt")
    solution_file = os.path.join(assignment_dir, "solution.pdf")

    if not os.path.exists(rubric_file):
        return {"error": f"Rubric not found at {rubric_file}"}
    if not os.path.exists(solution_file):
        return {"error": f"Solution not found at {solution_file}"}

    with open(rubric_file, 'r', encoding='utf-8') as f:
        rubric_text = f.read()

    # RAG: search course slides using student text (or rubric as fallback)
    search_query = extract_text_for_search(submission_path) or rubric_text
    slides_context = find_relevant_slides(search_query[:500], index, text_metadata, embedder)

    # Upload both PDFs for visual grading (Gemini sees tables, screenshots, handwriting)
    solution_obj = upload_to_gemini(solution_file)
    submission_obj = upload_to_gemini(submission_path)

    if not submission_obj:
        return {"error": "Failed to upload student submission to Gemini"}

    llm = get_llm()
    prompt = build_grading_prompt(rubric_text, slides_context)

    response = llm.generate_content([
        prompt,
        "--- INSTRUCTOR SOLUTION ---", solution_obj,
        "--- STUDENT SUBMISSION ---", submission_obj
    ])

    result = json.loads(
        response.text.replace("```json", "").replace("```", "").strip()
    )
    result["course_id"] = course_id
    result["assignment_id"] = assignment_id
    result["submission_file"] = os.path.basename(submission_path)

    return result


def get_presubmission_review(
    course_id: str,
    assignment_id: str,
    doc_text: str
):
    """
    Generate pre-submission formative feedback for a student's Google Doc draft.

    Args:
        course_id: e.g. "CSC130_Spring2026"
        assignment_id: e.g. "Module_5_Discussion"
        doc_text: plain text content of the student's Google Doc

    Returns:
        dict with per-section feedback and slide references
    """
    index, text_metadata, embedder = load_knowledge_base(course_id)
    if index is None:
        return {"error": f"No knowledge base found for course {course_id}"}

    assignment_dir = os.path.join("courses", course_id, "assignments", assignment_id)
    rubric_file = os.path.join(assignment_dir, "rubric.txt")

    if not os.path.exists(rubric_file):
        return {"error": f"Rubric not found at {rubric_file}"}

    with open(rubric_file, 'r', encoding='utf-8') as f:
        rubric_text = f.read()

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