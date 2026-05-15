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


def get_llm(task: str = "grading"):
    """
    Returns a configured Gemini model instance.
    Safety filters are relaxed for academic CS content
    (e.g. 'kill process', 'segfault', 'exploit') which would
    otherwise get flagged as harmful content.

    task="grading"  -> gemini-2.0-flash (cheaper, fully capable for rubric matching)
    task="review"   -> gemini-2.5-flash (better reasoning for nuanced student feedback)
    """
    model_name = 'gemini-2.0-flash' if task == "grading" else 'gemini-2.5-flash'
    return genai.GenerativeModel(
        model_name=model_name,
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


def build_submission_parts(file_path: str):
    """
    Page-level smart router. Replaces the old requires_vision() + upload_to_gemini() pair.

    For each page in the PDF:
      - If the page has extractable text AND no images → free text extraction
      - If the page is image-only or has embedded images → goes to Vision API

    Only the pages that actually need vision are uploaded, not the whole document.
    This preserves page order so Gemini always sees content in context
    (e.g. "see diagram below" on page 2 is followed immediately by the diagram on page 3).

    Returns a list of content parts ready to pass directly to llm.generate_content().
    For .txt files, returns a plain string in a list.
    """
    # TXT files: never need vision
    if not file_path.lower().endswith('.pdf'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [f.read()]
        except Exception:
            return []

    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Failed to open PDF {file_path}: {e}")
        return []

    text_parts = []       # Accumulated text content from cheap pages
    vision_page_nums = [] # Pages that need vision

    for page_num, page in enumerate(doc):
        page_text = page.get_text().strip()
        has_images = len(page.get_images(full=True)) > 0

        if has_images or len(page_text) < 50:
            # This page needs vision — mark it
            vision_page_nums.append(page_num)
            # Flush any accumulated text before this vision page so order is preserved
            if text_parts:
                yield_text = "\n\n".join(text_parts)
                text_parts = []
        else:
            # Pure text page — free
            text_parts.append(f"[Page {page_num + 1}]\n{page_text}")

    # Flush any remaining text pages
    parts = []
    if text_parts and not vision_page_nums:
        # All pages were text — simple case, no vision needed at all
        parts.append("\n\n".join(text_parts))
        return parts

    # Mixed doc: rebuild parts in page order
    text_buffer = []
    vision_set = set(vision_page_nums)

    for page_num, page in enumerate(doc):
        page_text = page.get_text().strip()
        has_images = len(page.get_images(full=True)) > 0

        if page_num not in vision_set:
            text_buffer.append(f"[Page {page_num + 1}]\n{page_text}")
        else:
            # Flush text buffer before this vision page
            if text_buffer:
                parts.append("\n\n".join(text_buffer))
                text_buffer = []
            parts.append(f"[Page {page_num + 1} — visual content below]")

    # Flush any trailing text
    if text_buffer:
        parts.append("\n\n".join(text_buffer))

    # Now upload ONLY the vision pages as a separate mini-PDF
    if vision_page_nums:
        writer = fitz.open()
        for pg in vision_page_nums:
            writer.insert_pdf(doc, from_page=pg, to_page=pg)
        tmp_vision_path = file_path.replace(".pdf", "_vision_pages.pdf")
        writer.save(tmp_vision_path)
        uploaded = genai.upload_file(tmp_vision_path, mime_type="application/pdf")
        parts.append(uploaded)
        # Clean up the mini-PDF
        try:
            os.remove(tmp_vision_path)
        except Exception:
            pass

        vision_count = len(vision_page_nums)
        total_pages = len(doc)
        print(f"Vision API used for {vision_count}/{total_pages} pages "
              f"(saved ~{100 - int(vision_count/total_pages*100)}% vision cost)")

    return parts


def find_relevant_slides(query: str, index, text_metadata, embedder, k=3):
    """Query the course-specific FAISS index for relevant slide content."""
    if index is None or not query:
        return ""
    query_vec = embedder.encode([query]).astype('float32')
    _, indices = index.search(query_vec, k)
    return "\n\n".join([text_metadata[i]['content'] for i in indices[0]])


def build_grading_prompt(rubric_text: str, slides_context: str):
    """
    The master grading prompt. Enforces the hierarchy:
    Rubric > Instructor Solution > Course Slides > Student Submission

    The handwriting protocol handles ambiguous scanned submissions
    by giving benefit of the doubt when surrounding logic is correct.
    """
    return f"""
You are an expert,fair, and highly capable AI Teaching Assistant grading for a university course. When evaluating student submissions—especially handwritten ones—do not penalize students for messy text extraction, spatial formatting issues, or minor OCR errors. If the student's final answer, conclusion, or core logic matches the provided solution or rubric criteria, assume their underlying methodology is correct unless you find an explicit, undeniable flaw in their written steps. Give the student the benefit of the doubt on formatting.

TASK: Compare the Student Submission against the Instructor Solution.
Grade strictly according to the Rubric. Use the Course Slides for context
on course-specific rules (e.g. custom method signatures, data structure constraints).

HANDWRITING PROTOCOL:
- If a handwritten digit is ambiguous, use surrounding context to infer.
- If the student's logic is correct but some symbols look off due to handwriting,
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

    # === PAGE-LEVEL ROUTING: INSTRUCTOR SOLUTION ===
    # build_submission_parts() extracts text from text pages (free) and
    # only uploads image/blank pages to Vision API (paid). Page order preserved.
    solution_parts = []
    if has_solution:
        solution_parts = build_submission_parts(tmp_solution_file)

    # === PAGE-LEVEL ROUTING: STUDENT SUBMISSION ===
    submission_parts = build_submission_parts(submission_path)

    if not submission_parts:
        return {"error": "Failed to process student submission"}

    llm = get_llm(task="grading")  # Uses cheaper gemini-2.0-flash
    prompt = build_grading_prompt(rubric_text, slides_context)

    # Assemble the full content list: prompt + solution parts + submission parts
    content = [prompt, "--- INSTRUCTOR SOLUTION ---"]
    content += solution_parts if solution_parts else ["No solution provided."]
    content += ["--- STUDENT SUBMISSION ---"]
    content += submission_parts

    response = llm.generate_content(content)

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

    llm = get_llm(task="review")  # Uses smarter gemini-2.5-flash for nuanced feedback
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