"""
app.py — Flask API for the AI Autograder System
================================================
This is the main backend that Canvas, Google Docs, and the
instructor dashboard all talk to. Every endpoint accepts JSON
and returns JSON.

Run locally:
    python app.py

Run in production (GCP Cloud Run):
    gunicorn app:app --bind 0.0.0.0:8080
"""

from flask import Flask, request, jsonify, render_template
from gcs_helper import upload_to_gcs
import os
import json
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

load_dotenv()


from process_documents import build_knowledge_base
from autograder import grade_submission, get_presubmission_review, grade_all_submissions


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload



# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "service": "AI Autograder API"})


# =============================================================================
# ENDPOINT 1: Build Knowledge Base for a Course
#
# Called when: a professor uploads course materials (slides, PDFs)
# Who calls it: instructor dashboard or Canvas webhook
#
# POST /build-kb
# Body (multipart/form-data):
#   course_id   — e.g. "CSC130_Spring2026"
#   files[]     — one or more PDF/TXT files (lecture slides, notes)
#
# Each course gets its own isolated FAISS index.
# CSC 130 slides never contaminate CSC 350 grading.
# =============================================================================

@app.route("/build-kb", methods=["POST"])
def build_kb():
    course_id = request.form.get("course_id")
    assignment_id = request.form.get("assignment_id")

    if not course_id or not assignment_id:
        return jsonify({"error": "course_id and assignment_id are required"}), 400

    files = request.files.getlist("files[]")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    # 1. Save materials to the fleeting /tmp folder
    tmp_materials_dir = f"/tmp/materials_{course_id}_{assignment_id}"
    os.makedirs(tmp_materials_dir, exist_ok=True)

    saved_files = []
    for f in files:
        filename = secure_filename(f.filename)
        save_path = os.path.join(tmp_materials_dir, filename)
        f.save(save_path)
        saved_files.append(filename)

    # 3. Pass the /tmp folder to the FAISS builder
    result = build_knowledge_base(course_id, assignment_id, tmp_materials_dir)
    result["files_uploaded"] = saved_files

    return jsonify(result)
# =============================================================================
# ENDPOINT 2: Upload Rubric + Solution for an Assignment
#
# Called when: professor creates a new assignment
# Who calls it: instructor dashboard
#
# POST /setup-assignment
# Body (multipart/form-data):
#   course_id       — e.g. "CSC130_Spring2026"
#   assignment_id   — e.g. "Module_5_Discussion"
#   rubric          — .txt file (plain text, written naturally by the prof)
#   solution        — .pdf file (instructor solution)
#
# Rubric format: just plain text. Example:
#   "Q1 (3pts): Student must explain the difference between..."
#   "Q2 (4pts): Code must use course Stack interface, not java.util.Stack"
# The grading prompt labels it explicitly so the model knows what it is.
# =============================================================================

@app.route("/grade", methods=["POST"])
def grade_single():
    course_id = request.form.get("course_id")
    assignment_id = request.form.get("assignment_id")
    submission_file = request.files.get("submission")

    if not all([course_id, assignment_id, submission_file]):
        return jsonify({"error": "course_id, assignment_id, and submission are required"}), 400

    filename = secure_filename(submission_file.filename)

    # Save the student's submission to /tmp
    tmp_submission_path = f"/tmp/{filename}"
    submission_file.save(tmp_submission_path)

    # Backup the submission to GCS for permanent storage
    gcs_submission_path = f"courses/{course_id}/assignments/{assignment_id}/submissions/{filename}"
    upload_to_gcs(tmp_submission_path, gcs_submission_path)

    try:
        # Grade the paper using the /tmp file
        result = grade_submission(course_id, assignment_id, tmp_submission_path)

        # Clean up the /tmp file after grading
        if os.path.exists(tmp_submission_path):
            os.remove(tmp_submission_path)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============================================================================
# ENDPOINT 3: Grade a Single Submission
#
# Called when: a student submits on Canvas (via LTI webhook later)
# Who calls it: Canvas or TA manually uploading a file
#
# POST /grade
# Body (multipart/form-data):
#   course_id       — e.g. "CSC130_Spring2026"
#   assignment_id   — e.g. "Module_5_Discussion"
#   submission      — student's PDF or TXT file
#
# Returns a JSON grading report. TA reviews and approves before
# any grade goes to Canvas — we never auto-post grades.
# =============================================================================

@app.route("/grade-all", methods=["POST"])
def grade_all():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    course_id = data.get("course_id")
    assignment_id = data.get("assignment_id")

    if not course_id or not assignment_id:
        return jsonify({"error": "course_id and assignment_id are required"}), 400

    try:
        results = grade_all_submissions(course_id, assignment_id)
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ENDPOINT 4: Batch Grade All Submissions
#
# Called when: professor clicks "Grade All" in the instructor dashboard
# Who calls it: instructor dashboard
#
# POST /grade-all
# Body (JSON):
#   { "course_id": "CSC130_Spring2026", "assignment_id": "Module_5_Discussion" }
#
# Skips already-graded files using grading_history.json.
# Returns list of grading reports for all students in that assignment.
# =============================================================================

@app.route("/setup-assignment", methods=["POST"])
def setup_assignment():
    course_id = request.form.get("course_id")
    assignment_id = request.form.get("assignment_id")

    if not course_id or not assignment_id:
        return jsonify({"error": "course_id and assignment_id are required"}), 400

    saved = []

    rubric_file = request.files.get("rubric")
    if rubric_file:
        # 1. Save to /tmp
        tmp_rubric_path = f"/tmp/{course_id}_{assignment_id}_rubric.txt"
        rubric_file.save(tmp_rubric_path)

        # 2. Upload to GCS
        gcs_rubric_path = f"courses/{course_id}/assignments/{assignment_id}/rubric.txt"
        upload_to_gcs(tmp_rubric_path, gcs_rubric_path)

        # 3. Clean up /tmp
        os.remove(tmp_rubric_path)
        saved.append("rubric.txt")

    solution_file = request.files.get("solution")
    if solution_file:
        # 1. Save to /tmp
        tmp_solution_path = f"/tmp/{course_id}_{assignment_id}_solution.pdf"
        solution_file.save(tmp_solution_path)

        # 2. Upload to GCS
        gcs_solution_path = f"courses/{course_id}/assignments/{assignment_id}/solution.pdf"
        upload_to_gcs(tmp_solution_path, gcs_solution_path)

        # 3. Clean up /tmp
        os.remove(tmp_solution_path)
        saved.append("solution.pdf")

    return jsonify({
        "status": "ok",
        "course_id": course_id,
        "assignment_id": assignment_id,
        "saved": saved
    })

# =============================================================================
# ENDPOINT 5: Pre-Submission AI Review (Google Docs "Get AI Review" button)
#
# Called when: student clicks the "Get AI Review" button in Google Docs
# Who calls it: Google Docs Apps Script add-on (google_docs_addon.js)
#
# POST /review
# Body (JSON):
#   {
#     "course_id": "CSC130_Spring2026",
#     "assignment_id": "Module_5_Discussion",
#     "doc_text": "...full text of student's Google Doc..."
#   }
#
# Returns per-section feedback with slide references.
# NEVER gives answers — only tells students what looks good or needs work.
# The no-answers constraint is enforced inside the prompt in autograder.py.
# The Google Docs add-on then inserts these as inline comments in the doc.
# =============================================================================

@app.route("/review", methods=["POST"])
def presubmission_review():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    course_id = data.get("course_id")
    assignment_id = data.get("assignment_id")
    doc_text = data.get("doc_text")

    if not all([course_id, assignment_id, doc_text]):
        return jsonify({"error": "course_id, assignment_id, and doc_text are required"}), 400

    try:
        result = get_presubmission_review(course_id, assignment_id, doc_text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =============================================================================
# ENDPOINT 6: Get Saved Grading Results (for TA review dashboard)
#
# Called when: TA opens the grading dashboard to review AI grades
# Who calls it: instructor dashboard frontend
#
# GET /results?course_id=CSC130_Spring2026&assignment_id=Module_5_Discussion
#
# Returns saved grading_history.json for that assignment.
# TA can review, adjust scores, and approve before posting to Canvas.
# =============================================================================

@app.route("/results", methods=["GET"])
def get_results():
    course_id = request.args.get("course_id")
    assignment_id = request.args.get("assignment_id")

    if not course_id or not assignment_id:
        return jsonify({"error": "course_id and assignment_id query params required"}), 400

    history_file = os.path.join(
        "courses", course_id, "assignments", assignment_id, "grading_history.json"
    )

    if not os.path.exists(history_file):
        return jsonify({"results": [], "message": "No grading history found yet"})

    with open(history_file, 'r') as f:
        history = json.load(f)

    return jsonify({"results": list(history.values()), "count": len(history)})

@app.route('/teacher', methods=['GET'])
def teacher_dashboard():
    # This simply loads the HTML page when they visit the URL
    return render_template('teacher_dashboard.html')

# Run with: python app.py (local dev) or gunicorn app:app --bind 0.0.0.0:8080 (production)
port = int(os.environ.get("PORT", 5000))
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))