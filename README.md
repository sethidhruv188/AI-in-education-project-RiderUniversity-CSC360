# AI-Powered Grading, Formative Feedback, and Personalized Learning Tool
### Rider University — CSC 410 Data Science & Big Data Analytics
**Dhruv Sethi, Nakul Mittal**

---

## What This Is

A fully integrated AI educational tool built for Rider University that addresses three real problems in higher education:

1. **Instructors spend hours grading** — our autograder handles the first pass, TA reviews and approves
2. **Students use misaligned AI tools** — our Google Docs add-on gives course-specific pre-submission feedback without giving away answers
3. **Canvas treats all students the same** — our personalization engine clusters students by learning gaps and generates individual study roadmaps

Built on top of Google's Gemini 2.5 Flash, FAISS vector search, and a RAG pipeline that ensures the AI only knows what the professor actually taught — not generic internet knowledge.

---

## Project Structure

```
AI-in-education-project-RiderUniversity/
│
├── app.py                        # Flask REST API — main entry point
├── autograder.py                 # Core grading + pre-submission review logic
├── process_documents.py          # Knowledge base builder (FAISS per course)
├── check_models.py               # Utility to list available Gemini models
│
├── google_docs_addon/
│   ├── Code.gs                   # Apps Script backend
│   ├── Sidebar.html              # Sidebar UI shown to students
│   └── appsscript.json           # Add-on manifest + OAuth scopes
│
├── courses/                      # Auto-generated, not committed to git
│   └── {course_id}/
│       ├── materials/            # Professor's lecture slides (PDFs)
│       ├── knowledge_base/       # FAISS index (auto-generated)
│       └── assignments/
│           └── {assignment_id}/
│               ├── rubric.txt
│               ├── solution.pdf
│               ├── submissions/
│               └── grading_history.json
│
├── .env                          # API keys — never commit this
└── requirements.txt
```

---

## How It Works

### Per-Course Knowledge Isolation
Every course gets its own FAISS vector index built from that course's lecture slides. CSC 130's grader has no idea what CSC 350 taught. This prevents hallucination and ensures feedback is always course-aligned.

### RAG Pipeline
When grading, the system:
1. Extracts text from the student submission
2. Queries the course's FAISS index for the 3 most relevant slide sections
3. Sends `rubric + slides context + instructor solution + student submission` to Gemini
4. Returns structured JSON with per-question scores and reasoning

### Grading Hierarchy
The prompt enforces a strict hierarchy of truth:
```
Rubric > Instructor Solution > Course Slides > Student Submission
```

### Logic Trap Detection
Course-specific rules override general knowledge. Example: in CSC 130, `pop()` is void and returns nothing. A generic AI would accept `System.out.println(stack.pop())` as correct — ours flags it because the course slides say otherwise.

---

## Setup

### Prerequisites
- Python 3.11
- A Google API key with Gemini access
- ngrok (for local testing of the Google Docs add-on)

### Installation

```bash
git clone https://github.com/sethidhruv188/AI-in-education-project-RiderUniversity-CSC360
cd AI-in-education-project-RiderUniversity-CSC360
pip install -r requirements.txt
```

Create a `.env` file in the root:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Running Locally

```bash
python app.py
```

Server starts at `http://localhost:5000`.

---

## API Endpoints

| Method | Endpoint | What it does |
|--------|----------|--------------|
| GET | `/` | Health check |
| POST | `/build-kb` | Build knowledge base from uploaded course materials |
| POST | `/setup-assignment` | Upload rubric + solution for an assignment |
| POST | `/grade` | Grade a single student submission |
| POST | `/grade-all` | Batch grade all submissions for an assignment |
| POST | `/review` | Pre-submission AI review (called by Google Docs add-on) |
| GET | `/results` | Fetch saved grading results for TA review |

### Example: Build Knowledge Base
```bash
POST /build-kb
form-data:
  course_id = CSC130_Spring2026
  files[] = Lecture_The_Stack_ADT.pdf
  files[] = CSC350_Module_5_Dynamic_programming.pdf
```

### Example: Grade All Submissions
```bash
POST /grade-all
Content-Type: application/json

{
  "course_id": "CSC130_Spring2026",
  "assignment_id": "Stack_Discussion"
}
```

### Example: Pre-Submission Review
```bash
POST /review
Content-Type: application/json

{
  "course_id": "CSC130_Spring2026",
  "assignment_id": "Stack_Discussion",
  "doc_text": "Q1: A stack is a LIFO data structure..."
}
```

---

## Testing with Postman

1. Start the server: `python app.py`
2. Open Postman
3. Run in this order:
   - `POST /build-kb` — upload course PDFs
   - `POST /setup-assignment` — upload rubric + solution
   - `POST /grade-all` — grade all submissions
   - `GET /results` — view grading reports

---

## Google Docs Add-on

The add-on gives students pre-submission feedback without giving away answers.

### Setup
1. Open a Google Doc
2. Go to Extensions → Apps Script
3. Paste `Code.gs` contents into the editor
4. Create a new HTML file named `Sidebar` and paste `Sidebar.html` contents
5. Replace `appsscript.json` with the provided manifest
6. Save and refresh the doc — "AI Review" appears in the menu bar

### How Students Use It
1. Doc must start with a header line:
   ```
   COURSE: CSC130_Spring2026 | ASSIGNMENT: Stack_Discussion
   ```
2. Student writes their answer below the header
3. Click **AI Review → Get AI Feedback**
4. Feedback appears as Google Docs comments — what looks good, what needs work, which slides to review
5. No answers are ever given — the prompt explicitly prohibits it

### ngrok for Local Testing
Since Google's servers can't reach localhost, use ngrok:
```bash
ngrok http 5000
```
Copy the generated URL and update line 1 of `Code.gs`:
```javascript
const API_BASE_URL = "https://your-ngrok-url.ngrok-free.app";
```

---

## Course Lifecycle

**When a course starts:**
- Professor uploads slides → `/build-kb` builds isolated FAISS index
- Professor creates assignments → `/setup-assignment` saves rubric + solution

**During the semester:**
- Students submit work → `/grade` or `/grade-all` runs automatically
- Students use Google Docs add-on for pre-submission feedback
- TAs review AI grading reports via `/results` and approve before posting to Canvas

**When a course ends:**
- Move `courses/{course_id}/` to `archive/`
- Strip submission PDFs, keep `grading_history.json` and `knowledge_base/`
- Professor can clone the knowledge base for next semester

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11 |
| API Framework | Flask |
| LLM | Gemini 2.5 Flash |
| Vector Database | FAISS |
| Embeddings | all-MiniLM-L6-v2 |
| PDF Processing | PyMuPDF (fitz) |
| Frontend Add-on | Google Apps Script |
| Deployment (planned) | GCP Cloud Run |
| Canvas Integration (planned) | LTI 1.3 |

---

## Roadmap

- [x] RAG pipeline with per-course FAISS isolation
- [x] Multimodal grading (handwriting, screenshots, tables)
- [x] Course-specific logic trap detection
- [x] Flask REST API with 6 endpoints
- [x] Google Docs add-on with AI review sidebar
- [ ] K-means student clustering + personalized learning paths
- [ ] GCP Cloud Run deployment
- [ ] Canvas LTI 1.3 webhook integration
- [ ] Scheduled batch grading after assignment due dates
- [ ] TA review dashboard frontend

---

## Paper

This project is documented in two research papers:

- **CSC 240** — *AI-Powered Grading, Formative Feedback, and Personalized Learning Tool for Rider University* — covers the autograder design, RAG pipeline, and validation experiments
- **CSC 410** — extends the above with K-means personalization, Google Docs integration, and Canvas deployment architecture

---

## Notes

- The `.env` file is never committed — add it to `.gitignore`
- `courses/` directory is never committed — it contains student data
- `faiss_index.bin` and `text_metadata.pkl` are auto-generated — no need to commit them
- Safety filters are set to `BLOCK_NONE` for CS terminology (e.g. "kill process") — this is intentional and documented in the paper
- ngrok URL changes every session — update `Code.gs` line 1 each time, or deploy to GCP for a permanent URL
