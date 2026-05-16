# AI-Powered Grading, Formative Feedback, and Personalized Learning Tool
### Rider University — CSC 410 Data Science & Big Data Analytics
**Dhruv Sethi, Nakul Mittal**

---

## What This Is

A fully integrated, cloud-native AI educational tool built for Rider University that addresses three real problems in higher education:

1. **Instructors spend hours grading** — our autograder handles the first pass, and the instructor reviews via a built-in web dashboard.
2. **Students use misaligned AI tools** — our Google Docs add-on gives course-specific pre-submission feedback without giving away answers.
3. **Canvas treats all students the same** — our personalization engine clusters students by learning gaps and generates individual study roadmaps.

Built on top of Google's Gemini 2.5 Flash, FAISS vector search, and a serverless Google Cloud architecture. The RAG pipeline ensures the AI only knows what the professor actually taught — not generic internet knowledge.

---

## Project Structure

```
AI-in-education-project-RiderUniversity/
│
├── app.py                      # Flask REST API — main entry point
├── autograder.py               # Core grading, routing + pre-submission review logic
├── process_documents.py        # Knowledge base builder (FAISS per course)
├── gcs_helper.py               # Google Cloud Storage interface for serverless I/O
├── check_models.py             # Utility to list available Gemini models
├── teacher_dashboard.html      # Frontend portal for batch grading
│
├── google_docs_addon/
│   ├── Code.gs                 # Apps Script backend
│   ├── Sidebar.html            # Sidebar UI shown to students
│   └── appsscript.json         # Add-on manifest + OAuth scopes
│
├── [Google Cloud Storage Bucket] # Replaces local folders for ephemeral cloud compute
│   └── courses/
│       └── {course_id}/
│           ├── assignments/{assignment_id}/materials/       # Slides (PDFs)
│           ├── assignments/{assignment_id}/knowledge_base/  # FAISS index
│           ├── assignments/{assignment_id}/rubric.txt
│           ├── assignments/{assignment_id}/solution.pdf
│           └── assignments/{assignment_id}/submissions/
│
├── .env                        # API keys & Bucket Names — NEVER commit this
├── gcp-credentials.json        # GCP Service Account key (Local dev only) — NEVER commit
└── requirements.txt
```

---

## How It Works

### Cloud-Native Serverless Architecture
The app runs on Google Cloud Run. Because serverless environments delete local files when they sleep, the system uses an in-memory streaming workflow. Files are temporarily saved to /tmp, uploaded to Google Cloud Storage (GCS), processed, and immediately wiped from the server's RAM to ensure security and scalability.

### Smart Routing (Cost Optimization)
The system dynamically scans uploaded PDFs before sending them to the AI:

- **Cheap Text Pipeline:** If the PDF is pure text, it extracts the strings and runs a cheap RAG query (fractions of a penny).
- **Expensive Vision Pipeline:** If the PDF contains images, screenshots, or handwriting, it routes to the Gemini Multimodal Vision API to visually "read" the paper.

### Per-Course Knowledge Isolation
Every assignment gets its own FAISS vector index built from that specific module's lecture slides. CSC 130's grader has no idea what CSC 350 taught. This prevents hallucination and ensures feedback is always course-aligned.

### Grading Hierarchy
The prompt enforces a strict hierarchy of truth:

```
Rubric > Instructor Solution > Course Slides > Student Submission
```

### Logic Trap Detection
Course-specific rules override general knowledge. Example: in CSC 130, pop() is void and returns nothing. A generic AI would accept System.out.println(stack.pop()) as correct — ours flags it because the course slides say otherwise.

---

## Setup & Local Development

### Prerequisites
- Python 3.11
- A Google AI Studio API key (for Gemini 2.5 Flash)
- A Google Cloud Platform (GCP) Account with a Cloud Storage Bucket
- A GCP Service Account JSON key with Storage Object Admin permissions

### Installation

```
git clone https://github.com/sethidhruv188/AI-in-education-project-RiderUniversity-CSC360
cd AI-in-education-project-RiderUniversity-CSC360
pip install -r requirements.txt
```

Create a .env file in the root directory:

```
GOOGLE_API_KEY=your_gemini_api_key_here
GCS_BUCKET_NAME=your-gcp-bucket-name
GOOGLE_APPLICATION_CREDENTIALS="gcp-credentials.json"
```

(Ensure gcp-credentials.json is in the project root for local testing.)

### Running Locally

```
python app.py
```

Server starts at http://localhost:5000. Even when running locally, the app will read and write directly to your live Google Cloud Storage bucket.

---

## API Endpoints

| Method | Endpoint       | What it does                                              |
|--------|----------------|-----------------------------------------------------------|
| GET    | /              | Health check                                              |
| GET    | /teacher       | UI Dashboard for Instructors to upload and batch grade    |
| POST   | /build-kb      | Build knowledge base from uploaded course materials       |
| POST   | /setup-assignment | Upload rubric + solution for an assignment             |
| POST   | /grade         | Grade a single student submission                         |
| POST   | /grade-all     | Batch grade all submissions for an assignment             |
| POST   | /review        | Pre-submission AI review (called by Google Docs add-on)   |
| GET    | /results       | Fetch saved grading results for TA review                 |

---

## Google Docs Add-on

The add-on gives students pre-submission feedback without giving away answers.

Note: The Docs Add-on strictly utilizes the text pipeline for rapid, cheap iteration. Final image/screenshot grading is reserved for the Canvas autograder.

### Setup

1. Open a Google Doc
2. Go to Extensions → Apps Script
3. Paste Code.gs contents into the editor
4. Create a new HTML file named Sidebar and paste Sidebar.html contents
5. Replace appsscript.json with the provided manifest
6. Important: Update Line 1 of Code.gs with your live Cloud Run URL:
   const API_BASE_URL = "https://your-cloud-run-url.a.run.app";
7. Save and refresh the doc.

---

## Deployment (Google Cloud Run)

This app is designed to be continuously deployed via GitHub to Google Cloud Run.

1. Push your code to GitHub (ensure .env and .json keys are in .gitignore).
2. In Google Cloud Console, create a Cloud Run service linked to your repository.
3. Add GOOGLE_API_KEY and GCS_BUCKET_NAME as Environment Variables in the Cloud Run settings.
4. Under the Security tab, attach your autograder-storage-agent Service Account so the container has native access to your Bucket (bypassing the need for a local JSON key file).
5. Deploy.

---

## Tech Stack

| Component       | Technology                  |
|-----------------|-----------------------------|
| Language        | Python 3.11                 |
| API Framework   | Flask                       |
| LLM             | Gemini 2.5 Flash            |
| Vector Database | FAISS                       |
| Embeddings      | all-MiniLM-L6-v2            |
| PDF Processing  | PyMuPDF (fitz)              |
| Cloud Storage   | Google Cloud Storage (GCS)  |
| Deployment      | GCP Cloud Run               |
| Frontend Add-on | Google Apps Script          |

---

## Roadmap

- [x] RAG pipeline with per-course FAISS isolation
- [x] Multimodal grading (handwriting, screenshots, tables)
- [x] Course-specific logic trap detection
- [x] Flask REST API
- [x] Google Docs add-on with AI review sidebar
- [x] GCP Cloud Run deployment & Cloud Storage Integration
- [x] Instructor review dashboard frontend
- [x] Smart Routing (Dynamic Vision vs. Text Pipeline)
- [ ] K-means student clustering + personalized learning paths
- [ ] Canvas LTI 1.3 webhook integration
- [ ] Scheduled batch grading after assignment due dates

---

## Paper

This project is documented in two research papers:

- CSC 360 — AI-Powered Grading, Formative Feedback, and Personalized Learning Tool for Rider University — covers the autograder design, RAG pipeline, and validation experiments
- CSC 410 — extends the above with K-means personalization, Google Docs integration, and Cloud deployment architecture

---

## Notes

- Security: The .env and gcp-credentials.json files are never committed.
- Safety Settings: Filters are set to BLOCK_NONE for CS terminology (e.g., "kill process") — this is intentional and documented in the paper to prevent false flags during code grading.
