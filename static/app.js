/* ─────────────────────────────────────────────────────────────────────────────
   RCBC AI AUTOGRADER — app.js (Wired to Flask Backend + Collapsible Details)
   ───────────────────────────────────────────────────────────────────────────── */
'use strict';

const COURSES = {
  'PHY110_Spring2026': {
    code: 'PHY 110', name: 'Physics I — Mechanics',
    assignments: [
      { id: 'Chapter_2_Homework', title: 'Chapter 2 Homework' },
      { id: 'Chapter_3_Homework', title: 'Chapter 3 Homework' },
      { id: 'Chapter_4_Homework', title: 'Chapter 4/5 Homework' },
      { id: 'Chapter_6_Homework', title: 'Chapter 6 Homework' },
      { id: 'Chapter_7_Homework', title: 'Chapter 7 Homework' },
      { id: 'Chapter_8_Homework', title: 'Chapter 8 Homework' },
      { id: 'Chapter_9_Homework', title: 'Chapter 9 Homework' },
      { id: 'Chapter_10_Homework', title: 'Chapter 10 Homework' },
      { id: 'Chapter_11_Homework', title: 'Chapter 11 Homework' },
      { id: 'Chapter_13_Homework', title: 'Chapter 13 Homework' }
    ],
  },
  'PHY212_Spring2026': {
    code: 'PHY 212', name: 'Physics II',
    assignments: [{ id: 'HW-1', title: 'HW-1' }],
  },
  'CSC130_Spring2026': {
    code: 'CSC 130', name: 'Data Structures and Algorithms',
    assignments: [
      { id: 'Stack_Discussion', title: 'Stack Discussion' },
      { id: 'Module_5_Discussion', title: 'Module 5 Dynamic Programming' }
    ],
  },
    'SST100_Summer2026': {
    code: 'SST 100', name: 'Principles of Sustainability',
    assignments: [
    { id: 'assignment_1', title: 'Assignment 1' },
    { id: 'assignment_2', title: 'Assignment 2' },
    { id: 'assignment_3', title: 'Assignment 3' },
    { id: 'assignment_4', title: 'Assignment 4' }
    ],
  }
};

const state = {
  currentCourse: null,
  selectedAssignment: null,
  uploadedFiles: [],
  isGrading: false,
};

const $ = (id) => document.getElementById(id);

/* ── COLLAPSIBLE TOGGLE SYSTEM ── */
window.toggleDetails = function(rowId) {
  const row = $(rowId);
  if (!row) return;
  if (row.style.display === "none" || row.style.display === "") {
    row.style.display = "table-row";
  } else {
    row.style.display = "none";
  }
};

/* ── VIEW ROUTING ── */
function showDashboard() {
  state.currentCourse = null; state.selectedAssignment = null; state.uploadedFiles = []; state.isGrading = false;
  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('nav-item--active'));
  $('nav-dashboard').classList.add('nav-item--active');
  $('view-dashboard').style.display = 'block';
  $('view-course').style.display = 'none';
}

function openCourse(courseId) {
  const course = COURSES[courseId];
  if (!course) return;

  state.currentCourse = courseId; state.selectedAssignment = null; state.uploadedFiles = []; state.isGrading = false;

  document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('nav-item--active'));
  $(`nav-${courseId}`).classList.add('nav-item--active');

  $('hero-course-code').innerText = course.code;
  $('hero-course-name').innerText = course.name;

  $('view-dashboard').style.display = 'none';
  $('view-course').style.display = 'block';

  initSearchableDropdown(course.assignments);
  resetUploadPanel();
  $('results-tbody').innerHTML = '<tr><td colspan="5" class="empty-state">No submissions graded yet.</td></tr>';
}

/* ── SEARCHABLE DROPDOWN LOGIC ── */
function initSearchableDropdown(assignments) {
  const input = $('assignment-search');
  const list = $('assignment-dropdown-list');

  input.value = '';
  list.innerHTML = '';
  state.selectedAssignment = null;
  updateGradeBtn();

  function renderList(filterText = '') {
    list.innerHTML = '';
    const filtered = assignments.filter(a => a.title.toLowerCase().includes(filterText.toLowerCase()));

    if (filtered.length === 0) {
      list.innerHTML = '<li class="search-select__option" style="color:#888; cursor:default;">No matches found</li>';
      return;
    }

    filtered.forEach(a => {
      const li = document.createElement('li');
      li.className = 'search-select__option';
      if (state.selectedAssignment && state.selectedAssignment.id === a.id) {
        li.classList.add('is-selected');
      }
      li.innerText = a.title;
      li.onclick = () => {
        state.selectedAssignment = a;
        input.value = a.title;
        list.classList.remove('is-open');
        updateGradeBtn();
      };
      list.appendChild(li);
    });
  }

  input.onfocus = () => {
    renderList(input.value);
    list.classList.add('is-open');
  };

  input.onkeyup = (e) => {
    state.selectedAssignment = null;
    updateGradeBtn();
    renderList(e.target.value);
  };

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-select')) {
      list.classList.remove('is-open');
    }
  });
}

/* ── FILE UPLOAD ── */
function handleFiles(files) {
  state.uploadedFiles = Array.from(files);
  renderFileList(); updateGradeBtn();
}

function removeFile(index) {
  state.uploadedFiles.splice(index, 1);
  renderFileList(); updateGradeBtn();
  $('file-input').value = '';
}

function renderFileList() {
  const list = $('file-list');
  list.innerHTML = '';
  state.uploadedFiles.forEach((file, i) => {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.innerHTML = `
      <span title="${file.name}">📄 ${file.name}</span>
      <button class="file-item__remove" onclick="removeFile(${i})">X</button>`;
    list.appendChild(item);
  });
}

function updateGradeBtn() {
  const btn = $('grade-btn');
  btn.disabled = !(state.selectedAssignment && state.uploadedFiles.length > 0 && !state.isGrading);
}

const dropZone = $('upload-zone');
dropZone.ondragover = (e) => { e.preventDefault(); dropZone.classList.add('is-dragover'); };
dropZone.ondragleave = () => dropZone.classList.remove('is-dragover');
dropZone.ondrop = (e) => {
  e.preventDefault(); dropZone.classList.remove('is-dragover');
  const files = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
  if (files.length) handleFiles(files);
};
$('file-input').onchange = (e) => { if (e.target.files.length) handleFiles(e.target.files); };

function resetUploadPanel() {
  $('file-list').innerHTML = ''; $('file-input').value = '';
  $('progress-wrap').style.display = 'none'; $('progress-fill').style.width = '0%';
  updateGradeBtn();
}

/* ── BACKEND GRADING CONTEXT WIRE ── */
/* ── BACKEND GRADING CONTEXT WIRE ── */
async function startGrading() {
  if (!state.selectedAssignment || state.uploadedFiles.length === 0 || state.isGrading) return;

  // --- NEW: CONFIRMATION DIALOG ---
  const courseObj = COURSES[state.currentCourse];
  const courseText = `${courseObj.code} (${courseObj.name})`;
  const assignmentText = state.selectedAssignment.title;

  const confirmationMessage = `Wait! Please confirm your selection:\n\nCourse: ${courseText}\nAssignment: ${assignmentText}\n\nDo you want to proceed and run the AI Autograder?`;

  if (!window.confirm(confirmationMessage)) {
    console.log("Grading cancelled by user.");
    return; // Stops the function immediately if they click Cancel
  }
  // --------------------------------

  state.isGrading = true;
  updateGradeBtn();

  const count = state.uploadedFiles.length;
  $('progress-wrap').style.display = 'block';
  $('results-tbody').innerHTML = '';

  let currentFileIndex = 0;

  async function gradeNextFile() {
    if (currentFileIndex >= count) {
      $('progress-fill').style.width = '100%';
      $('progress-pct').innerText = '100%';
      $('progress-label').innerText = 'Complete';
      state.isGrading = false;
      updateGradeBtn();
      return;
    }

    const file = state.uploadedFiles[currentFileIndex];
    $('progress-label').innerText = `Grading ${currentFileIndex + 1} of ${count}...`;

    const tr = document.createElement('tr');
    tr.id = `grading-row-${currentFileIndex}`;
    tr.innerHTML = `
      <td>⏳ ${file.name}</td>
      <td><span class="status-badge status-badge--pending">Analyzing...</span></td>
      <td>--</td><td>--</td><td>AI is reading...</td>
    `;
    $('results-tbody').appendChild(tr);

    let formData = new FormData();
    formData.append('submission', file);
    formData.append('course_id', state.currentCourse);
    formData.append('assignment_id', state.selectedAssignment.id);

    try {
      let response = await fetch('/grade', { method: 'POST', body: formData });
      let result = await response.json();

      const detailsId = `details-${Date.now()}-${currentFileIndex}`;
      const flagLabel = result.flagged_for_review ? '⚠️ Review' : '✅ Clean';

      // Update Main Row with Toggle Button
      tr.innerHTML = `
        <td><strong>${file.name}</strong></td>
        <td><span class="status-badge status-badge--Complete">Complete</span></td>
        <td><strong>${result.score || 'Error'}</strong></td>
        <td>${flagLabel}</td>
        <td><button class="btn-toggle" onclick="toggleDetails('${detailsId}')">View Details</button></td>
      `;

      // Build and Append Collapsible Sub-Row
      const detailsTr = document.createElement('tr');
      detailsTr.id = detailsId;
      detailsTr.className = 'details-row';

      let breakdownHTML = `<td colspan="5"><div class="details-box">`;
      breakdownHTML += `<h4>Overall Summary</h4><p>${result.feedback_summary || 'No summary provided.'}</p>`;
      breakdownHTML += `<h4>Detailed Rubric Breakdown</h4><ul>`;

      if (result.detailed_grading) {
        result.detailed_grading.forEach(q => {
          breakdownHTML += `<li><strong>${q.question} (${q.points}):</strong> ${q.reason}</li>`;
        });
      } else {
        breakdownHTML += `<li>No question-by-question breakdown parsed.</li>`;
      }
      breakdownHTML += `</ul></div></td>`;
      detailsTr.innerHTML = breakdownHTML;

      // Appends right underneath the active row
      $('results-tbody').appendChild(detailsTr);

    } catch (error) {
      tr.innerHTML = `
        <td>${file.name}</td>
        <td><span class="status-badge status-badge--error">Failed</span></td>
        <td>Error</td><td>--</td><td>Server unreachable.</td>
      `;
    }

    const pct = Math.floor(((currentFileIndex + 1) / count) * 100);
    $('progress-fill').style.width = pct + '%';
    $('progress-pct').innerText = pct + '%';

    currentFileIndex++;
    gradeNextFile();
  }

  gradeNextFile();
}

window.App = { showDashboard, openCourse, startGrading };