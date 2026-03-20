const API_BASE_URL = "YOUR_NGROK_OR_PRODUCTION_URL_HERE"; // Add URL from ngrok

function onOpen() {
  DocumentApp.getUi()
    .createMenu("AI Review")
    .addItem("Get AI Feedback", "showSidebar")
    .addToUi();
}

function showSidebar() {
  const html = HtmlService.createHtmlOutputFromFile("Sidebar")
    .setTitle("AI Review")
    .setWidth(300);
  DocumentApp.getUi().showSidebar(html);
}

function getDocContext() {
  const doc = DocumentApp.getActiveDocument();
  const body = doc.getBody();
  const fullText = body.getText();
  const lines = fullText.split("\n").filter(l => l.trim() !== "");

  // Read course + assignment from the first line of the doc
  // Expected format: COURSE: CSC130_Spring2026 | ASSIGNMENT: Stack_Discussion
  let courseId = null;
  let assignmentId = null;

  if (lines.length > 0) {
    const header = lines[0];
    const courseMatch = header.match(/COURSE:\s*(\S+)/i);
    const assignmentMatch = header.match(/ASSIGNMENT:\s*(\S+)/i);
    if (courseMatch) courseId = courseMatch[1].trim();
    if (assignmentMatch) assignmentId = assignmentMatch[1].trim();
  }

  // Everything after the first line is the student's actual content
  const docText = lines.slice(1).join("\n").trim();

  return {
    courseId: courseId,
    assignmentId: assignmentId,
    docText: docText,
    hasHeader: courseId !== null && assignmentId !== null
  };
}

function runAIReview() {
  const context = getDocContext();

  if (!context.hasHeader) {
    return {
      error: true,
      message: "Could not find course header on line 1. Make sure your doc starts with:\nCOURSE: CSC130_Spring2026 | ASSIGNMENT: Stack_Discussion"
    };
  }

  if (!context.docText || context.docText.length < 20) {
    return {
      error: true,
      message: "Your document seems empty. Write your answer first, then request a review."
    };
  }

  const payload = {
    course_id: context.courseId,
    assignment_id: context.assignmentId,
    doc_text: context.docText
  };

  const options = {
    method: "post",
    contentType: "application/json",
    payload: JSON.stringify(payload),
    muteHttpExceptions: true,
    headers: {
      "ngrok-skip-browser-warning": "true"
    }
  };

  try {
    const response = UrlFetchApp.fetch(API_BASE_URL + "/review", options);
    const code = response.getResponseCode();
    const raw = response.getContentText();

    if (code !== 200) {
      return { error: true, message: "API error " + code + ": " + raw };
    }

    const result = JSON.parse(raw);

    if (result.error) {
      return { error: true, message: result.error };
    }

    // Insert feedback as comments in the document
    insertCommentsInDoc(result);

    return { error: false, result: result };

  } catch (e) {
    return { error: true, message: "Request failed: " + e.toString() };
  }
}

function insertCommentsInDoc(result) {
  const doc = DocumentApp.getActiveDocument();
  const docId = doc.getId();
  const token = ScriptApp.getOAuthToken();

  const commentsToAdd = [];

  if (result.overall_impression) {
    commentsToAdd.push("OVERALL: " + result.overall_impression);
  }

  if (result.items && result.items.length > 0) {
    result.items.forEach(function(item) {
      const emoji = item.status === "looks good" ? "✓" :
                    item.status === "needs review" ? "⚠" : "?";
      let text = emoji + " " + item.section + ": " + item.comment;
      if (item.reference) {
        text += "\n→ " + item.reference;
      }
      commentsToAdd.push(text);
    });
  }

  // Add each feedback item as a document-level comment
  commentsToAdd.forEach(function(commentText) {
    const url = "https://www.googleapis.com/drive/v3/files/" + docId + "/comments?fields=id";
    const payload = JSON.stringify({
      content: commentText,
      anchor: JSON.stringify({ r: "head" })
    });
    UrlFetchApp.fetch(url, {
      method: "post",
      contentType: "application/json",
      headers: { Authorization: "Bearer " + token },
      payload: payload,
      muteHttpExceptions: true
    });
  });
}

function buildCommentText(item) {
  const statusEmoji = {
    "looks good": "✓",
    "needs review": "⚠",
    "unclear": "?"
  };

  const emoji = statusEmoji[item.status] || "•";
  let text = emoji + " " + item.section + ": " + item.comment;

  if (item.reference) {
    text += "\n→ Reference: " + item.reference;
  }

  return text;
}
