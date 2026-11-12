/**
 * ChemoTwin Trainer - progress store (Google Apps Script, bound to a Google Sheet).
 * 1. Change TOKEN below to your own secret word (the same word goes into Streamlit Secrets: progress_token).
 * 2. Deploy > New deployment > Web app; Execute as: Me; Who has access: Anyone. Copy the Web app URL.
 */
const TOKEN = "CHANGE-ME-to-a-secret-word";
const FIELDS = ["time", "trainee", "case_id", "topic", "case_type", "score", "safe"];

function sheet_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName("progress");
  if (!sh) {
    sh = ss.insertSheet("progress");
    sh.getRange("A:B").setNumberFormat("@");          // keep time and names as plain text
    sh.appendRow(FIELDS);
  }
  return sh;
}

function out_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function rows_() {
  const values = sheet_().getDataRange().getValues();
  const head = values.shift();
  return values.map(v => Object.fromEntries(head.map((k, i) => [k, v[i]])));
}

function doPost(e) {
  const body = JSON.parse(e.postData.contents || "{}");
  if (body.token !== TOKEN) return out_({ok: false, error: "wrong token"});
  const r = body.row || {};
  if (!String(r.trainee || "").trim()) return out_({ok: false, error: "no trainee name"});
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    sheet_().appendRow(FIELDS.map(f => (r[f] === undefined ? "" : String(r[f]))));
  } finally {
    lock.releaseLock();
  }
  return out_({ok: true});
}

function doGet(e) {
  const p = e.parameter || {};
  if (p.token !== TOKEN) return out_({ok: false, error: "wrong token"});
  const rows = rows_();
  if (p.action === "trainees") {
    const names = [...new Set(rows.map(r => String(r.trainee).trim()).filter(Boolean))];
    return out_({ok: true, trainees: names});
  }
  if (p.action === "history") {
    const name = String(p.trainee || "").trim().toLowerCase();
    return out_({ok: true, rows: rows.filter(r => String(r.trainee).trim().toLowerCase() === name)});
  }
  return out_({ok: false, error: "unknown action"});
}
