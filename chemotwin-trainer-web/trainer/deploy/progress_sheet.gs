/**
 * ChemoTwin Trainer - progress store (Google Apps Script, bound to a Google Sheet).
 * 1. Change TOKEN below to your own secret word (the same word goes into Streamlit Secrets: progress_token).
 * 2. Deploy > New deployment > Web app; Execute as: Me; Who has access: Anyone. Copy the Web app URL.
 *
 * UPDATING an existing deployment: paste this file over the old code, put YOUR token back in TOKEN, then
 * Deploy > Manage deployments > (pencil) > Version: New version > Deploy. The Web app URL stays the same.
 *
 * Tabs:
 *   progress  one row per answered case (the "errors" column is added automatically)
 *   trainees  one row per trainee with a hash of their personal code. To reset a forgotten code,
 *             delete that trainee's row here; their next sign-in sets a new code.
 */
const TOKEN = "CHANGE-ME-to-a-secret-word";
const FIELDS = ["time", "trainee", "case_id", "topic", "case_type", "score", "safe", "errors"];
const ACCOUNT_FIELDS = ["trainee", "code_hash", "created"];

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

/** The header row of the progress tab; columns added in a later version are appended to it. */
function header_(sh) {
  const width = Math.max(sh.getLastColumn(), 1);
  const head = sh.getRange(1, 1, 1, width).getValues()[0].map(h => String(h).trim());
  while (head.length && !head[head.length - 1]) head.pop();      // drop empty cells after the last title
  const missing = FIELDS.filter(f => head.indexOf(f) === -1);
  if (missing.length) {
    sh.getRange(1, head.length + 1, 1, missing.length).setValues([missing]);
    return head.concat(missing);
  }
  return head;
}

function accounts_() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  let sh = ss.getSheetByName("trainees");
  if (!sh) {
    sh = ss.insertSheet("trainees");
    sh.getRange("A:C").setNumberFormat("@");
    sh.appendRow(ACCOUNT_FIELDS);
  }
  return sh;
}

function norm_(name) {
  return String(name || "").trim().replace(/\s+/g, " ").toLowerCase();
}

/** The stored code hash for a trainee, or "" when the trainee has none yet. */
function codeHash_(name) {
  const key = norm_(name);
  const values = accounts_().getDataRange().getValues();
  for (let i = 1; i < values.length; i++) {
    if (norm_(values[i][0]) === key && String(values[i][1]).trim()) return String(values[i][1]).trim();
  }
  return "";
}

function out_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function rows_() {
  const values = sheet_().getDataRange().getValues();
  const head = values.shift().map(h => String(h).trim());
  return values.map(v => Object.fromEntries(head.map((k, i) => [k, v[i]])));
}

function doPost(e) {
  const body = JSON.parse(e.postData.contents || "{}");
  if (body.token !== TOKEN) return out_({ok: false, error: "wrong token"});

  if (body.action === "register") {                    // first sign-in: store the trainee's code hash
    const name = String(body.trainee || "").trim().replace(/\s+/g, " ");
    const hash = String(body.code_hash || "").trim();
    if (!name || !hash) return out_({ok: false, error: "name and code are required"});
    const lock = LockService.getScriptLock();
    lock.waitLock(10000);
    try {
      const existing = codeHash_(name);
      if (existing) return out_({ok: true, code_hash: existing, created: false});   // never overwrite a code
      accounts_().appendRow([name, hash, new Date().toISOString()]);
    } finally {
      lock.releaseLock();
    }
    return out_({ok: true, code_hash: hash, created: true});
  }
  if (body.action) return out_({ok: false, error: "unknown action"});

  const r = body.row || {};
  if (!String(r.trainee || "").trim()) return out_({ok: false, error: "no trainee name"});
  const lock = LockService.getScriptLock();
  lock.waitLock(10000);
  try {
    const sh = sheet_();
    const head = header_(sh);                          // write each value under its own column name
    sh.appendRow(head.map(f => (r[f] === undefined || r[f] === null ? "" : String(r[f]))));
  } finally {
    lock.releaseLock();
  }
  return out_({ok: true});
}

function doGet(e) {
  const p = e.parameter || {};
  if (p.token !== TOKEN) return out_({ok: false, error: "wrong token"});
  if (p.action === "account") {
    return out_({ok: true, code_hash: codeHash_(p.trainee)});
  }
  const rows = rows_();
  if (p.action === "trainees") {
    const names = [...new Set(rows.map(r => String(r.trainee).trim()).filter(Boolean))];
    return out_({ok: true, trainees: names});
  }
  if (p.action === "history") {
    const name = norm_(p.trainee);
    return out_({ok: true, rows: rows.filter(r => norm_(r.trainee) === name)});
  }
  if (p.action === "all") {                            // tutor view: every row in one request
    return out_({ok: true, rows: rows.filter(r => String(r.trainee).trim())});
  }
  return out_({ok: false, error: "unknown action"});
}
