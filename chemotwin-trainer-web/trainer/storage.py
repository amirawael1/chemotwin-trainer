"""
trainer/storage.py
──────────────────
Where trainee progress is kept.

* On a laptop (no settings): one CSV file per trainee in trainer/progress/.
* Online (Streamlit Community Cloud): rows are sent to a Google Sheet through a small Google Apps Script
  web app (trainer/deploy/progress_sheet.gs). Settings come from Streamlit "Secrets":

      progress_url   = "https://script.google.com/macros/s/.../exec"
      progress_token = "the same secret word as in the script"
      tutor_code     = "a code that unlocks the All trainees view"

The app only calls trainees(), load_history(name) and save(row); it does not know which store is used.
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

FIELDS = ["time", "trainee", "case_id", "topic", "case_type", "score", "safe"]
PROGRESS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "progress")


def _secret(key: str) -> str:
    try:
        import streamlit as st
        return str(st.secrets.get(key, "") or "")
    except Exception:          # no secrets file on a laptop
        return ""


def online() -> bool:
    return bool(_secret("progress_url"))


def tutor_code() -> str:
    return _secret("tutor_code")


def _clean(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        r = {f: r.get(f, "") for f in FIELDS}
        try:
            r["score"] = int(float(r["score"]))
        except (TypeError, ValueError):
            r["score"] = None
        r["safe"] = str(r["safe"]).strip().lower() in ("true", "1", "yes")
        out.append(r)
    return out


# ------------------------------------------------------------------ laptop: CSV files
def _path(name: str) -> str:
    slug = "".join(ch for ch in name.strip().lower().replace(" ", "_") if ch.isalnum() or ch in "-_")
    return os.path.join(PROGRESS_DIR, (slug or "trainee") + ".csv")


def _local_history(name: str) -> List[Dict]:
    path = _path(name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["trainee"] = name
    return _clean(rows)


def _local_trainees() -> List[str]:
    out = []
    if os.path.isdir(PROGRESS_DIR):
        for f in sorted(os.listdir(PROGRESS_DIR)):
            if f.endswith(".csv"):
                with open(os.path.join(PROGRESS_DIR, f), encoding="utf-8") as fh:
                    rows = list(csv.DictReader(fh))
                named = [r.get("trainee") for r in rows if r.get("trainee")]
                out.append(named[-1] if named else f[:-4])
    return out


def _local_save(row: Dict) -> None:
    os.makedirs(PROGRESS_DIR, exist_ok=True)
    rows = _local_history(row["trainee"]) + [row]
    with open(_path(row["trainee"]), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ------------------------------------------------------------------ online: Google Sheet
def _get(**params) -> Dict:
    import requests
    params["token"] = _secret("progress_token")
    r = requests.get(_secret("progress_url"), params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "progress store refused the request"))
    return data


def _post(row: Dict) -> None:
    import requests
    r = requests.post(_secret("progress_url"), json={"token": _secret("progress_token"), "row": row}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(data.get("error", "progress store refused the row"))


# ------------------------------------------------------------------ what the app uses
def trainees() -> List[str]:
    return _get(action="trainees").get("trainees", []) if online() else _local_trainees()


def load_history(name: str) -> List[Dict]:
    if not name:
        return []
    if online():
        rows = _get(action="history", trainee=name).get("rows", [])
        for r in rows:
            r["trainee"] = name
        return _clean(rows)
    return _local_history(name)


def save(row: Dict) -> None:
    if row.get("trainee"):
        _post(row) if online() else _local_save(row)
