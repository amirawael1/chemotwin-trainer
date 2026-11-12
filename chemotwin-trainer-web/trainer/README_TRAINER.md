# ChemoTwin Trainer

A practice tool for doctors in training: assess a chemotherapy-toxicity case, then compare your answer with the
guideline answer (CTCAE v6.0, ESC 2022, ESMO 2016, FDA doxorubicin label) and with what the AI systems said.
For education only; it does not make clinical decisions.

## Start
Double-click `Start ChemoTwin Trainer.bat`, or in the project folder:
```
pip install streamlit
python -m streamlit run trainer\app.py
```
The app opens in your browser at http://localhost:8501.

## Screens
- **Practice:** choose a topic and case type, read the record, give grades, cardiotoxicity, decision, urgency,
  actions and data requests, then *Check my answer*.
- **Feedback:** item by item (✅ right, ⚠️ minor, ⛔ safety issue), the guideline rule behind each difference,
  then the saved answers of the pure AI and of ChemoTwin, and how ChemoTwin checked itself.
- **My progress:** score by topic; download as CSV. With a name entered, results are also saved in
  `trainer\progress\<name>.csv`.
- **About ChemoTwin:** the five agents and why the checks matter.

## How the scoring works
Your answer is turned into the same structured assessment the AI systems produce and scored by the same scorer
(`agentic/scoring.py`) against the same answer key. Free-text checks (citations, warning flags) are not applied.

## Data
360 cases: 59 guideline edge cases, 178 everyday hospital-format records, 100 cardio-oncology clinic cases,
23 real published cases. AI answers come from `viewer/viewer_data.json` (rebuild with `python viewer\build_viewer.py`).
