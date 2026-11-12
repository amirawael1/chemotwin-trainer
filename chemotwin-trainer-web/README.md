# ChemoTwin Trainer

Practice chemotherapy-toxicity assessment against current guidelines (NCI CTCAE v6.0, ESC 2022 cardio-oncology,
ESMO 2016, FDA doxorubicin label). For education only, not for decisions about real patients.

Author: Amira Wael, MSc thesis, German International University (GIU), Cairo.

Run locally: `pip install -r requirements.txt` then `streamlit run trainer/app.py`.
Online: Streamlit Community Cloud, main file `trainer/app.py`; progress is stored in a Google Sheet through
`trainer/deploy/progress_sheet.gs` (settings in Streamlit Secrets, see `trainer/deploy/secrets_example.toml`).
