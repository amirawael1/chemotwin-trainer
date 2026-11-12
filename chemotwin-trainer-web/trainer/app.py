"""
ChemoTwin Trainer: chemotherapy-toxicity assessment training against current guidelines.

Run from the project folder:
    python -m streamlit run trainer/app.py

Educational tool for doctors in training. It does not make clinical decisions.
"""
import html
import io
import os
import random
import re
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agentic.assessment import TERMS_REQUIRED  # noqa: E402
from trainer import core, storage  # noqa: E402


TEACHING_CASES = {
    "Neutropenia grading after the CTCAE v6.0 update": "CUR-ANC-700",
    "HER2 therapy surveillance: urgency": "CON-HER2-SURV",
    "Trastuzumab washout before anthracycline": "CON-WASHOUT",
    "Reported fever without a temperature value": "LA-12992047",
    "Heart-failure symptoms in the nursing notes": "LB-0009",
}

st.set_page_config(page_title="ChemoTwin Trainer", page_icon="✚", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display: none;}
header[data-testid="stHeader"] {background: transparent; height: 2.6rem;}   /* keeps the sidebar open/close arrow */
.block-container {padding-top: 2.6rem; max-width: 1400px;}
h1, h2, h3, h4 {letter-spacing: -0.01em;}
.topbar {display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
         background:#134E8E; border-radius:6px; padding:14px 22px; margin-bottom:20px;}
.topbar .t1 {font-size:1.5rem; font-weight:700; color:#FFFFFF; line-height:1.1;}
.topbar .t2 {font-size:0.95rem; color:#D6E4F5; margin-top:2px;}
.topbar .refs {color:#FFFFFF; font-size:0.82rem; background:rgba(255,255,255,0.14); padding:4px 10px; border-radius:4px;}
.brand {font-size:1.45rem; font-weight:700; color:#134E8E;}
.case-id {color:#6B7785; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.06em;}
.case-title {font-size:1.25rem; font-weight:650; color:#15202B; margin:2px 0 6px 0;}
.chip {display:inline-block; padding:1px 9px; border-radius:4px; font-size:0.78rem; font-weight:600; margin-right:6px;
       background:#EEF2F6; color:#3E4C5B; border:1px solid #DCE3EA;}
.sec {font-size:0.76rem; font-weight:700; color:#134E8E; text-transform:uppercase; letter-spacing:0.07em;
      margin:14px 0 4px 0; border-bottom:1px solid #E3E8EE; padding-bottom:3px;}
.record {background:#FBFCFD; border:1px solid #DCE3EA; border-left:4px solid #134E8E; border-radius:4px;
         padding:14px 16px; white-space:pre-wrap; font-family:"Consolas","Menlo",monospace; font-size:0.84rem;
         line-height:1.5; color:#1E2A36; max-height:620px; overflow-y:auto;}
.pill {display:inline-block; white-space:nowrap; padding:1px 8px; border-radius:3px; font-size:0.76rem; font-weight:700;}
.ok {background:#E4F2EA; color:#1B6B45;} .bad {background:#FBE6E4; color:#A1261C;} .warn {background:#FCF1DC; color:#8A5300;}
.summary {display:flex; gap:28px; align-items:center; border:1px solid #DCE3EA; border-radius:6px; padding:12px 16px;
          margin-bottom:12px; background:#FBFCFD;}
.summary .num {font-size:1.9rem; font-weight:700; color:#134E8E; line-height:1;}
.summary .lab {font-size:0.78rem; color:#5B6878; text-transform:uppercase; letter-spacing:0.05em;}
table.review {width:100%; border-collapse:collapse; font-size:0.88rem;}
table.review th {text-align:left; color:#5B6878; font-weight:600; font-size:0.76rem; text-transform:uppercase;
                 letter-spacing:0.05em; border-bottom:1px solid #DCE3EA; padding:6px 8px;}
table.review td {border-bottom:1px solid #EEF1F4; padding:6px 8px; vertical-align:top;}
.notes {background:#FFF8EC; border:1px solid #F0D9A8; border-left:5px solid #C98A12; border-radius:6px;
         padding:12px 18px; margin-top:16px; font-size:0.92rem; color:#2B2416;}
.notes .h {font-weight:700; color:#8A5300; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;}
.notes ul {margin:0; padding-left:18px;} .notes li {margin-bottom:5px;}
.about {background:#F2F6FB; border:1px solid #D3DFEC; border-left:5px solid #134E8E; border-radius:6px;
        padding:12px 18px; margin-top:12px; font-size:0.92rem;}
.about .h {font-weight:700; color:#134E8E; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:6px;}
.muted {color:#5B6878; font-size:0.84rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading cases…")
def bank():
    cases = core.load_bank()
    return cases, {c.case_id: c for c in cases}



CASES, BY_ID = bank()
ss = st.session_state
ss.setdefault("case_id", None)
ss.setdefault("results", {})      # case_id -> feedback
ss.setdefault("history", [])      # list of dicts
ss.setdefault("trainee", "")


def pill(text, cls):
    return f'<span class="pill {cls}">{html.escape(text)}</span>'


def sec(text):
    st.markdown(f'<div class="sec">{html.escape(text)}</div>', unsafe_allow_html=True)


def new_case(topic, kind):
    c = core.pick(CASES, topic=None if topic == "All topics" else topic,
                  kind=None if kind == "All settings" else kind,
                  exclude=[h["case_id"] for h in ss.history])
    if c:
        ss.case_id = c.case_id


def trainees():
    try:
        return storage.trainees()
    except Exception:
        st.warning("The progress store cannot be reached right now.")
        return []


def load_history(name):
    try:
        return storage.load_history(name)
    except Exception:
        st.warning("Your earlier results could not be loaded right now; new results will still be saved.")
        return []


def save_progress(name, row):
    try:
        storage.save(row)
        return True
    except Exception:
        st.warning("This result could not be saved (connection problem). Your feedback is still shown.")
        return False


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown('<div class="brand">ChemoTwin</div><div class="muted">Toxicity assessment training</div>',
                unsafe_allow_html=True)
    st.write("")
    page = st.radio("Section", ["Practice", "Progress", "How to use"], label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="sec">Trainee</div>', unsafe_allow_html=True)
    if ss.get("trainee"):
        st.markdown(f"Signed in as **{html.escape(ss.trainee)}**")
        if st.button("Change trainee", width="stretch"):
            ss.trainee, ss.history, ss.results = "", [], {}
            st.rerun()
    else:
        if storage.online():          # online: never list other students' names
            typed = st.text_input("Full name", placeholder="e.g. Sara Ahmed")
        else:
            known = trainees()
            choice = st.selectbox("Your name", ["New trainee"] + known) if known else "New trainee"
            typed = st.text_input("Full name", placeholder="e.g. Sara Ahmed") if choice == "New trainee" else choice
        if st.button("Sign in", type="primary", width="stretch", disabled=not (typed or "").strip()):
            ss.trainee = typed.strip()
            ss.history, ss.results = load_history(ss.trainee), {}
            st.rerun()
        st.caption("Sign in with the same name each time to keep your progress. "
                   "Your results are stored for your course tutor.")
    if page == "Practice":
        topic = st.selectbox("Topic", ["All topics"] + core.TOPICS)
        kind_label = st.selectbox("Setting", ["All settings"] + list(core.CASE_TYPE.values()))
        kind = next((k for k, v in core.CASE_TYPE.items() if v == kind_label), "All settings")
        if st.button("New case", type="primary", width="stretch"):
            new_case(topic, kind)
        st.write("")
        st.markdown('<div class="sec">Teaching cases</div>', unsafe_allow_html=True)
        for label, cid in TEACHING_CASES.items():
            if st.button(label, key=f"teach_{cid}", width="stretch"):
                ss.case_id = cid
                ss.results.pop(cid, None)
    st.divider()
    st.caption("For training only. Not for clinical decisions.")

if ss.case_id is None or ss.case_id not in BY_ID:
    ss.case_id = core.pick(CASES, rng=random.Random()).case_id

st.markdown('<div class="topbar"><div><div class="t1">ChemoTwin</div><div class="t2">Chemotherapy toxicity '
            'assessment · training</div></div>'
            '<div class="refs">CTCAE v6.0 · ESC 2022 · ESMO 2016 · FDA doxorubicin label</div></div>',
            unsafe_allow_html=True)


# ------------------------------------------------------------------ pages
def practice():
    case = BY_ID[ss.case_id]
    st.markdown(f'<div class="case-id">Case {html.escape(case.case_id)}</div>'
                f'<div class="case-title">{html.escape(case.title)}</div>'
                f'<span class="chip">{html.escape(case.topic)}</span><span class="chip">{html.escape(case.case_type)}</span>',
                unsafe_allow_html=True)
    st.write("")
    show = st.toggle("Show patient record", value=True, key="show_record")
    if show:
        left, right = st.columns([5, 6], gap="large")
        with left:
            sec("Patient record")
            st.markdown(f'<div class="record">{html.escape(case.record)}</div>', unsafe_allow_html=True)
    else:
        right = st.container()
    with right:
        res = ss.results.get(case.case_id)
        if res is None:
            answer_form(case)
        else:
            show_feedback(case, res)
    if res is not None:              # full width below the record and the review
        st.write("")
        _, b1, b2 = st.columns([6, 2, 2])
        if b1.button("Retry case", width="stretch"):
            ss.results.pop(case.case_id, None)
            st.rerun()
        if b2.button("Next case", type="primary", width="stretch"):
            new_case("All topics", "All settings")
            st.rerun()


def answer_form(case):
    k = case.case_id
    with st.form(f"form_{k}", border=False):
        sec("1 · Grade the side effects (CTCAE v6.0)")
        grades = {}
        cols = st.columns(4)
        for i, t in enumerate(TERMS_REQUIRED):
            with cols[i % 4]:
                grades[t] = st.selectbox(core.TERM_SHORT[t], core.GRADE_OPTIONS, key=f"g_{k}_{i}")
        sec("2 · Heart damage from treatment (ESC 2022)")
        c1, c2 = st.columns(2)
        with c1:
            ctrcd = st.selectbox("Seen on tests, no symptoms", list(core.CTRCD_ASYMPTOMATIC),
                                 format_func=core.CTRCD_ASYMPTOMATIC.get, key=f"ca_{k}")
        with c2:
            ctrcd_s = st.selectbox("With heart-failure symptoms", list(core.CTRCD_SYMPTOMATIC),
                                   format_func=core.CTRCD_SYMPTOMATIC.get, key=f"cs_{k}")
        sec("3 · Chemotherapy plan")
        decision = st.radio("Decision", list(core.DECISIONS), format_func=core.DECISIONS.get, horizontal=True,
                            label_visibility="collapsed", key=f"d_{k}")
        sec("4 · How urgent is it?")
        level = st.radio("Urgency", list(core.LEVELS), index=3, format_func=core.LEVELS.get, horizontal=True,
                         label_visibility="collapsed", key=f"l_{k}")
        sec("5 · What would you do now?")
        actions = st.multiselect("Orders", list(core.ACTIONS), format_func=core.ACTION_LABELS.get,
                                 label_visibility="collapsed", key=f"a_{k}", placeholder="Choose all that apply")
        sec("6 · Which results would you request?")
        data_req = st.multiselect("Investigations", list(core.DATA_REQUESTS), format_func=core.DATA_REQUESTS.get,
                                  label_visibility="collapsed", key=f"r_{k}", placeholder="Choose any results you are missing")
        st.write("")
        sent = st.form_submit_button("Submit assessment", type="primary", width="stretch")
    if sent:
        a = core.build_assessment(grades, ctrcd, ctrcd_s, decision, level, actions, data_req)
        fb = core.feedback(case, a)
        ss.results[k] = fb
        row = {"time": datetime.now().strftime("%Y-%m-%d %H:%M"), "trainee": ss.get("trainee", ""),
               "case_id": k, "topic": case.topic,
               "case_type": case.case_type, "score": fb["score"], "safe": fb["safe"]}
        ss.history.append(row)
        save_progress(ss.get("trainee", ""), row)
        if not ss.get("trainee"):
            st.toast("Not saved: sign in from the sidebar to keep your progress.")
        st.rerun()


def show_feedback(case, fb):
    status = (pill("Safe plan", "ok") if fb["safe"] else
              pill(f"{fb['n_unsafe']} safety concern{'s' if fb['n_unsafe'] > 1 else ''}", "bad"))
    st.markdown(f'<div class="summary"><div><div class="num">{fb["score"]}%</div>'
                f'<div class="lab">Agreement with guidelines</div></div><div>{status}</div></div>',
                unsafe_allow_html=True)

    rows = fb["rows"]
    shown = [r for r in rows if not (r["group"] == "grade" and r["ok"] and r["yours"] == "0")]
    body = ""
    for r in shown:
        st_ = pill("Correct", "ok") if r["ok"] else (pill("Unsafe", "bad") if r["significant"] else pill("Small difference", "warn"))
        body += (f"<tr><td>{html.escape(str(r['item']))}</td><td>{html.escape(str(r['yours']))}</td>"
                 f"<td>{html.escape(str(r['guideline']))}</td><td>{st_}</td></tr>")
    sec("Your answer compared with the guidelines")
    st.markdown('<table class="review"><tr><th>Item</th><th>Your answer</th><th>Guidelines say</th><th></th></tr>'
                + body + "</table>", unsafe_allow_html=True)
    hidden = len(rows) - len(shown)
    if hidden:
        st.caption(f"{hidden} further grades: 0, correct.")

    points = [f"<b>{html.escape(str(r['item']))}</b>: {html.escape(r['why'])}" for r in rows if not r["ok"] and r["why"]]
    points += [html.escape(t) for t in fb.get("tips", [])]
    if not points:
        points = ["Your assessment matches the guidelines on every checked item."]
    st.markdown('<div class="notes"><div class="h">Teaching points</div><ul>'
                + "".join(f"<li>{p}</li>" for p in points) + "</ul></div>", unsafe_allow_html=True)
    if case.what_it_tests:
        st.markdown(f'<div class="about"><div class="h">About this case</div>{html.escape(case.what_it_tests)}</div>',
                    unsafe_allow_html=True)



def progress():
    mine, everyone = st.tabs(["My progress", "All trainees"])
    with mine:
        if ss.trainee:
            st.markdown(f"**{html.escape(ss.trainee)}**")
        else:
            st.caption("Not signed in: results from this session only, not saved. Sign in from the sidebar.")
        h = ss.history
        if not h:
            st.markdown('<div class="muted">No cases completed yet.</div>', unsafe_allow_html=True)
        else:
            df = pd.DataFrame(h)
            df["score"] = pd.to_numeric(df["score"], errors="coerce")
            c1, c2, c3 = st.columns(3)
            c1.metric("Cases completed", len(df))
            c2.metric("Average agreement", f"{round(df['score'].mean())}%")
            c3.metric("Safe plans", f"{round(100 * df['safe'].astype(bool).mean())}%")
            sec("Agreement by topic")
            st.bar_chart(df.groupby("topic")["score"].mean(), color="#134E8E")
            sec("Case log")
            show = df.drop(columns=["trainee"], errors="ignore").rename(columns={
                "time": "Date", "case_id": "Case", "topic": "Topic", "case_type": "Setting",
                "score": "Agreement (%)", "safe": "Safe plan"})
            st.dataframe(show, hide_index=True, width="stretch")
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("Export (CSV)", buf.getvalue(),
                               f"chemotwin_{(ss.trainee or 'session').replace(' ', '_')}.csv", "text/csv")
    with everyone:
        code = storage.tutor_code()
        if code and st.text_input("Tutor code", type="password", key="tutor_code_in") != code:
            st.caption("For the course tutor.")
            return
        rows = []
        for name in trainees():
            d = pd.DataFrame(load_history(name))
            if d.empty:
                continue
            d["score"] = pd.to_numeric(d["score"], errors="coerce")
            rows.append({"Trainee": name, "Cases": len(d), "Average agreement (%)": round(d["score"].mean()),
                         "Safe plans (%)": round(100 * d["safe"].mean()), "Last activity": str(d["time"].iloc[-1])})
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Saved online in the course progress sheet." if storage.online()
                       else "Saved on this computer in trainer/progress, one file per trainee.")
        else:
            st.markdown('<div class="muted">No saved progress yet.</div>', unsafe_allow_html=True)


def guide():
    sec("How to use")
    st.markdown("""
1. Choose a topic and setting in the sidebar, then **New case**. Or open one of the **Teaching cases**.
2. Read the patient record.
3. Answer the six questions: grades, heart, plan, urgency, management steps and results to request.
4. Press **Submit assessment** to compare your answer with the guidelines and read the teaching points.
5. Your results are saved under **Progress** when you enter your name.
""")
    sec("Guidelines used")
    st.markdown("""
- **NCI CTCAE v6.0** (2025): grading of side effects
- **ESC 2022 Cardio-oncology Guidelines**: heart damage from cancer treatment
- **ESMO 2016**: cardiovascular toxicity of cancer treatment
- **FDA doxorubicin label**: dosing, cumulative dose, liver function and the trastuzumab interval
""")
    st.caption("The cases are training cases written for practice; some are adapted from published case reports. "
               "For education only, not for decisions about real patients.")


{"Practice": practice, "Progress": progress, "How to use": guide}[page]()
