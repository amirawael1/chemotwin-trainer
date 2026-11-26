"""
ChemoTwin Trainer: chemotherapy-toxicity assessment training against current guidelines.

Run from the project folder:
    python -m streamlit run trainer/app.py

Educational tool for doctors in training. It does not make clinical decisions.
The screen shows one thing at a time: a start page, then a case in three short steps, then the feedback.
"""
import html
import io
import json
import os
import random
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1  # noqa: F401  (st.components.v1.html)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agentic.assessment import TERMS_REQUIRED  # noqa: E402
from trainer import core, replay, storage  # noqa: E402

TEACHING_CASES = {
    "Neutropenia grading after the CTCAE v6.0 update": "CUR-ANC-700",
    "HER2 therapy surveillance: urgency": "CON-HER2-SURV",
    "Trastuzumab washout before anthracycline": "CON-WASHOUT",
    "Reported fever without a temperature value": "LA-12992047",
}
PAGES = ["Practice", "Progress", "Help"]

st.set_page_config(page_title="ChemoTwin Trainer", page_icon="✚", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"],
[data-testid="stToolbarActions"], [data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] {display: none;}
header[data-testid="stHeader"] {background: transparent; height: 0.5rem;}
.block-container {padding-top: 1.6rem; max-width: 1180px;}
.brand {font-size:1.35rem; font-weight:700; color:#134E8E; line-height:1.2;}
.brand span {font-weight:400; color:#5B6878; font-size:0.95rem; margin-left:8px;}
.rule {border-bottom:2px solid #134E8E; margin:6px 0 22px 0;}
.hero {font-size:1.6rem; font-weight:650; color:#15202B; margin:18px 0 4px 0;}
.muted {color:#5B6878; font-size:0.9rem;}
.case-title {font-size:1.2rem; font-weight:650; color:#15202B; margin:0 0 10px 0;}
.sec {font-size:0.76rem; font-weight:700; color:#134E8E; text-transform:uppercase; letter-spacing:0.07em;
      margin:16px 0 6px 0;}
.step {font-size:0.78rem; color:#5B6878; text-transform:uppercase; letter-spacing:0.06em;}
.q {font-size:1.15rem; font-weight:650; color:#15202B; margin:2px 0 12px 0;}
.chart {border:1px solid #DCE3EA; border-radius:6px; background:#FFFFFF; padding:6px 16px 12px 16px;}
.chart table {width:100%; border-collapse:collapse; font-size:0.88rem;}
.chart td, .chart th {padding:5px 8px 5px 0; vertical-align:top; border-bottom:1px solid #EEF1F4; text-align:left;}
.chart th {color:#5B6878; font-weight:600; font-size:0.74rem; text-transform:uppercase; letter-spacing:0.05em;}
.chart td.k {color:#5B6878; width:34%;}
.chart .plain {font-size:0.88rem; padding:5px 0; border-bottom:1px solid #EEF1F4;}
.chart .head {font-size:0.74rem; font-weight:700; color:#134E8E; letter-spacing:0.06em; padding-top:12px;}
tr.hl td, .chart .plain.hl {background:#FFF6D6;}
.chart .mono {font-family:Consolas,Menlo,monospace; font-size:0.8rem; white-space:pre-wrap;}
.pill {display:inline-block; padding:1px 8px; border-radius:3px; font-size:0.76rem; font-weight:700; white-space:nowrap;}
.ok {background:#E4F2EA; color:#1B6B45;} .bad {background:#FBE6E4; color:#A1261C;} .warn {background:#FCF1DC; color:#8A5300;}
.verdict {border-radius:6px; padding:14px 18px; margin-bottom:14px; font-size:1.05rem; font-weight:650;}
.verdict.ok {background:#E4F2EA; color:#1B6B45;} .verdict.bad {background:#FBE6E4; color:#A1261C;}
.verdict span {font-weight:400; font-size:0.9rem; margin-left:10px;}
.item {border:1px solid #DCE3EA; border-left:4px solid #C98A12; border-radius:6px; padding:10px 14px; margin-bottom:8px;
       font-size:0.9rem;}
.item.unsafe {border-left-color:#A1261C;}
.item .t {font-weight:650; color:#15202B;} .item .cmp {color:#3E4C5B; margin:3px 0;} .item .why {color:#5B6878;}
.invite {border:1px solid #BFD3EA; background:#F2F7FC; border-radius:6px; padding:10px 14px; margin:0 0 8px 0;
          font-size:0.92rem; color:#134E8E;}
.invite span {display:block; color:#3E4C5B; font-size:0.86rem; margin-top:2px;}
table.review {width:100%; border-collapse:collapse; font-size:0.86rem;}
table.review th {text-align:left; color:#5B6878; font-weight:600; font-size:0.74rem; text-transform:uppercase;
                 border-bottom:1px solid #DCE3EA; padding:5px 8px;}
table.review td {border-bottom:1px solid #EEF1F4; padding:5px 8px; vertical-align:top;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading cases…")
def bank():
    cases = core.load_bank()
    return cases, {c.case_id: c for c in cases}


CASES, BY_ID = bank()
ss = st.session_state
for _k, _v in (("case_id", None), ("results", {}), ("history", []), ("trainee", ""), ("answers", {}), ("steps", {}),
               ("f_topic", "All topics"), ("f_kind", "All settings"), ("mode", "practice"), ("started", {}),
               ("view", None), ("study_level", ""), ("pipe_open", {}), ("clash", "")):
    ss.setdefault(_k, _v)


def pill(text, cls):
    return f'<span class="pill {cls}">{html.escape(text)}</span>'


def sec(text):
    st.markdown(f'<div class="sec">{html.escape(text)}</div>', unsafe_allow_html=True)


def open_case(case_id, mode="practice"):
    ss.case_id, ss.mode, ss.view = case_id, mode, None
    ss.results.pop(case_id, None)
    ss.answers.pop(case_id, None)
    ss.steps[case_id] = 1
    ss.started[case_id] = time.time()
    ss.pipe_open.pop(case_id, None)


def study_locked():
    """The study cases stay out of practice until the participant has answered all ten."""
    return core.STUDY_CASES if core.next_study_case(ss.history) else []


def new_case():
    topic, kind = ss.f_topic, ss.f_kind
    c = core.pick(CASES, topic=None if topic == "All topics" else topic,
                  kind=None if kind == "All settings" else kind,
                  exclude=[h["case_id"] for h in ss.history], never=study_locked())
    if c:
        open_case(c.case_id)
    return c is not None


def trainees():
    try:
        return storage.trainees()
    except Exception:
        st.warning("The progress store cannot be reached right now.")
        return []


def load_history(name):
    """(rows, reached). reached is False when the progress store could not be read, so an empty list is
    never mistaken for "this trainee has not answered anything yet"."""
    try:
        return storage.load_history(name), True
    except Exception:
        return [], False


def quiet_trainees():
    """The saved trainee ids, for the duplicate-name check at sign-in only: never shown, and if they
    cannot be read the check is simply not made."""
    try:
        return storage.trainees()
    except Exception:
        return []


def save_progress(row):
    try:
        storage.save(row)
    except Exception:
        st.warning("This result could not be saved (connection problem). Your feedback is still shown.")


# ------------------------------------------------------------------ top bar (no sidebar: works on a phone too)
left, right = st.columns([5, 4])
with left:
    st.markdown('<div class="brand">ChemoTwin<span>Chemotherapy toxicity training</span></div>', unsafe_allow_html=True)
with right:
    page = st.radio("Section", PAGES, horizontal=True, label_visibility="collapsed", key="page") or "Practice"
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ start page
def sign_in_box():
    if ss.trainee:
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"Signed in as **{html.escape(core.display_name(ss.trainee))}**")
        if c2.button("Sign out", width="stretch"):
            ss.trainee, ss.history, ss.results, ss.case_id = "", [], {}, None
            ss.clash, ss.in_name, ss.in_code = "", "", ""
            st.rerun()
        return
    with st.expander("Sign in to save your progress", expanded=True):
        typed = st.text_input("Full name", placeholder="e.g. Sara Ahmed", key="in_name")
        code = st.text_input("Personal code", type="password", placeholder="Choose 4 or more characters",
                             key="in_code",
                             help="You choose it the first time. Use the same name and code each time.")
        ready = bool((typed or "").strip()) and len(code or "") >= 4
        if st.button("Sign in", type="primary", disabled=not ready):
            ss.clash = ""
            tid = core.trainee_id(typed, code)
            hist, reached = load_history(tid)
            verdict = core.sign_in_verdict(tid, hist, reached, quiet_trainees() if reached else [])
            if verdict == "offline":
                st.error("Your saved answers cannot be read at the moment, so signing in now would look as if "
                         "they were gone. Please try again in a minute. You can practise without signing in.")
            elif verdict == "name_taken":
                ss.clash = tid          # kept in the session so the message survives the next run
            else:
                ss.trainee, ss.history, ss.results = tid, hist, {}
                st.rerun()
        if ss.clash:
            st.error("**This name already has saved answers under a different personal code.** Check the code "
                     "you chose the first time and sign in again, so that you continue where you stopped.")
            if st.button("This is my first time with this name: start a new record"):
                ss.trainee, ss.history, ss.results, ss.clash = ss.clash, [], {}, ""
                st.rerun()
            st.caption("If someone with the same name already uses the trainer, add your middle name or surname.")
        st.caption("Without signing in you can still practise, but nothing is saved.")


def home():
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="hero">Practise assessing chemotherapy side effects</div>'
                    '<div class="muted">Read a patient record, give your assessment in three short steps, '
                    'and compare it with the guidelines.</div>', unsafe_allow_html=True)
        st.write("")
        sign_in_box()
        st.write("")
        if st.button("Start a case", type="primary", width="stretch"):
            if new_case():
                st.rerun()
            st.warning("No cases match the topic and setting you chose. Choose another under 'Choose a topic'.")
        plan = core.practice_plan(CASES, ss.history) if ss.history else None
        if plan and (plan["repeat"] or plan["weakest"]):
            if st.button("Practise what I got wrong", width="stretch"):
                c = core.pick_targeted(CASES, ss.history, never=study_locked())
                if c:
                    open_case(c.case_id)
                    st.rerun()
            bits = ([f"weakest topic: {plan['weakest']} ({plan['weakest_score']}%)"] if plan["weakest"] else []) + \
                   ([f"{len(plan['repeat'])} case{'s' if len(plan['repeat']) > 1 else ''} to repeat"] if plan["repeat"] else [])
            st.caption("From your results: " + "; ".join(bits) + ".")
        study_box()
        with st.expander("Choose a topic", expanded=ss.f_topic != "All topics" or ss.f_kind != "All settings"):
            # widget keys are separate from the saved choice: Streamlit forgets a widget's value while the start
            # page is not shown, and the choice must survive "Next case"
            ss.w_topic = ss.f_topic
            st.selectbox("Topic", ["All topics"] + core.TOPICS, key="w_topic",
                         on_change=lambda: ss.__setitem__("f_topic", ss.w_topic))
            # only the settings that have cases on this topic are offered, so a choice never leads to nothing
            pool = [c for c in CASES if ss.f_topic == "All topics" or c.topic == ss.f_topic]
            kinds = ["All settings"] + [k for k in core.CASE_TYPE if any(c.dataset == k for c in pool)]
            if ss.f_kind not in kinds:
                ss.f_kind = "All settings"
            ss.w_kind = ss.f_kind
            st.selectbox("Setting", kinds, key="w_kind", format_func=lambda k: core.CASE_TYPE.get(k, k),
                         on_change=lambda: ss.__setitem__("f_kind", ss.w_kind))
            if st.button("Start a case on this topic", type="primary", width="stretch", key="start_topic"):
                if new_case():
                    st.rerun()
                st.warning("No cases match this choice. Choose another setting.")
        with st.expander("Teaching cases"):
            for label, cid in TEACHING_CASES.items():
                if cid in BY_ID and st.button(label, key=f"teach_{cid}", width="stretch"):
                    open_case(cid)
                    st.rerun()
        st.caption("For training only. Not for clinical decisions.")


# ------------------------------------------------------------------ the study (10 fixed cases)
def study_box():
    if not ss.trainee:
        st.caption("Sign in to take part in the study: 10 cases, about 35 minutes.")
        return
    done, nxt = core.study_rows(ss.history), core.next_study_case(ss.history)
    if nxt is None:
        if st.button("My study results", width="stretch"):
            ss.view = "study_end"
            st.rerun()
    elif not done:
        if st.button("Take part in the study (10 cases)", width="stretch"):
            ss.view = "study_intro"
            st.rerun()
    else:
        if st.button(f"Continue the study ({len(done)} of {len(core.STUDY_CASES)} done)", width="stretch"):
            ss.study_level = ss.study_level or core.study_level(ss.history)
            open_case(nxt, "study")
            st.rerun()


def study_intro():
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown('<div class="hero">The study</div>', unsafe_allow_html=True)
        st.markdown(f"""
This study checks whether **ChemoTwin**, a decision-support tool, gives recommendations that agree with
clinical judgement. Your answers are the clinical reference: we compare the decisions ChemoTwin recommends
with the decisions that doctors and doctors in training make on the same cases. **We are evaluating the tool,
not you.**

- **{len(core.STUDY_CASES)} cases**, the same for everyone, in the same order. About 35 minutes.
- Answer each case on your own, without books, websites or AI tools.
- The correct answers and how ChemoTwin handled each case are shown **at the end**, not during the study.
- Each answer is saved when you submit it. You can stop and continue later with the same name and code.

Your answers are reported in a master's thesis **anonymously**, by training level only. Taking part is voluntary,
and you can stop at any time.
""")
        level = st.selectbox("Your training level", core.STUDY_LEVELS, index=None, placeholder="Choose your level")
        agree = st.checkbox("I agree that my answers are used anonymously in this research.")
        b1, b2 = st.columns([1, 2])
        if b1.button("Back", width="stretch"):
            ss.view = None
            st.rerun()
        if b2.button("Start the study", type="primary", width="stretch", disabled=not (level and agree)):
            ss.study_level = level
            open_case(core.STUDY_CASES[0], "study")
            st.rerun()


def study_end():
    rows = core.study_rows(ss.history)
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        st.markdown('<div class="hero">Your study results</div>', unsafe_allow_html=True)
        if len(rows) < len(core.STUDY_CASES):
            st.markdown(f"{len(rows)} of {len(core.STUDY_CASES)} cases answered.")
        fbs = {}
        for cid, r in rows.items():
            try:
                fbs[cid] = core.feedback(BY_ID[cid], json.loads(r["answer"]))
            except Exception:          # an answer saved without its details: show the saved verdict only
                fbs[cid] = None
        safe = sum(bool(r["safe"]) for r in rows.values())
        c1, c2 = st.columns(2)
        c1.metric("Safe plans", f"{safe} of {len(rows)}")
        scores = [r["score"] for r in rows.values() if isinstance(r.get("score"), int)]
        c2.metric("Average agreement with the guidelines", f"{round(sum(scores) / len(scores))}%" if scores else "-")
        st.caption("Open each case to see the guideline answer and how ChemoTwin handled it.")
        for i, cid in enumerate(core.STUDY_CASES, 1):
            if cid not in rows:
                continue
            case, r = BY_ID[cid], rows[cid]
            a, b = st.columns([5, 1])
            a.markdown(f"**{i}. {html.escape(case.topic)}** · {html.escape(case.title)} &nbsp; "
                       + (pill("Safe", "ok") if r["safe"] else pill("Not safe", "bad"))
                       + f' <span class="muted">{r["score"]}% agreement</span>', unsafe_allow_html=True)
            if fbs.get(cid) and b.button("Open", key=f"study_open_{cid}", width="stretch"):
                fb = fbs[cid]
                fb["scope"] = core.terms_in_scope(case)
                ss.results[cid], ss.case_id, ss.mode, ss.view = fb, cid, "review", None
                st.rerun()
        st.write("")
        if st.button("Start page", width="stretch"):
            ss.view = None
            st.rerun()


# ------------------------------------------------------------------ the record as a chart
def show_record(case, mark=None):
    """mark (only after submitting) = the values the guideline answer rests on; they are highlighted."""
    v = core.record_view(case.record)
    pat = mark["pattern"] if mark else None
    out = ['<div class="chart">']
    rows = ""
    for label, content in v["lines"]:
        hl = " hl" if pat and pat.search(label + " " + content) and not content.isupper() else ""
        if label:
            rows += f'<tr class="{hl.strip()}"><td class="k">{html.escape(label)}</td><td>{html.escape(content)}</td></tr>'
        else:
            if rows:
                out.append(f"<table>{rows}</table>")
                rows = ""
            if content.isupper() and len(content) < 60:
                out.append(f'<div class="head">{html.escape(content)}</div>')
            elif " | " in content:
                out.append(f'<div class="plain mono{hl}">{html.escape(content)}</div>')
            else:
                out.append(f'<div class="plain{hl}">{html.escape(content)}</div>')
    if rows:
        out.append(f"<table>{rows}</table>")
    if v["labs"]:
        cols = [c for c in ("baseline", "current", "reference") if c in v["heads"]]
        head = "".join(f"<th>{html.escape(v['heads'][c])}</th>" for c in cols)
        marked = mark["labs"] if mark else set()
        body = "".join(("<tr class='hl'>" if r["name"].lower() in marked else "<tr>") + "<td class='k'>" + html.escape(r["name"]) + "</td>"
                       + "".join(f"<td>{html.escape(r.get(c, '–'))}</td>" for c in cols) + "</tr>" for r in v["labs"])
        out.append(f'<div class="head">LABORATORY</div><table><tr><th></th>{head}</tr>{body}</table>')
        for n in v["notes"]:
            out.append(f'<div class="plain muted">{html.escape(n)}</div>')
    out.append("</div>")
    st.markdown("".join(out), unsafe_allow_html=True)
    if mark and any(m in "".join(out) for m in ('class="hl"', "class='hl'", ' hl"')):
        st.caption("Highlighted: the parts of the record that decide the guideline answer.")
    elif mark:
        st.caption("Nothing in this record is abnormal enough to change the plan.")


# ------------------------------------------------------------------ the case in three steps
def case_page():
    case = BY_ID[ss.case_id]
    done = case.case_id in ss.results
    top, back = st.columns([6, 1])
    study_n = (f"Study case {core.STUDY_CASES.index(case.case_id) + 1} of {len(core.STUDY_CASES)} · "
               if ss.mode == "study" else "")
    top.markdown(f'<div class="case-title">{study_n}{html.escape(case.title)}'
                 + (f' <span class="muted">· case {html.escape(case.case_id)} · {html.escape(case.topic)}</span>' if done else "")
                 + '</div>', unsafe_allow_html=True)
    if ss.mode == "review":
        if back.button("Study results", width="stretch"):
            ss.case_id, ss.view = None, "study_end"
            st.rerun()
    elif back.button("Save and leave" if ss.mode == "study" else "Start page", width="stretch"):
        ss.case_id = None
        st.rerun()
    if done and ss.pipe_open.get(case.case_id):
        show_pipeline(case)
    if st.toggle("Show patient record", value=True, key="show_record"):
        left_col, right_col = st.columns([6, 5], gap="large")
        with left_col:
            show_record(case, core.deciding_values(case, ss.results[case.case_id]) if done else None)
    else:
        right_col = st.container()
    with right_col:
        if done:
            show_feedback(case, ss.results[case.case_id])
        else:
            steps(case)


def show_pipeline(case):
    """After the answer: an animated walk through the five agents on this same case (a recorded run), full width
    at the top of the page, so it is seen as soon as it is opened."""
    run = replay.recorded(case.case_id)
    if not run:
        return
    head, close = st.columns([6, 1])
    head.markdown('<div class="sec">How ChemoTwin handled this case</div>', unsafe_allow_html=True)
    if close.button("Hide", key=f"pipe_hide_{case.case_id}", width="stretch"):
        ss.pipe_open[case.case_id] = False
        st.rerun()
    stp = replay.pipeline_steps(case, run)
    st.components.v1.html(replay.render(stp, replay.load_runs().get("model", "")), height=replay.height(stp),
                          scrolling=False)
    st.caption("This is the full ChemoTwin system. Four agents are rules in code (blue); only the clinical reasoning "
               "agent (orange) is a language model, and its answer is released only after the coordinator's checks. "
               "The run shown was recorded during the evaluation of the system; nothing is generated now.")
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)


def pipeline_invite(case):
    """A clear invitation right under the verdict (the first thing read after submitting)."""
    if replay.load_error():
        st.caption("The ChemoTwin replay is not available: " + replay.load_error() + ".")
        return
    if not replay.recorded(case.case_id) or ss.pipe_open.get(case.case_id):
        return
    st.markdown('<div class="invite"><b>See ChemoTwin work on this same case</b>'
                '<span>Five agents, step by step: what each one checked and how the final answer was reached.</span></div>',
                unsafe_allow_html=True)
    if st.button("▶  Watch how ChemoTwin handled it", key=f"pipe_open_{case.case_id}", type="primary", width="stretch"):
        ss.pipe_open[case.case_id] = True
        st.rerun()


def steps(case):
    k = case.case_id
    step = ss.steps.setdefault(k, 1)
    ans = ss.answers.setdefault(k, {})
    scope = core.terms_in_scope(case)
    st.markdown(f'<div class="step">Step {step} of 3</div>', unsafe_allow_html=True)

    if step == 1:
        st.markdown('<div class="q">What is abnormal?</div>', unsafe_allow_html=True)
        picked = st.pills("Mark every item that is abnormal or cannot be graded. Leave the others unmarked.",
                          scope, selection_mode="multi", format_func=core.TERM_SHORT.get,
                          default=[t for t in ans.get("abnormal", {}) if t in scope], key=f"p_{k}") or []
        grades = {}
        for t in picked:
            prev = ans.get("abnormal", {}).get(t)
            grades[t] = st.radio(f"{core.TERM_SHORT[t]}: CTCAE v6.0 grade", core.ABNORMAL_GRADES, horizontal=True,
                                 index=core.ABNORMAL_GRADES.index(prev) if prev in core.ABNORMAL_GRADES else None,
                                 key=f"g_{k}_{TERMS_REQUIRED.index(t)}")
        heart_opts = list(core.CTRCD_ASYMPTOMATIC)
        heart = st.radio("Heart damage from treatment (ESC 2022)", heart_opts, horizontal=True,
                         format_func=core.CTRCD_ASYMPTOMATIC.get, index=heart_opts.index(ans.get("heart", "none")),
                         key=f"h_{k}")
        if st.button("Next", type="primary", width="stretch"):
            open_ = [core.TERM_SHORT[t] for t in picked if grades[t] is None]
            if open_:
                st.error("Choose a grade for: " + ", ".join(open_) + ".")
            else:
                ans["abnormal"], ans["heart"] = grades, heart
                ss.steps[k] = 2
                st.rerun()

    elif step == 2:
        st.markdown('<div class="q">What is your plan?</div>', unsafe_allow_html=True)
        d_opts, l_opts = list(core.DECISIONS), list(core.LEVELS)
        decision = st.radio("Chemotherapy", d_opts, format_func=core.DECISIONS.get,
                            index=d_opts.index(ans["decision"]) if ans.get("decision") in d_opts else None, key=f"d_{k}")
        level = st.radio("How urgent is it?", l_opts, format_func=core.LEVELS.get,
                         index=l_opts.index(ans["level"]) if ans.get("level") in l_opts else None, key=f"l_{k}")
        b1, b2 = st.columns([1, 3])
        if b1.button("Back", width="stretch"):
            ss.steps[k] = 1
            st.rerun()
        if b2.button("Next", type="primary", width="stretch"):
            if decision is None or level is None:
                st.error("Choose a plan and how urgent it is.")
            else:
                ans["decision"], ans["level"] = decision, level
                ss.steps[k] = 3
                st.rerun()

    else:
        st.markdown('<div class="q">What do you do now?</div>', unsafe_allow_html=True)
        actions = st.pills("Treatment and monitoring (choose any that apply)", list(core.ACTIONS),
                           selection_mode="multi", format_func=core.ACTION_LABELS.get,
                           default=ans.get("actions", []), key=f"a_{k}") or []
        requests_ = st.pills("Results to request (only if something is missing)", list(core.DATA_REQUESTS),
                             selection_mode="multi", format_func=core.DATA_REQUESTS.get,
                             default=ans.get("requests", []), key=f"r_{k}") or []
        b1, b2 = st.columns([1, 3])
        if b1.button("Back", width="stretch"):
            ans["actions"], ans["requests"] = list(actions), list(requests_)
            ss.steps[k] = 2
            st.rerun()
        if b2.button("Submit", type="primary", width="stretch"):
            a = core.simple_assessment(case, ans.get("abnormal", {}), ans.get("heart", "none"), ans["decision"],
                                       ans["level"], list(actions), list(requests_))
            fb = core.feedback(case, a)
            fb["scope"] = scope
            study = ss.mode == "study"
            row = {"time": datetime.now(ZoneInfo("Africa/Cairo")).strftime("%Y-%m-%d %H:%M"), "trainee": ss.trainee,
                   "case_id": k, "topic": case.topic, "case_type": case.case_type, "score": fb["score"],
                   "safe": fb["safe"], "codes": "|".join(fb["codes"]),
                   "session": "study" if study else "practice", "level": ss.study_level if study else "",
                   "seconds": round(time.time() - ss.started.get(k, time.time())),
                   "answer": json.dumps(a, ensure_ascii=False, separators=(",", ":"))}
            ss.history.append(row)
            if ss.trainee:
                save_progress(row)
            if study:
                # no feedback during the study: straight on to the next case, answers at the end
                nxt = core.next_study_case(ss.history)
                if nxt:
                    open_case(nxt, "study")
                else:
                    ss.case_id, ss.view = None, "study_end"
            else:
                ss.results[k] = fb
            st.rerun()


def show_feedback(case, fb):
    cls = "ok" if fb["safe"] else "bad"
    head = "Safe plan" if fb["safe"] else ("Not safe: " + str(fb["n_unsafe"]) + " safety concern" + ("s" if fb["n_unsafe"] > 1 else ""))
    st.markdown(f'<div class="verdict {cls}">{head}<span>{fb["score"]}% agreement with the guidelines</span></div>',
                unsafe_allow_html=True)
    pipeline_invite(case)
    scope_short = {core.TERM_SHORT[t] for t in fb.get("scope", TERMS_REQUIRED)}
    rows = [r for r in fb["rows"] if r["group"] != "grade" or r["item"] in scope_short or not r["ok"]]
    wrong = [r for r in rows if not r["ok"]]
    if not wrong:
        st.markdown("Your assessment matches the guidelines on every item.")
    for r in wrong:
        tag = pill("Unsafe", "bad") if r["significant"] else pill("Small difference", "warn")
        cmp_ = (f'<div class="cmp">You: {html.escape(str(r["yours"]))} · Guidelines: {html.escape(str(r["guideline"]))}</div>'
                if r["yours"] or r["guideline"] else "")
        st.markdown(f'<div class="item {"unsafe" if r["significant"] else ""}"><span class="t">{html.escape(str(r["item"]))}</span> {tag}'
                    f'{cmp_}<div class="why">{html.escape(r["why"])}</div></div>', unsafe_allow_html=True)
    for t in fb.get("tips", []):
        st.caption(t)
    if not ss.trainee:
        st.caption("Not saved: sign in on the start page to keep your progress.")
    with st.expander("Show all details"):
        shown = [r for r in rows if not (r["group"] == "grade" and r["ok"] and r["guideline"] == "0")]
        body = "".join(
            f"<tr><td>{html.escape(str(r['item']))}</td><td>{html.escape(str(r['yours']))}</td>"
            f"<td>{html.escape(str(r['guideline']))}</td><td>"
            + (pill("Correct", "ok") if r["ok"] else pill("Unsafe", "bad") if r["significant"] else pill("Small difference", "warn"))
            + "</td></tr>" for r in shown)
        st.markdown('<table class="review"><tr><th>Item</th><th>Your answer</th><th>Guidelines say</th><th></th></tr>'
                    + body + "</table>", unsafe_allow_html=True)
        if case.what_it_tests:
            st.markdown(f"**About this case.** {html.escape(case.what_it_tests)}")
        if fb.get("under_review"):
            st.caption("This case is based on a published report. Its answer key is still under clinical review.")
    b1, b2 = st.columns(2)
    if b1.button("Retry", width="stretch"):
        open_case(case.case_id)
        st.rerun()
    if b2.button("Next case", type="primary", width="stretch"):
        new_case()
        st.rerun()


def practice():
    if ss.view == "study_intro":
        study_intro()
    elif ss.view == "study_end":
        study_end()
    elif ss.case_id is None or ss.case_id not in BY_ID:
        home()
    else:
        case_page()


def _first_attempts(df):
    """Adds the attempt number per case; averages use first attempts only (a retry has seen the answer)."""
    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["safe"] = df["safe"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["attempt"] = df.groupby("case_id").cumcount() + 1
    return df, df[df["attempt"] == 1]


def progress():
    mine, everyone = st.tabs(["My progress", "All trainees"])
    with mine:
        if ss.trainee:
            st.markdown(f"**{html.escape(core.display_name(ss.trainee))}**")
        else:
            st.caption("Not signed in: results from this session only, not saved. Sign in on the start page.")
        h = ss.history
        if core.next_study_case(h) and core.study_rows(h):
            # mid-study: no verdicts yet, so the study answers stay hidden until all ten are done
            h = [x for x in h if x.get("session") != "study"]
            st.caption("Your study answers appear here after you finish the study.")
        if not h:
            st.markdown('<div class="muted">No cases completed yet.</div>', unsafe_allow_html=True)
        else:
            df, first = _first_attempts(pd.DataFrame(h))
            c1, c2, c3 = st.columns(3)
            c1.metric("Cases completed", len(first))
            c2.metric("Average agreement", f"{round(first['score'].mean())}%")
            c3.metric("Safe plans", f"{round(100 * first['safe'].mean())}%")
            st.caption("First attempts only. Retries are listed below but not counted.")
            sec("Agreement by topic")
            st.bar_chart(first.groupby("topic")["score"].mean(), color="#134E8E")
            sec("Case log")
            show = df.drop(columns=["trainee", "codes", "answer", "level", "seconds"], errors="ignore").rename(columns={
                "time": "Date", "case_id": "Case", "topic": "Topic", "case_type": "Setting",
                "score": "Agreement (%)", "safe": "Safe plan", "attempt": "Attempt", "session": "Session"})
            st.dataframe(show, hide_index=True, width="stretch")
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            st.download_button("Export (CSV)", buf.getvalue(),
                               f"chemotwin_{core.display_name(ss.trainee or 'session').replace(' ', '_')}.csv", "text/csv")
    with everyone:
        code = storage.tutor_code()
        if storage.online() and not code:
            st.caption("Closed: no tutor code is set for this app.")
            return
        if code and st.text_input("Tutor code", type="password", key="tutor_code_in") != code:
            st.caption("For the course tutor.")
            return
        rows, study, unread = [], [], 0
        for name in trainees():
            hist, reached = load_history(name)
            if not reached:
                unread += 1
                continue
            study += [dict(h, trainee=name) for h in core.study_rows(hist).values()]
            d = pd.DataFrame(hist)
            if d.empty:
                continue
            d, first = _first_attempts(d)
            rows.append({"Trainee": name, "Cases": len(first), "Retries": len(d) - len(first),
                         "Average agreement (%)": round(first["score"].mean()),
                         "Safe plans (%)": round(100 * first["safe"].mean()), "Last activity": str(d["time"].iloc[-1])})
        if unread:
            st.warning(f"{unread} record{'s' if unread > 1 else ''} could not be read just now, "
                       "so the table below is incomplete.")
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            if study:
                buf = io.StringIO()
                pd.DataFrame(study).to_csv(buf, index=False)
                st.download_button(f"Download study answers ({len(study)} rows, CSV)", buf.getvalue(),
                                   "chemotwin_study_answers.csv", "text/csv")
            st.caption("First attempts only. " + ("Saved online in the course progress sheet." if storage.online()
                       else "Saved on this computer in trainer/progress, one file per trainee."))
        else:
            st.markdown('<div class="muted">No saved progress yet.</div>', unsafe_allow_html=True)


def guide():
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        sec("How it works")
        st.markdown("""
1. Press **Start a case** and read the patient record.
2. **Step 1:** mark what is abnormal and grade it. **Step 2:** choose your plan and how urgent it is. **Step 3:** choose what you do now.
3. Press **Submit** to see whether your plan is safe and where it differs from the guidelines.

Sign in with your name and a personal code to save your results. Only your first attempt at each case counts.
""")
        sec("Guidelines")
        st.markdown("""
- **NCI CTCAE v6.0** (2025): grading of side effects
- **ESC 2022 Cardio-oncology Guidelines**: heart damage from cancer treatment
- **ESMO 2016**: cardiovascular toxicity of cancer treatment
- **FDA doxorubicin label**: dosing, cumulative dose, liver function and the trastuzumab interval
""")
        st.caption("The cases are written for practice; some are adapted from published case reports. "
                   "For education only, not for decisions about real patients.")


{"Practice": practice, "Progress": progress, "Help": guide}[page]()
