"""
trainer/replay.py
─────────────────
"How ChemoTwin handled this case": an animated walk through the five agents for one case.

Where each part comes from (nothing is invented and no model is called while the trainee watches):
  * Monitoring, Safety and Guideline agents: computed now, by the same deterministic code the pipeline runs.
  * Clinical reasoning agent (the language model) and the Coordinator's checks: the RECORDED run of the full
    system (D_full) from the thesis experiments, stored in trainer/data/pipeline_runs.json.

pipeline_steps(case, run) builds the steps; render(steps, meta) returns one self-contained HTML page.
"""
from __future__ import annotations

import html
import json
import os
from typing import Any, Dict, List, Optional

from agentic.assessment import TERMS_REQUIRED
from trainer import core

RUNS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "pipeline_runs.json")

AGENTS = [
    {"name": "Monitoring", "kind": "code", "role": "Checks that the record holds what a safe decision needs."},
    {"name": "Safety", "kind": "code", "role": "Applies the grading and safety rules to the values in the record."},
    {"name": "Guideline", "kind": "code", "role": "Finds what the guidelines require for these findings."},
    {"name": "Clinical reasoning", "kind": "llm", "role": "A language model writes the assessment and plan."},
    {"name": "Coordinator", "kind": "code", "role": "Checks the plan against the rules; sends it back or releases it."},
]
_cache: Optional[Dict[str, Any]] = None


def load_runs() -> Dict[str, Any]:
    global _cache
    if _cache is None:
        try:
            with open(RUNS_FILE, encoding="utf-8") as f:
                _cache = json.load(f)
        except (FileNotFoundError, ValueError):
            _cache = {"model": "", "runs": {}}
    return _cache


def recorded(case_id: str) -> Optional[Dict[str, Any]]:
    return load_runs().get("runs", {}).get(case_id)


def _row(k: str, v: Any, tone: str = "") -> Dict[str, str]:
    return {"k": k, "v": str(v), "tone": tone}


def _plan(a: Dict[str, Any]) -> str:
    return core.DECISIONS.get(a.get("chemotherapy_decision"), str(a.get("chemotherapy_decision") or "not stated"))


def _urgency(a: Dict[str, Any]) -> str:
    return core.LEVELS.get(a.get("urgency_level"), str(a.get("urgency_level") or "not stated"))


def _abnormal(a: Dict[str, Any]) -> str:
    out = []
    for t in a.get("toxicities") or []:
        g = t.get("grade")
        if g not in (0, "0", None):
            out.append(f"{core.TERM_SHORT.get(t.get('term'), t.get('term'))} {'not evaluable' if g == 'not_evaluable' else 'grade ' + str(g)}")
    return "; ".join(out) or "all items grade 0"


def _monitoring(case: core.Case) -> List[Dict[str, str]]:
    comp = case.ref.get("completeness") or {}
    rows = []
    if case.dataset == "B":
        rows.append(_row("Scope", "Cardiac review only; blood counts, liver and kidney are outside this review."))
    crit = comp.get("critical_missing") or []
    rows.append(_row("Safety-critical data", "Missing: " + ", ".join(crit) if crit else "All present",
                     "warn" if crit else "ok"))
    other = [o.split(" (")[0] for o in comp.get("other_missing") or []]
    if other:
        rows.append(_row("Also not in the record", ", ".join(other)))
    rows.append(_row("Safe to decide now?", "No: request the missing data first" if crit else "Yes",
                     "warn" if crit else "ok"))
    return rows


def _safety(case: core.Case, key: Dict[str, Any]) -> List[Dict[str, str]]:
    ref, rows = case.ref, []
    scope = core.terms_in_scope(case)
    found = False
    for t in TERMS_REQUIRED:
        g = key["grades"][t]["grade"]
        if t not in scope or g == 0:
            continue
        found = True
        label = "Not evaluable" if g == "not_evaluable" else f"Grade {g}"
        rule = key["grades"][t].get("rule") or ""
        rows.append(_row(core.TERM_SHORT[t], label + (": " + core._cut(rule, 150) if rule else ""),
                         "warn" if g != "not_evaluable" else ""))
    if not found:
        rows.append(_row("CTCAE v6.0 grading", "All items grade 0", "ok"))
    esc = ref.get("esc") or {}
    nums = []
    if esc.get("lvef_pct") is not None:
        nums.append(f"LVEF {esc['lvef_pct']:g}%")
    if esc.get("lvef_point_drop"):
        nums.append(f"fall of {esc['lvef_point_drop']:g} points")
    if esc.get("gls_relative_decline_pct"):
        nums.append(f"GLS relative decline {esc['gls_relative_decline_pct']:g}%")
    cat = core.CTRCD_ASYMPTOMATIC.get(key["ctrcd"], str(key["ctrcd"]).replace("_", " "))
    rows.append(_row("Heart (ESC 2022)", cat + (" (" + ", ".join(nums) + ")" if nums else ""),
                     "warn" if key["ctrcd"] in ("mild", "moderate", "severe") else ""))
    dose = ref.get("dose") or {}
    bits = []
    if dose.get("cumulative_doxorubicin_mg_m2") is not None:
        bits.append(f"cumulative doxorubicin {dose['cumulative_doxorubicin_mg_m2']:g} mg/m2"
                    + (" (higher-risk range)" if dose.get("cumulative_dose_higher_risk") else ""))
    band = str(dose.get("hepatic_dose_band") or "")
    if band and "reduce" in band.lower() or "hold" in band.lower() or "omit" in band.lower():
        bits.append("liver: " + band)
    if dose.get("kdigo_aki_criterion_met"):
        bits.append("kidney: creatinine at least 1.5 times baseline")
    if bits:
        rows.append(_row("Dose and organ function", "; ".join(bits), "warn" if len(bits) > 1 else ""))
    for it in ref.get("high_interactions") or []:
        rows.append(_row("Medicines", core._cut(str(it.get("description") or it.get("with")), 200), "warn"))
    if ref.get("chemo_allergy"):
        rows.append(_row("Medicines", "Documented allergy to a drug in the current regimen", "bad"))
    if key["hold_reasons"]:
        rows.append(_row("Hold required", core._cut("; ".join(key["hold_reasons"]), 200), "bad"))
    return rows


def _guideline(case: core.Case, key: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    if key.get("esc_management"):
        rows.append(_row("ESC 2022", key["esc_management"]))
    for a in key["required_actions"]:
        rows.append(_row("Required", a[:1].upper() + a[1:], "warn"))
    if not key["required_actions"]:
        rows.append(_row("Required", "No specific treatment required", "ok"))
    lv = key["levels"]
    if lv:
        rows.append(_row("Urgency", " or ".join(core.LEVELS[x] for x in lv)))
    rows.append(_row("Plan allowed", key["decision"]))
    for c in case.ref.get("applicable_conflicts") or []:
        rows.append(_row("Guidelines differ", str(c).replace("_", " ") + ": must be disclosed in the answer"))
    return rows


def _draft(a: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = [_row("Findings", _abnormal(a)), _row("Plan", _plan(a)), _row("Urgency", _urgency(a))]
    for x in (a.get("actions") or [])[:3]:
        rows.append(_row("Action", core._cut(str(x.get("action") or ""), 170)))
    more = len(a.get("actions") or []) - 3
    if more > 0:
        rows.append(_row("", f"and {more} more action{'s' if more > 1 else ''}"))
    return rows


def _changes(prev: Dict[str, Any], new: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = []
    for label, f in (("Findings", _abnormal), ("Plan", _plan), ("Urgency", _urgency)):
        a, b = f(prev), f(new)
        rows.append(_row(label, f"{a}  →  {b}", "ok") if a != b else _row(label, b + " (unchanged)"))
    na, nb = len(prev.get("actions") or []), len(new.get("actions") or [])
    if na != nb:
        rows.append(_row("Actions", f"{na}  →  {nb}", "ok"))
    return rows


def _check(objections: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    hard = [o for o in objections if o.get("severity") == "hard"]
    soft = [o for o in objections if o.get("severity") != "hard"]
    tidy = lambda o: core._cut(str(o.get("message")).replace(": see tool findings.", ".").replace("Urgency level ", "Urgency ").replace("level ", "urgency "), 190)
    rows = [_row("Must fix", tidy(o), "bad") for o in hard[:4]]
    rows += [_row("Note", tidy(o)) for o in soft[:2]]
    if not objections:
        rows.append(_row("Result", "No objection: grades, values, decision and urgency agree with the rules", "ok"))
    return rows


def pipeline_steps(case: core.Case, run: Dict[str, Any]) -> List[Dict[str, Any]]:
    key = core.guideline_answer(case)
    steps = [
        {"node": 0, "title": "Monitoring agent", "rows": _monitoring(case)},
        {"node": 1, "title": "Safety agent", "rows": _safety(case, key)},
        {"node": 2, "title": "Guideline agent", "rows": _guideline(case, key)},
    ]
    rounds = [r for r in run.get("rounds") or [] if r.get("assessment")]
    if not rounds and run.get("final"):
        rounds = [{"round": 0, "assessment": run["final"], "objections": []}]
    prev = None
    for i, rd in enumerate(rounds):
        a = rd["assessment"]
        if i == 0:
            steps.append({"node": 3, "title": "Clinical reasoning agent: first draft", "rows": _draft(a)})
        else:
            steps.append({"node": 3, "title": f"Clinical reasoning agent: revision {i}", "rows": _changes(prev, a),
                          "loop": True})
        steps.append({"node": 4, "title": "Coordinator: check" + (f" of revision {i}" if i else " of the draft"),
                      "rows": _check(rd.get("objections") or [])})
        prev = a
    released = bool(run.get("released", run.get("status") != "WITHHELD_ESCALATED"))
    n_rev = max(len(rounds) - 1, 0)
    final = run.get("final") or (prev or {})
    if released:
        rows = [_row("Decision", "Released to the clinician", "ok"), _row("Plan", _plan(final)),
                _row("Urgency", _urgency(final)),
                _row("Route", "Approved on the first draft" if n_rev == 0 else
                     f"Approved after {n_rev} revision{'s' if n_rev > 1 else ''}")]
    else:
        rows = [_row("Decision", "Not released: referred to a clinician", "bad"),
                _row("Why", f"Still not safe after {n_rev} revision{'s' if n_rev != 1 else ''}; "
                            "the system does not release a plan it cannot verify")]
    steps.append({"node": 4, "title": "Coordinator: outcome", "rows": rows,
                  "outcome": "released" if released else "withheld"})
    return steps


_ICONS = [
    '<circle cx="11" cy="11" r="6"/><path d="M16 16l5 5"/>',
    '<path d="M12 3l8 3v6c0 5-3.4 8.2-8 9-4.6-.8-8-4-8-9V6z"/><path d="M8.5 12l2.5 2.5L15.5 10"/>',
    '<path d="M3 5c3-1 6-1 9 1 3-2 6-2 9-1v14c-3-1-6-1-9 1-3-2-6-2-9-1z"/><path d="M12 6v14"/>',
    '<path d="M6 3h9l4 4v14H6z"/><path d="M9 11h7M9 15h7M9 7h3"/>',
    '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/>',
]

_PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box} body{margin:0;font-family:"Segoe UI",system-ui,-apple-system,Arial,sans-serif;color:#15202B;background:#fff}
.wrap{border:1px solid #DCE3EA;border-radius:8px;padding:18px 20px 14px 20px}
.top{display:flex;justify-content:space-between;align-items:baseline;flex-wrap:wrap;gap:6px;margin-bottom:14px}
.top .h{font-size:15px;font-weight:650} .top .src{font-size:12px;color:#5B6878}
.track{position:relative;display:grid;grid-template-columns:repeat(5,1fr);margin:6px 0 4px 0}
.line{position:absolute;left:10%;right:10%;top:23px;height:2px;background:#DCE3EA}
.fill{position:absolute;left:10%;top:23px;height:2px;background:#134E8E;width:0;transition:width .7s ease}
.node{position:relative;text-align:center;z-index:1}
.dot{width:48px;height:48px;border-radius:50%;margin:0 auto;background:#fff;border:2px solid #C9D2DC;display:flex;
     align-items:center;justify-content:center;transition:all .35s ease}
.dot svg{width:22px;height:22px;fill:none;stroke:#8A96A3;stroke-width:1.7;stroke-linecap:round;stroke-linejoin:round;transition:stroke .35s}
.node .lab{font-size:12.5px;font-weight:600;color:#6B7785;margin-top:7px;transition:color .35s}
.node .kind{font-size:10.5px;color:#8A96A3;letter-spacing:.04em;text-transform:uppercase}
.node.done .dot{border-color:var(--c);background:var(--c)} .node.done .dot svg{stroke:#fff}
.node.done .lab{color:#15202B}
.node.on .dot{border-color:var(--c);box-shadow:0 0 0 6px var(--halo);animation:pulse 1.8s ease-in-out infinite}
.node.on .dot svg{stroke:var(--c)} .node.on .lab{color:var(--c)}
.code{--c:#134E8E;--halo:rgba(19,78,142,.13)} .llm{--c:#B8600B;--halo:rgba(184,96,11,.15)}
@keyframes pulse{0%,100%{box-shadow:0 0 0 5px var(--halo)}50%{box-shadow:0 0 0 10px var(--halo)}}
.loop{height:34px;margin:0 0 2px 0;position:relative}
.loop svg{position:absolute;left:0;top:0;width:100%;height:34px;overflow:visible}
.loop path{fill:none;stroke:#B8600B;stroke-width:1.6;stroke-dasharray:5 5;opacity:0;transition:opacity .4s}
.loop text{font-size:11px;fill:#B8600B;opacity:0;transition:opacity .4s}
.loop.show path{opacity:1;animation:dash 1.2s linear infinite} .loop.show text{opacity:1}
@keyframes dash{to{stroke-dashoffset:-20}}
.card{border-top:1px solid #E3E8EE;padding-top:12px;min-height:248px}
.card .t{font-size:16px;font-weight:650;margin-bottom:1px} .card .r{font-size:12.5px;color:#5B6878;margin-bottom:10px}
.tag{display:inline-block;font-size:10.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding:1px 7px;
     border-radius:3px;margin-left:8px;vertical-align:2px}
.tag.code{background:#E8EFF8;color:#134E8E} .tag.llm{background:#FBEEDD;color:#8A4A06}
.row{display:grid;grid-template-columns:170px 1fr;gap:10px;padding:6px 0;border-bottom:1px solid #F0F3F6;font-size:13.5px;
     opacity:0;transform:translateY(5px);animation:in .4s ease forwards}
.row .k{color:#5B6878} .row .v{line-height:1.4}
.row.ok .v{color:#1B6B45} .row.warn .v{color:#8A5300} .row.bad .v{color:#A1261C;font-weight:600}
@keyframes in{to{opacity:1;transform:none}}
.out{margin-top:10px;padding:9px 12px;border-radius:6px;font-size:14px;font-weight:650;opacity:0;animation:in .5s ease .5s forwards}
.out.released{background:#E4F2EA;color:#1B6B45} .out.withheld{background:#FBE6E4;color:#A1261C}
.bar{display:flex;align-items:center;gap:8px;margin-top:12px}
button{font:inherit;font-size:13px;padding:5px 12px;border:1px solid #C9D2DC;background:#fff;border-radius:5px;cursor:pointer;color:#15202B}
button:hover{border-color:#134E8E;color:#134E8E} button.p{background:#134E8E;border-color:#134E8E;color:#fff;min-width:74px}
button:disabled{opacity:.4;cursor:default}
.count{font-size:12px;color:#5B6878;margin-left:auto}
.prog{height:3px;background:#EEF1F4;border-radius:2px;margin-top:10px;overflow:hidden} .prog i{display:block;height:3px;background:#134E8E;width:0}
@media (max-width:640px){.dot{width:38px;height:38px}.dot svg{width:18px;height:18px}.line,.fill{top:18px}
 .node .lab{font-size:10.5px}.node .kind{display:none}.row{grid-template-columns:1fr;gap:1px}.row .k{font-size:12px}}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}.row,.out{opacity:1;transform:none}}
</style></head><body><div class="wrap">
<div class="top"><div class="h">How ChemoTwin handled this case</div><div class="src">__SRC__</div></div>
<div class="track"><div class="line"></div><div class="fill" id="fill"></div>__NODES__</div>
<div class="loop" id="loop"><svg viewBox="0 0 100 34" preserveAspectRatio="none">
<path d="M 90 2 C 90 22, 70 22, 70 4" vector-effect="non-scaling-stroke"/></svg>
<svg><text x="80%" y="33" text-anchor="middle" id="looptext">sent back for revision</text></svg></div>
<div class="card" id="card"></div>
<div class="prog"><i id="prog"></i></div>
<div class="bar"><button id="back">Back</button><button id="play" class="p">Pause</button><button id="next">Next</button>
<button id="again">Replay</button><span class="count" id="count"></span></div>
</div>
<script>
const STEPS=__STEPS__, AGENTS=__AGENTS__;
let i=0, playing=true, timer=null, t0=0, dur=0, raf=null;
const $=id=>document.getElementById(id);
const esc=s=>String(s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function show(){
  const s=STEPS[i], a=AGENTS[s.node];
  document.querySelectorAll(".node").forEach((n,k)=>{n.classList.toggle("on",k===s.node);n.classList.toggle("done",k<s.node||(k===s.node&&!!s.outcome));
    if(s.outcome&&k===s.node)n.classList.remove("on");});
  const reached=Math.max(...STEPS.slice(0,i+1).map(x=>x.node));
  document.querySelectorAll(".node").forEach((n,k)=>{if(k<=reached&&k!==s.node)n.classList.add("done");});
  $("fill").style.width=(reached*20)+"%";
  $("loop").classList.toggle("show",!!s.loop);
  let h='<div class="t">'+esc(s.title)+'<span class="tag '+a.kind+'">'+(a.kind==="llm"?"language model":"rules in code")+'</span></div><div class="r">'+esc(a.role)+'</div>';
  s.rows.forEach((r,k)=>{h+='<div class="row '+r.tone+'" style="animation-delay:'+(0.15+k*0.22)+'s"><div class="k">'+esc(r.k)+'</div><div class="v">'+esc(r.v)+'</div></div>';});
  if(s.outcome)h+='<div class="out '+s.outcome+'">'+(s.outcome==="released"?"Released: a verified plan reaches the clinician":"Withheld: the case goes to a clinician without an AI plan")+'</div>';
  $("card").innerHTML=h;
  $("count").textContent="Step "+(i+1)+" of "+STEPS.length;
  $("back").disabled=i===0; $("next").disabled=i===STEPS.length-1;
  schedule();
}
function schedule(){
  clearTimeout(timer); cancelAnimationFrame(raf); $("prog").style.width="0";
  if(!playing||i===STEPS.length-1){ if(i===STEPS.length-1){playing=false;$("play").textContent="Play";} return; }
  dur=2600+STEPS[i].rows.length*1100; t0=performance.now();
  const tick=()=>{const p=Math.min(1,(performance.now()-t0)/dur);$("prog").style.width=(p*100)+"%";if(p<1&&playing)raf=requestAnimationFrame(tick);};
  raf=requestAnimationFrame(tick);
  timer=setTimeout(()=>{i++;show();},dur);
}
$("next").onclick=()=>{if(i<STEPS.length-1){i++;show();}};
$("back").onclick=()=>{if(i>0){i--;show();}};
$("again").onclick=()=>{i=0;playing=true;$("play").textContent="Pause";show();};
$("play").onclick=()=>{playing=!playing;$("play").textContent=playing?"Pause":"Play";if(playing&&i===STEPS.length-1)i=0;show();};
show();
</script></body></html>"""


def render(steps: List[Dict[str, Any]], model: str = "") -> str:
    nodes = "".join(
        f'<div class="node {a["kind"]}"><div class="dot"><svg viewBox="0 0 24 24">{_ICONS[k]}</svg></div>'
        f'<div class="lab">{html.escape(a["name"])}</div>'
        f'<div class="kind">{"language model" if a["kind"] == "llm" else "rules in code"}</div></div>'
        for k, a in enumerate(AGENTS))
    src = ("ChemoTwin full system (all five agents, the safest set-up), recorded run"
           + (f" · the clinical reasoning agent used {html.escape(model)}" if model else ""))
    safe = lambda o: json.dumps(o, ensure_ascii=False).replace("</", "<\\/")
    return (_PAGE.replace("__NODES__", nodes).replace("__SRC__", src)
            .replace("__STEPS__", safe(steps)).replace("__AGENTS__", safe(AGENTS)))


def height(steps: List[Dict[str, Any]]) -> int:
    return 330 + 34 * max(len(s["rows"]) + (2 if s.get("outcome") else 0) for s in steps) + 40
