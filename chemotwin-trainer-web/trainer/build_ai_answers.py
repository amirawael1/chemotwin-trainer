"""
trainer/build_ai_answers.py
───────────────────────────
Makes the small file trainer/ai_answers.json from viewer/viewer_data.json, so the online trainer can
show "how the AI systems answered this case" without the 4 MB viewer file.

  python trainer/build_ai_answers.py

Keeps, for each case and each system, only the status, the safety verdict, the final plan and urgency,
and the scored errors. Reads saved results only; no model calls.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "viewer", "viewer_data.json")
OUT = os.path.join(ROOT, "trainer", "ai_answers.json")
ARMS = ("A_pure", "B_rag", "C_tools", "D_full")


def main():
    with open(SRC, encoding="utf-8") as f:
        data = json.load(f)
    cases = []
    for c in data.get("cases", []):
        runs = {}
        for arm in ARMS:
            r = (c.get("runs") or {}).get(arm)
            if not r:
                continue
            final = r.get("final") or {}
            runs[arm] = {"status": r.get("status"), "unsafe": bool(r.get("unsafe")),
                         "final": {"chemotherapy_decision": final.get("chemotherapy_decision"),
                                   "urgency_level": final.get("urgency_level")},
                         "errors": [{"code": e.get("code"), "significant": bool(e.get("significant"))}
                                    for e in r.get("errors") or []]}
        if runs:
            cases.append({"case_id": c["case_id"], "runs": runs})
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"model": data.get("model"), "arms": list(ARMS), "cases": cases}, f, separators=(",", ":"))
    print(f"{len(cases)} cases -> {OUT} ({os.path.getsize(OUT) // 1024} kB)")


if __name__ == "__main__":
    main()