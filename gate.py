"""Shortcut-leak gate: checks no LEARNABLE shortcut separates the two scenarios.

Two layers:
  1. TEXT lint (content/register/style, numbers stripped). These are ~50% BY CONSTRUCTION -- the
     generator enforces number-mask-identity, so a difference here only means a hand-edit slipped in.
     Useful lint, not proof.
  2. NUMBERS-AWARE probe (the real negative control): a classifier given the PIVOTAL number (the one that
     differs between the two scenarios), the fact's bound, and the fact's keywords, evaluated leave-PAIR-out
     AND leave-FACT-out. If a learnable numeric/keyword rule separates the classes it fires here. It should
     be ~chance, because the magnitude direction is balanced (~45/45) so no fixed rule works.

What is NOT a leak: a hand-written solver that actually does the threshold comparison scores high -- that's
the task's depth (single-constraint revision), not a shortcut. See the README Scope note.

    python gate.py eval_counterfactual.jsonl
"""
import json, sys, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_score, GroupKFold
from scipy.sparse import hstack, csr_matrix

KILL = 0.65
NUMPAT = re.compile(r"\d[\d,]*\.?\d*")


def _nums(s):
    return [float(x.replace(",", "")) for x in NUMPAT.findall(s)]


def _strip(s):
    return re.sub(r"\d[\d,.%$]*", " ", s)


def _pivotal(r):
    d = _nums(r["defeat_question"] + " " + r["defeat_position"])
    h = _nums(r["hold_question"] + " " + r["hold_position"])
    if len(d) != len(h):
        return None, None
    diffs = [(a, b) for a, b in zip(d, h) if a != b]
    return diffs[0] if len(diffs) == 1 else (None, None)


def run(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    fails = []

    # 1. TEXT lint
    tx, y, g = [], [], []
    for r in rows:
        tx.append(_strip(r["defeat_question"] + " " + r["defeat_position"])); y.append(1); g.append(r["pair_id"])
        tx.append(_strip(r["hold_question"] + " " + r["hold_position"])); y.append(0); g.append(r["pair_id"])
    y = np.array(y); gkf = GroupKFold(5)
    print(f"{len(rows)} pairs.")
    print("TEXT lint (numbers stripped -- ~50% by construction, so only catches hand-edits):")
    for name, X in [("content tf-idf", TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(tx)),
                    ("register (func-words)", CountVectorizer(vocabulary=sorted(ENGLISH_STOP_WORDS)).fit_transform(tx))]:
        acc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=gkf, groups=g).mean()
        print(f"  {name:<24} {100*acc:5.1f}%{'  <-- hand-edit?' if acc >= KILL else ''}")

    # 2. NUMBERS-AWARE probe (the real negative control)
    facts, feats, yy, pid, fid, bad = [], [], [], [], [], 0
    maskf = lambda f: re.sub(r"\d[\d,.%$]*", "N", f.strip().lower())
    for r in rows:
        dp, hp = _pivotal(r)
        if dp is None:
            bad += 1; continue
        fn = _nums(r["fact"]); fb = fn[0] if fn else 0.0
        for p, lab in [(dp, 1), (hp, 0)]:
            facts.append(r["fact"]); feats.append([p, fb, p - fb]); yy.append(lab)
            pid.append(r["pair_id"]); fid.append(maskf(r["fact"]))
    yy = np.array(yy)
    Xn = hstack([CountVectorizer(min_df=2).fit_transform(facts), csr_matrix(np.array(feats))]).tocsr()
    print("NUMBERS-AWARE probe (pivotal number + fact bound + fact keywords) -- the real control:")
    for label, grp in [("leave-pair-out", pid), ("leave-fact-out", fid)]:
        acc = cross_val_score(LogisticRegression(max_iter=3000), Xn, yy, cv=GroupKFold(5), groups=grp).mean()
        mark = "  <-- LEARNABLE SHORTCUT" if acc >= KILL else ""
        print(f"  {label:<24} {100*acc:5.1f}%{mark}")
        if acc >= KILL:
            fails.append(label)

    # diagnostics (positional pivotal, not max number)
    dbig = sum(1 for r in rows for dp, hp in [_pivotal(r)] if dp is not None and dp > hp)
    resolved = sum(1 for r in rows if _pivotal(r)[0] is not None)
    distinct = len(set(maskf(r["fact"]) for r in rows))
    print(f"\nDIAGNOSTICS: pivotal magnitude defeat>hold {dbig}/{resolved} (want ~50%); "
          f"distinct facts {distinct}/{len(rows)} (templating if <<); unresolved pivots {bad}")
    print("\nGATE FAILED: a learnable shortcut separates the classes." if fails else
          "GATE PASSED: no learnable shortcut -- only correctly doing the comparison scores high.")
    return not fails


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1] if len(sys.argv) > 1 else "eval_counterfactual.jsonl") else 1)
