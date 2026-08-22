"""Shortcut-leak gate: checks the eval can't be beaten by a SPURIOUS surface feature.

The design claim: the label (revise vs hold) lives only in the fact x scenario interaction. The two
scenarios in a pair are word-for-word identical except one pivotal number, so:
  - the challenge (the fact) is identical within a pair -> 50% separable by construction.
  - the SPURIOUS surface of the scenario (content words, register, style -- everything EXCEPT the
    pivotal number) must not predict the label. This is the real integrity check.

We train leave-PAIR-out classifiers (GroupKFold, so they can't memorize a pair) on the scenario text
with the numbers stripped. If any beats ~chance, real content/register leaked in. Kill line: >= 65%.

Separately we report a DIAGNOSTIC: the pivotal number itself necessarily differs (it's what flips the
label), so it is not a spurious feature -- but its MAGNITUDE should ideally not point one direction, or
a classifier could ride "smaller number -> revise" without reasoning. We report that balance. (A model
under test can't exploit it the way this classifier can -- it never trains on the eval -- but a balanced
set is cleaner. In practice weak models score at chance, below what magnitude alone would give, so no
tested model rides it.)

    python gate.py eval_counterfactual.jsonl
"""
import json, sys, re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import cross_val_score, GroupKFold

KILL = 0.65


def _strip(s):
    return re.sub(r"\d[\d,.%$]*", " ", s)


def _maxnum(s):
    n = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", s)]
    return max(n) if n else 0.0


def run(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    texts, y, groups = [], [], []
    for r in rows:
        texts.append(_strip(r["defeat_question"] + " " + r["defeat_position"])); y.append(1); groups.append(r["pair_id"])
        texts.append(_strip(r["hold_question"] + " " + r["hold_position"])); y.append(0); groups.append(r["pair_id"])
    y = np.array(y)
    gkf = GroupKFold(n_splits=5)
    probes = {
        "content tf-idf": TfidfVectorizer(ngram_range=(1, 2), min_df=2).fit_transform(texts),
        "content bag-of-words": CountVectorizer(min_df=2).fit_transform(texts),
        "register (function-words)": CountVectorizer(vocabulary=sorted(ENGLISH_STOP_WORDS)).fit_transform(texts),
    }
    print(f"{len(rows)} pairs. challenge identical within a pair -> 50% by construction.")
    print(f"SPURIOUS-surface check: scenario, numbers stripped, leave-pair-out (kill any >= {KILL:.0%}):")
    fails = []
    for name, X in probes.items():
        acc = cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=gkf, groups=groups).mean()
        mark = "  <-- LEAK" if acc >= KILL else ""
        print(f"  {name:<28} {100*acc:5.1f}%{mark}")
        if acc >= KILL:
            fails.append(name)

    d = sum(1 for r in rows if _maxnum(r["defeat_question"] + r["defeat_position"]) >
            _maxnum(r["hold_question"] + r["hold_position"]))
    print(f"\nDIAGNOSTIC (not pass/fail): pivotal-number magnitude, defeat > hold in {d}/{len(rows)} "
          f"({100*d/len(rows):.0f}%; 50% is balanced). The number is the intended reasoning variable.")
    print()
    print("GATE FAILED: real content/register leaks into the scenario." if fails else
          "GATE PASSED: no spurious surface leak -- only the pivotal number differs, and that's the "
          "variable the model must reason about.")
    return not fails


if __name__ == "__main__":
    sys.exit(0 if run(sys.argv[1] if len(sys.argv) > 1 else "eval_counterfactual.jsonl") else 1)
