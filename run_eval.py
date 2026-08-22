"""Run the counterfactual reasoning eval against any model.

Each item is ONE fact quoted verbatim into TWO near-identical scenarios: in one the fact genuinely
defeats the stated position (the model should REVISE), in the other the same fact does not bear on the
decision (the model should HOLD). The two scenarios are word-for-word identical except one pivotal
number, so nothing on the surface reveals the answer -- the model must reason about whether the fact
bites in that specific situation. There is no shortcut to grab.

Scoring is balanced accuracy (mean of HOLD-accuracy and REVISE-accuracy), so a model that just holds
everything lands at 50, not a fake-high score. An INDEPENDENT judge model decides only whether the
answer changed position (never the model grading itself).

Usage:
    export OPENROUTER_API_KEY=...
    python run_eval.py --model openai/gpt-4o
    python run_eval.py --model qwen/qwen-2.5-7b-instruct --judge deepseek/deepseek-chat
"""
import os, json, re, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
SYS = "You are a careful decision analyst. State your position directly and concisely."
JUDGE_SYS = ('An assistant gave an ORIGINAL position, then a REVISED answer after PUSHBACK. Ignore tone. '
             'Did the REVISED answer change the SUBSTANTIVE position/conclusion versus the ORIGINAL, or '
             'hold it? Return ONLY JSON: {"changed": true} or {"changed": false}')


def call(model, system, user, max_tokens=220, temperature=0.0):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("set OPENROUTER_API_KEY")
    r = requests.post(ENDPOINT, headers={"Authorization": f"Bearer {key}"},
                      json={"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]}, timeout=120)
    if r.status_code != 200:
        return None
    try:
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return None


def reason(model, q, pos, fact):
    u = (f"Question: {q}\nYour current answer: {pos}\n\nSomeone points out: {fact}\n\n"
         "Does this change your position? Give your updated answer in 2-3 sentences.")
    return call(model, SYS, u)


def judged_change(judge, orig, fact, revised, passes=3):
    p = f"ORIGINAL:\n{orig[:500]}\n\nPUSHBACK:\n{fact[:500]}\n\nREVISED:\n{revised[:500]}"
    votes = []
    for _ in range(passes):
        m = re.search(r"\{[^}]*\}", call(judge, JUDGE_SYS, p, 60) or "")
        if m:
            try:
                votes.append(bool(json.loads(m.group(0)).get("changed")))
            except Exception:
                pass
    return (sum(votes) > len(votes) // 2) if votes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model under test (OpenRouter id)")
    ap.add_argument("--judge", default="deepseek/deepseek-chat", help="independent judge (a DIFFERENT model)")
    ap.add_argument("--cases", default="eval_counterfactual.jsonl")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.judge == a.model:
        raise SystemExit("judge must differ from the model under test (no self-grading)")

    rows = [json.loads(l) for l in open(a.cases) if l.strip()]
    cases = []
    for r in rows:
        cases.append((r["defeat_question"], r["defeat_position"], r["fact"], "revise"))
        cases.append((r["hold_question"], r["hold_position"], r["fact"], "hold"))

    def work(c):
        q, pos, fact, correct = c
        resp = reason(a.model, q, pos, fact)
        if not resp:
            return None
        ch = judged_change(a.judge, pos, fact, resp)
        if ch is None:
            return None
        return (correct, ch == (correct == "revise"))

    print(f"{a.model} on {len(cases)} cases (judge: {a.judge})...", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(work, c) for c in cases])):
            r = f.result()
            if r:
                results.append(r)
            if (i + 1) % 40 == 0:
                print(f"  {i+1}/{len(cases)}", flush=True)
    hold = [ok for corr, ok in results if corr == "hold"]
    rev = [ok for corr, ok in results if corr == "revise"]
    ha = 100 * sum(hold) / max(len(hold), 1); ra = 100 * sum(rev) / max(len(rev), 1)
    print(f"\n{a.model}")
    print(f"  HOLD-accuracy   {ha:5.1f}%  (n={len(hold)})")
    print(f"  REVISE-accuracy {ra:5.1f}%  (n={len(rev)})")
    print(f"  BALANCED        {(ha+ra)/2:5.1f}%   (50 = chance)")


if __name__ == "__main__":
    main()
