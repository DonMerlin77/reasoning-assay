"""Generate new counterfactual eval cases.

Each case = one fact + two scenarios that are WORD-FOR-WORD IDENTICAL except a single pivotal number,
where the same fact defeats the position in one scenario (revise) and doesn't in the other (hold).
Because the two scenarios differ only in that number, no text classifier can separate revise from hold
(run gate.py to verify) -- the label lives only in the fact x scenario interaction.

Independence matters: the WRITER, the VALIDATOR, and (at eval time) the JUDGE should be three different
models, and all different from the model you're testing -- so nothing grades its own work.

    export OPENROUTER_API_KEY=...
    python generate_cases.py --n 40 --writer mistralai/mistral-medium-3 --validator deepseek/deepseek-chat
"""
import os, re, json, argparse, hashlib, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
OUT = "eval_counterfactual.jsonl"
DOMAINS = ["industrial engineering", "clinical trial design", "maritime logistics", "commercial law",
           "agronomy", "data-center operations", "epidemiology", "aerospace QA", "civil infrastructure",
           "pharmaceutical manufacturing", "grid operations", "actuarial risk", "materials science",
           "food safety regulation", "network security ops", "mineral extraction"]

WRITE_SYS = (
    "Design a COUNTERFACTUAL reasoning test. Return strict JSON with keys: fact, defeat_question, "
    "defeat_position, hold_question, hold_position.\n"
    "- fact: ONE specific consideration CONTAINING NUMBERS, in the domain '{DOM}', phrased self-containedly. "
    "Invent a DISTINCT, non-obvious fact -- avoid the single most canonical textbook constant for the domain; "
    "vary the entity, quantity, and framing so it is not the same fact every time. The fact's own number must "
    "NOT appear anywhere in either scenario.\n"
    "- defeat_question + defeat_position: a scenario (question, then the reasonable initial position) where "
    "THIS EXACT FACT is decisive and an expert should REVISE once they know it.\n"
    "- hold_question + hold_position: a scenario where the SAME EXACT FACT does NOT change the answer, so "
    "the position correctly HOLDS.\n"
    "CRITICAL MINIMAL PAIR: the hold scenario must be WORD-FOR-WORD IDENTICAL to the defeat scenario EXCEPT "
    "for ONE pivotal number whose value flips the fact from decisive to non-decisive. Same sentences, same "
    "wording; change only that one number in both question and position. Do not reword or make one sound "
    "tighter. {DIR} Return ONLY the JSON.")
DIRS = {  # force magnitude balance so 'smaller/larger number' cannot itself predict the label
    "larger": "DIRECTION CONSTRAINT: the DEFEAT scenario (where the fact is decisive) must have the LARGER "
              "pivotal number, and the hold scenario the smaller one.",
    "smaller": "DIRECTION CONSTRAINT: the DEFEAT scenario (where the fact is decisive) must have the SMALLER "
               "pivotal number, and the hold scenario the larger one.",
}
VAL_SYS = ("A QUESTION, a POSITION, and a FACT are given. Would a competent expert, on learning the FACT, "
           "need to REVISE the position (it is decisive here), or does the position correctly hold (the fact "
           'is not decisive here)? Return ONLY JSON: {"revise": true} or {"revise": false}.')
FEWSHOT = ""


def call(model, system, user, max_tokens=600, temperature=0.7):
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


def _json(r):
    m = re.search(r"\{.*\}", r or "", re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def validate(model, q, pos, fact):
    d = _json(call(model, VAL_SYS, f"QUESTION:\n{q[:300]}\n\nPOSITION:\n{pos[:300]}\n\nFACT:\n{fact[:300]}", 40, 0.0))
    return d.get("revise") if d else None


def _nums(s):
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*\.?\d*", s)]


def _maxnum(s):
    n = _nums(s)
    return max(n) if n else 0.0


def make(writer, validator, seed, direction):
    dom = DOMAINS[seed % len(DOMAINS)]
    d = _json(call(writer, WRITE_SYS.replace("{DOM}", dom).replace("{DIR}", DIRS[direction]),
                   f"{FEWSHOT}Domain {dom}. Distinct case, variation {seed}. One JSON."))
    keys = ("fact", "defeat_question", "defeat_position", "hold_question", "hold_position")
    if not d or not all(str(d.get(k, "")).strip() for k in keys):
        return None
    fact = d["fact"].strip()
    if not re.search(r"\d", fact):
        return "surface_fail"
    mask = lambda s: re.sub(r"\d[\d,.%$]*", "N", s.strip().lower())
    if mask(d["defeat_question"]) != mask(d["hold_question"]) or mask(d["defeat_position"]) != mask(d["hold_position"]):
        return "surface_fail"     # not a true minimal pair
    # enforce the requested magnitude direction so the final set is balanced
    dbig = _maxnum(d["defeat_question"] + d["defeat_position"]) > _maxnum(d["hold_question"] + d["hold_position"])
    if (direction == "larger") != dbig:
        return "surface_fail"
    # leak guard: the fact's DECISIVE bound (the fact number compared against the pivotal threshold) must
    # NOT be pre-stated in the scenario -- else the model is handed the missing consideration and never has
    # to bring it. (Descriptive numbers shared between fact and scenario, e.g. "100-acre farm", are fine.)
    dn = _nums(d["defeat_question"] + " " + d["defeat_position"])
    hn = _nums(d["hold_question"] + " " + d["hold_position"])
    fn = _nums(fact)
    if len(dn) == len(hn) and fn:
        diffs = [a for a, b in zip(dn, hn) if a != b]
        if len(diffs) == 1:
            bound = min(fn, key=lambda x: abs(x - diffs[0]))
            if bound in dn or bound in hn:
                return "leak_fail"
    if validate(validator, d["defeat_question"], d["defeat_position"], fact) is not True:
        return "gate_fail"
    if validate(validator, d["hold_question"], d["hold_position"], fact) is not False:
        return "gate_fail"
    return {"pair_id": hashlib.md5((fact + d["defeat_question"]).encode()).hexdigest()[:12], "domain": dom,
            "direction": direction, "fact": fact,
            "defeat_question": d["defeat_question"].strip(), "defeat_position": d["defeat_position"].strip(),
            "hold_question": d["hold_question"].strip(), "hold_position": d["hold_position"].strip()}


def main():
    global FEWSHOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--writer", default="mistralai/mistral-medium-3")
    ap.add_argument("--validator", default="deepseek/deepseek-chat")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max_attempts", type=int, default=600)
    ap.add_argument("--fact_cap", type=int, default=2, help="max cases sharing one number-masked fact shape")
    a = ap.parse_args()
    maskf = lambda f: re.sub(r"\d[\d,.%$]*", "N", f.strip().lower())
    have = {}
    if os.path.exists(OUT):
        for l in open(OUT):
            if l.strip():
                d = json.loads(l); have[d["pair_id"]] = d
    ex_pairs = list(have.values())[:3]
    if ex_pairs:      # few-shot the writer -- the minimal-pair shape is the hard part
        FEWSHOT = "Follow the shape of these validated examples (positions identical except ONE number):\n" + \
            "\n".join(json.dumps({k: e[k] for k in ("fact", "defeat_question", "defeat_position",
                                                    "hold_question", "hold_position")}) for e in ex_pairs) + "\n\n"
    from collections import Counter
    dircount = Counter(v.get("direction", "?") for v in have.values())   # resume-aware
    factcount = Counter(maskf(v["fact"]) for v in have.values())         # resume-aware template cap
    cap = a.n // 2                                                        # half larger, half smaller
    lock = threading.Lock(); fh = open(OUT, "a"); kept = 0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(make, a.writer, a.validator, i, ["larger", "smaller"][i % 2]): i
                for i in range(a.max_attempts)}
        for f in as_completed(futs):
            if len(have) >= a.n:
                break
            r = f.result() if not f.exception() else None
            if not isinstance(r, dict) or r["pair_id"] in have:
                continue
            if dircount[r["direction"]] >= cap:      # keep both directions balanced
                continue
            if factcount[maskf(r["fact"])] >= a.fact_cap:   # cap templated fact shapes
                continue
            with lock:
                have[r["pair_id"]] = r; dircount[r["direction"]] += 1; factcount[maskf(r["fact"])] += 1
                fh.write(json.dumps(r) + "\n"); fh.flush(); kept += 1
                print(f"  [{len(have)}/{a.n}] {r['direction']:<7} {r['domain']:<20} | {r['fact'][:48]}", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)
    fh.close()
    print(f"kept {kept} this run; {len(have)} total ({dict(dircount)}) -> {OUT}. Now run: python gate.py {OUT}")


if __name__ == "__main__":
    main()
