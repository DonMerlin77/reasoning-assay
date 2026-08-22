# Reasoning Assay — a shortcut-proof reasoning eval

Most evals that try to measure "reasoning" can be gamed. The moment you build a test, a model can find
a surface shortcut — a keyword, a register, a length cue — that passes it without reasoning. Block one
shortcut and the correlation just moves to the next feature. It's a ladder, and it goes deeper than you'd
expect (word choice → tone → length → which specific words carry meaning).

**Reasoning Assay removes the ladder entirely** by changing *what the label depends on*. Instead of two
different questions where the wording leaks the answer, each item is **one fact quoted verbatim into two
near-identical scenarios**:

- in the **defeat** scenario, the fact genuinely defeats the stated position → the model should **REVISE**
- in the **hold** scenario, the same fact doesn't bear on the decision → the model should **HOLD**

The two scenarios are **word-for-word identical except a single pivotal number**. So there is nothing on
the surface to key on — the label lives only in the *interaction* between the fact and the scenario, and
an interaction is not a property any single-text classifier can read. The model has to actually reason
about whether the fact bites in that specific situation. There's no shortcut to grab.

## Scoring

Balanced accuracy = mean of HOLD-accuracy and REVISE-accuracy, so a model that just holds everything
lands at **50 (chance)**, not a fake-high score. An **independent judge model** decides only whether the
answer changed position — never the model grading itself.

## It's validated

The eval was stress-tested from both ends before being trusted:

- **Positive control** — a strong reasoner should ace it, and does. `gpt-4o` scores **~91%** on the
  shipped cases (run via `run_eval.py`), vs weak models at chance.
- **Negative control** — a surface-shortcut classifier is helpless: content, register, and style all
  score at **chance** (`gate.py`, leave-pair-out, numbers stripped). And the pivotal number's magnitude
  is **balanced** (the defeat scenario is the larger number in half the pairs, smaller in the other half —
  `gate.py` confirms 50/50), so "smaller number → revise" carries no signal either. The only thing that
  differs is the number's *meaning in context*, which requires reasoning.
- **Label audit** — an independent model verified the cases are correctly labeled (~99%).
- **Two adversarial falsification passes** tried to break the design and were answered.

And it produces a clean signal across model sizes and families (same-construction cases):

| model | balanced accuracy |
|---|---|
| 1.5B | ~52% (chance) |
| 3B | ~56% (still ~chance — defaults to holding) |
| 7B | **~94%** |
| 72B | **~97%** |
| gpt-4o | **~91%** |

Reasoning-under-pressure emerges as a **cliff between 3B and 7B** — a stock 7B basically solves it with no
training, everything below sits at chance. The ~40-point gap between strong and weak models on the *same
cases* is the eval doing its job: measuring reasoning, not surface. (The sweep across sizes was run on
same-construction cases; the positive control is reconfirmed on the exact shipped set at ~91%.)

> Honest scope: no eval is provably "unfoolable." The defensible claim is that strong reasoners score
> ~90%+, while weak models *and* surface-shortcut classifiers both score at chance — so the number
> reflects reasoning, not wording. Try to break it; that's what `gate.py` is for.

## Usage

```bash
export OPENROUTER_API_KEY=...          # any OpenAI-compatible endpoint works with small tweaks

python run_eval.py --model openai/gpt-4o
python run_eval.py --model qwen/qwen-2.5-7b-instruct --judge deepseek/deepseek-chat
python gate.py eval_counterfactual.jsonl        # verify the cases have no surface leak
python generate_cases.py --n 40                 # make more cases (writer/validator configurable)
```

The judge must be a **different** model than the one under test (no self-grading), and ideally different
again from the writer/validator used to make the cases.

## Files

- `eval_counterfactual.jsonl` — the cases (one fact, two minimal-pair scenarios each)
- `run_eval.py` — score any model
- `gate.py` — the shortcut-leak gate (the eval's own integrity check)
- `generate_cases.py` — make more cases

## Limitations

- No eval is provably unfoolable; this one is *validated*, not proven perfect. The defensible claim is
  strong reasoners ~90%+, weak models and surface classifiers at chance.
- The scale table mixes model families at some points (3B is Llama, 7B+ Qwen). The gap is far too large
  to be a family effect, but a same-family sweep would pin the cliff more precisely.

## Credits

Built by **Thomas Semrad**, with **Claude (Anthropic)** as a development collaborator — pair-designing the
methodology, the falsification passes, and the validation.

MIT licensed. Built to test whether a small model is actually reasoning or just bluffing — the thing you
need to know before you trust one running unsupervised.
