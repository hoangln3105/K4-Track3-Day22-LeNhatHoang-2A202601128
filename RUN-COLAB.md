# Run plan — Colab T4

Personal working notes, not part of the submission. Delete before pushing if you like.

**Why Colab:** this laptop has an RTX 3050 Ti with **4 GB VRAM**. `HARDWARE-GUIDE.md`
needs ≥ 12 GB for the T4 tier (Qwen2.5-3B ≈ 10 GB, because DPO holds *both* the policy
and the frozen reference model). Even the lightest row in their table (1.5B, ≈ 5 GB)
does not fit. Local training is not an option — free Colab T4 is.

---

## 0. Before you start

1. Upload **`colab/Lab22_DPO_T4.ipynb`** to Colab (File → Upload notebook).
   The notebook is self-contained: it creates `/content/lab22` itself and does **not**
   clone this repo, so the patched dataset id travels inside the file. No push needed first.
2. **Runtime → Change runtime type → T4 GPU.** Cell 4 asserts on this and will stop you.
3. *(Optional, for NB4)* If you want the automated judge instead of hand-judging 8 prompts,
   add a key **before** running the NB4 section:
   ```python
   import os
   os.environ["OPENAI_API_KEY"] = "sk-..."      # or ANTHROPIC_API_KEY
   ```
   Without a key NB4 still writes `judge_results.json`, but every row is
   `"winner": "tie", "justification": "MANUAL — fill in"` — you have to fill them yourself.
   The rubric accepts manual judging; it does not accept leaving the placeholders.

---

## 1. Run order

| Stage | Time (T4) | Produces |
|---|---:|---|
| A. Setup + pip install | ~5 min | — |
| NB1 SFT-mini | ~10 min | `adapters/sft-mini/` · `02-sft-loss.png` |
| NB2 preference data | ~2 min | `data/pref/train.parquet` |
| NB3 **DPO** | ~30 min | `adapters/dpo/` · `dpo_metrics.json` · `03-dpo-reward-curves.png` |
| NB4 compare + judge | ~10 min | `side_by_side.jsonl` · `judge_results.json` · `04-side-by-side-table.png` |
| **⏹ CORE DONE export** | ~1 min | `lab22-artifacts.zip` → Drive (or download) |

Core stops at NB4. NB5 (GGUF) and NB6 (benchmark) are bonus only — skip on a first pass.

## Leaving it running while you do something else

The run is ~50 minutes of core. To walk away safely:

1. **Plug the laptop in.** This machine is set to never sleep on AC, but sleeps after
   **3 minutes on battery** — and a sleep drops the network, which disconnects Colab and
   recycles the VM. (`powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE` to re-check.)
2. **Run the Drive-mount cell** in section A (cell 8) and approve the OAuth popup *before*
   you leave. With Drive mounted, the export writes to
   `MyDrive/lab22-artifacts/lab22-artifacts.zip` instead of firing a browser download —
   a download needs a focused tab and silently does nothing if you are away.
3. **Leave the tab open** in its own window. Free Colab needs the browser connected;
   closing the tab or hibernating the machine ends the session.
4. Then `Run all` is fine — the guard cell after the export stops the run before the
   bonus stages (see below).

If the runtime dies mid-run anyway, nothing is recoverable except what already reached
Drive — so the export sits immediately after NB4, not at the end of the file.

## `Run all` is safe now, but only because of the guard

The notebook stitches all six stages into one file, so a plain Run all would keep going
into NB5 (GGUF merge, multi-GB, needs a llama.cpp build) and NB6 (`lm-eval`, 30+ min) —
both bonus-only, both able to exhaust free-Colab disk or wall-clock.

The cell right after the export (`STOP_AFTER_CORE = True`, cell 93) raises `CoreComplete`
to halt the run there. **That red cell is intentional, not a failure** — the message says
so. Flip it to `False` if you want the bonus stages in the same sitting.

Colab writes to `/content/lab22`, which is wiped when the runtime recycles; `make verify`
reads the same files out of this repo. Unzip the artifact zip into the **repo root**,
keeping folder structure. The export cell prints the `verify.py` checklist *before*
saving, so you find out what is missing while the runtime is still alive.

If you run the bonus stages later, use the `Re-export` cell at the very bottom to rebuild
the zip with NB6's results included.

Before submitting, either delete the guard cell or set it to `False` and re-run —
`rubric.md` gives 5 points for "reproducible from ... Colab Run-all", and you do not want
a grader reading `CoreComplete` as a crash.

---

## 2. What `make verify` hard-fails on

From `scripts/verify.py` — these are checked as files, not as quality:

- `adapters/sft-mini/adapter_config.json`
- `adapters/dpo/adapter_config.json`
- `adapters/dpo/dpo_metrics.json` (must contain `end_reward_gap`)
- `data/pref/train.parquet`
- `data/eval/side_by_side.jsonl`
- `data/eval/judge_results.json`
- `submission/screenshots/` with **≥ 3 images** — NB1/NB3/NB4 save exactly 3 by themselves
- `submission/REFLECTION.md` with **fewer than 3** template placeholders left

> **Trap:** that last check is a placeholder count, not a content check. Filling in only the
> §1 Setup table can make `make verify` pass while §3 and §6 are still empty — and those two
> essays are worth 20 of 100 points. Green `verify` ≠ done.

---

## 3. The screenshot that carries the most weight

`03-dpo-reward-curves.png` — 22 of 100 points sit on NB3, and 10 of those require you to
read **`chosen` and `rejected` separately**, not just the gap.

NB3 has a self-check cell right after the plot that prints one of four verdicts
(`INTENDED` / `LIKELIHOOD DISPLACEMENT` / `FAILURE` / `AMBIGUOUS`). **Screenshot that text
too.** If it says likelihood displacement — gap grew because `rejected` fell faster than
`chosen` rose — that is a teachable result, not a failed run; say so in REFLECTION §3 and
you keep the points.

---

## 4. REFLECTION sections that are actually graded

| Section | Requirement | Points |
|---|---|---:|
| §3 Reward curves | ≥ 100 words, **both** curves interpreted | 5 (+10 via NB3 criterion) |
| §6 Personal reflection | ≥ 150 words, one decision walked through | part of 15 |
| §1, §2, §4 | tables filled with your real numbers | part of 15 |
| §5 β-sweep | bonus; if skipped, write a 3-sentence hypothesis instead | 0 |
| §7 Benchmark | needs NB6 — bonus only | 0 core |

---

## 5. Known rough edges in the lab repo

- **`5CD-AI/Vietnamese-alpaca-cleaned` does not exist on HuggingFace.** It was the shipped
  default for `SFT_DATASET`; `load_dataset` fails on it outright. Replaced everywhere with
  `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (52,002 rows, `train` split only), and
  `format_alpaca_to_chat` now reads that dataset's `instruction_vi / input_vi / output_vi`
  columns with a fallback to plain Alpaca names.
- **xformers has no GQA backward kernel on T4 — patched.** Mid-`trainer.train()`:
  `NotImplementedError: No operator found for memory_efficient_attention_backward`
  on a `(2, 512, 2, 8, 128)` BMGHK tensor — `fa2B`/`fa3B` need capability ≥ 8.0 (T4 is 7.5)
  and `cutlassB` does not support BMGHK at all, so every backend is rejected. The forward
  had already succeeded, so it only blows up once training starts. Unsloth's
  `models/_utils.py` catches `ModuleNotFoundError` on `import xformers` and sets
  `HAS_XFORMERS = False`, after which `select_attention_backend()` returns SDPA — which
  handles GQA on sm_75. The pip cell now uninstalls xformers when
  `torch.cuda.get_device_capability()[0] < 8`, before the first `import unsloth`.
  Ampere and newer keep it, so the BigGPU tier is unaffected.

- **The 4-bit base models have no chat template — patched.** `unsloth/Qwen2.5-3B-bnb-4bit`
  (and the 7B) ship `tokenizer_config.json` without `chat_template`; only the `-Instruct`
  repos have one. Every `apply_chat_template()` call therefore died with
  `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set`
  — first in NB1's Alpaca formatting, and it would have hit NB4's generation helper next.
  All six tokenizer load sites now install ChatML (Qwen2.5's native format) when the
  template is missing, set `eos_token` to `<|im_end|>` so generation stops on its own, and
  keep `pad_token` different from `eos_token` so the collator does not mask the eos the
  model has to learn to emit.

- **`submission/screenshots/README.md` contradicts `rubric.md`.** It lists 7 shots under a
  heading that says "Minimum (6 shots)" and includes NB5/NB6 images, but the rubric marks
  NB5/NB6 as optional and `verify.py` only needs 3 images. Follow the rubric.
- **`03_dpo_train.py` fragility — patched.** The `dpo_metrics` dict read `last_chosen` /
  `last_gap`, which existed only when the self-check cell's condition held
  (`len(logs) >= 5`); a short run raised `NameError` *after* training finished, losing
  `dpo_metrics.json`. Those names are now bound to `None` up-front. The reward-plot cell's
  no-`loss`-column fallback (`logs[logs.index]`) was broken the same way; fixed too.
- **`scripts/verify.py` crashed on this laptop — patched.** It prints `ⓘ` whenever NB5/NB6
  are absent, i.e. on every core-only run, and a cp1252 Windows console cannot encode it:
  `UnicodeEncodeError`, traceback, exit 1 before the checklist ever printed. Now forces
  UTF-8 on stdout.
- **`verify.py` treats its own "WARN" as a hard failure.** `check_dpo_metrics` appends the
  `end_reward_gap <= 0` message to the same `problems` list as real misses, so a negative
  reward gap fails `make verify` despite the text saying "Not a hard fail". Left as shipped
  — do not paper over it; if the gap comes out negative, that is a genuine result to write
  up in REFLECTION §3, and the 3 gatekeeper points are worth less than a faked pass.
- **Colab cell 5 pre-creates `adapters/sft-mini` and `adapters/dpo`**, which defeats the
  `assert SFT_PATH.exists()` "NB1 must run first" guards in NB2/NB3. Run the stages in
  order; the guard will not catch you.
- **`PREF_SLICE` differs between the two paths:** the Colab file uses 2000 pairs on T4,
  `notebooks/02_preference_data.py` uses 1000. The Colab number is the one that matches the
  REFLECTION template's example; expect NB3 to take proportionally longer.
