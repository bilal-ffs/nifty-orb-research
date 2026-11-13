# Reproducing and testing the research

## Data-free validation

Use Python 3.11-3.13 and install `requirements-research.txt`, then run:

```powershell
python scripts/run_tests.py
python scripts/check_publication.py
```

Each suite runs in its own Python process to avoid collisions between historical
script imports named `train`, `run`, `strategy` and `experiment`. No download or
historical backtest occurs. Historical suites require explicit selection even
though saved aggregate metrics are included. Do not count skipped tests as passes.

## Historical reproduction in a disposable working copy

Commands below write outputs and may overwrite included aggregate snapshots and
reports. Use a separate local working copy for historical experiments, and keep
the publication copy pristine. The commands require your own authorized source;
raw data, trained models and row-level exports remain ignored by Git.

Input schema: `date,open,high,low,close,volume`, with minute OPEN timestamps in
`DD-MM-YYYY HH:MM` format interpreted in Asia/Kolkata. See PROVENANCE.md for the
known cleaned source hash. Exact historical equality requires that exact source;
a schema-compatible replacement does not guarantee identical events or results.
The conservative cleaning tool from the old local project is not bundled because
it rewrites source files. The baseline audits its input without mutating it.

To reproduce only the baseline into a new output folder:

```powershell
python orb_pipeline.py --source "D:/private/NIFTY_50_minute.csv" --output results/baseline
```

To reproduce all dependent experiments, place the authorized cleaned input at
`NIFTY_50_minute.csv` in the root of the disposable working copy and run, in order:

```powershell
python orb_pipeline.py --source NIFTY_50_minute.csv --output .
python orb_diagnostics.py
python orb_logistic_v1/train.py
python orb_logistic_directional_v2/run.py
python orb_vervoort_atr_research/experiment.py
python orb_vervoort_atr_research/charts.py
python scripts/run_tests.py --artifacts
```

The baseline writes the ML feature/event and trade ledgers used by subsequent
scripts. Diagnostics write decomposition and conditioned tables. Model commands
write folds, OOF predictions, model files, metrics and reports beside their code.
Vervoort reads the root source, restricts numeric research to 2015-2021 and writes
its own ledgers, classifiers and reports. Optional historical tests deliberately
assert the original study's counts and hashes; they can fail for different data
or environments. Synthetic tests are the portable contract checks.

The pinned research dependencies match the recorded local environment. Exact
floating-point reconstruction on other operating systems is not guaranteed.
Never load an untrusted joblib/pickle file; generate private models locally.

## Legacy translation: separate source and assumptions

```powershell
python legacy_translation/orb_backtest.py --input "D:/private/ORIGINAL_NIFTY_50_minute.csv"
python legacy_translation/analyze_shorts.py
python legacy_translation/analyze_orb_width.py
```

The legacy program writes beside itself and uses the older pre-cleaning input.
Its optional original TradingView export is not included. Its before-cost,
full-history results are not the baseline or its OOS validation. Full-sample
quartile analyses are exploratory. See the legacy method/report for differences.

## Publication and Git

Review `.gitignore` before adding any new outputs. `check_publication.py` checks
tracked and unignored files when run inside this repository; ignored private files
remain local. It checks selected credential patterns, large files, private data
formats, machine-specific paths and relative links in current publication docs.
It does not certify data rights or scan complete Git history.

The environment used to prepare this folder blocked creation of `.git`, so
initialize it locally first with `git init -b main`. No remote, commit or push is created by the test or publication scripts. Initial
local publication is your choice:

```powershell
git init -b main
git status --short --untracked-files=all
git add .
git diff --cached --stat
git commit -m "Publish NIFTY ORB research and validation"
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

Review staged contents before committing. Market-data and supplied-rule provenance
remain documented uncertainties; the MIT license does not resolve those rights.
