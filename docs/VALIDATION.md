# Publication validation

Publication preparation: 2026-11-13. Original datasets and source projects were
retained unchanged. The signal, execution and cost rules were not modified.

The publication copy adds documentation, MIT licensing, ignore rules, CI,
publication checks, synthetic ML tests and explicit opt-in historical suites.

Local environment: Python 3.13.9, NumPy 2.3.5, pandas 2.3.3,
scikit-learn 1.9.1 and joblib 1.6.0.

- Portable suites: 28 synthetic unittest tests passed; 41 private-artifact
  tests not selected by the portable runner. Legacy execution assertions passed separately.
- Historical verification: all 69 unittest tests passed with no skips in a
  disposable private working copy, including the 28 synthetic checks and 41
  artifact checks. Legacy execution assertions passed separately. Existing
  local CSV/model artifacts were supplied only to the private copy and removed
  with that copy after verification. Publication does not include them.
- Artifact checks reproduced stored probabilities, fold scalers, training-label
  maturity, counts, fee/P&L reconciliation and source hashes. This verifies
  internal consistency, not a new full backtest or independent replication.
- Original-source hashes were rechecked: no original source/report files changed.
- All publication Python files parse; all current Markdown links resolve locally.
- The bounded publication-content check passed on the curated folder.
- Git initialization was denied by session filesystem permissions. The completed
  folder has no Git history or remote; run `git init -b main` locally before
  reviewing, staging and pushing it.
- CI is configured for Python 3.11, 3.12 and 3.13 on Linux; those remote jobs
  have not run during local preparation.
- Historical reports/metrics are preserved evidence, not newly optimized results.

The portable suites validate algorithm contracts on fixtures. They do not prove
source licensing, full calendar completeness, actual futures executability,
independent replication or an untouched holdout.
