"""Run isolated unittest suites; historical artifacts require explicit opt-in."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', action='store_true',
                        help='Also reconcile private historical data and trained models')
    args = parser.parse_args()
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['ORB_ARTIFACT_TESTS'] = '1' if args.artifacts else '0'
    env['PYTHONPATH'] = str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    if args.artifacts:
        required = ['NIFTY_50_minute.csv', 'orb_ml_is.csv', 'orb_is_trades.csv',
                    'orb_cost_decomposition.csv', 'orb_feature_conditioned_is.csv',
                    'orb_exit_distribution_is.csv', 'orb_logistic_v1/fold_1.joblib',
                    'orb_logistic_directional_v2/long_fold_1.joblib',
                    'orb_vervoort_atr_research/atr_0.5_fold_1.joblib']
        missing = [p for p in required if not (ROOT / p).is_file()]
        if missing:
            parser.error('Historical artifacts missing; see docs/REPRODUCING.md: '
                         + ', '.join(missing))
    suites = [('tests', 'test_*.py'),
              ('orb_vervoort_atr_research', 'test_strategy.py')]
    if args.artifacts:
        suites = [('tests', 'test_*.py'), ('.', 'orb_diagnostic_tests.py'),
                  ('orb_logistic_v1', 'test_*.py'),
                  ('orb_logistic_directional_v2', 'test_*.py'),
                  ('orb_vervoort_atr_research', 'test_*.py')]
    else:
        print('41 historical-artifact tests not selected; use --artifacts after '
              'private reconstruction.', flush=True)
    for directory, pattern in suites:
        print(f'\nRunning {directory}/{pattern}', flush=True)
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover',
                        '-s', directory, '-p', pattern, '-v'],
                       cwd=ROOT, env=env, check=True)
    subprocess.run([sys.executable, '-B', 'check_orb_backtest.py'],
                   cwd=ROOT / 'legacy_translation', env=env, check=True)
    print('\nAll selected suites passed. Historical suites are opt-in.', flush=True)


if __name__ == '__main__':
    main()
