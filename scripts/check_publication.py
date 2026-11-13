"""Check public files without executing research or reading ignored private data.

When Git exists, only tracked or unignored files are inspected. Otherwise the
curated folder is inspected, excluding environments, caches and private outputs.
This bounded check is not a complete secret-history or licensing audit.
"""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AGGREGATES = {
    'orb_vervoort_atr_research/atr_multiplier_comparison.csv',
    'orb_vervoort_atr_research/long_short_breakdown.csv',
    'orb_logistic_directional_v2/directional_diagnostics.csv',
    'orb_logistic_directional_v2/v1_v2_comparison.csv',
}
PATTERNS = [re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
            re.compile(r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}'),
            re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')]


def public_files():
    if (ROOT / '.git').exists():
        result = subprocess.run(['git', '-c', f'safe.directory={ROOT.as_posix()}',
                                 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
                                cwd=ROOT, capture_output=True, check=True)
        return [ROOT / name.decode('utf-8') for name in result.stdout.split(b'\0') if name]
    return [p for p in ROOT.rglob('*') if p.is_file() and
            not any(part.startswith('.venv') or part in
                    {'.git', '__pycache__', 'results', '.pytest_cache'} for part in p.relative_to(ROOT).parts)]


def main():
    errors = []
    files = public_files()
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        if p.name.startswith('.env') and p.name != '.env.example':
            errors.append(f'{rel}: environment file')
        if p.suffix.lower() in {'.csv', '.gz', '.parquet', '.feather', '.joblib', '.pkl', '.pickle', '.pem', '.key'} and rel not in AGGREGATES:
            errors.append(f'{rel}: private/unreviewed data, model or key format')
        if p.stat().st_size > 5_000_000:
            errors.append(f'{rel}: large publication artifact')
        if p.suffix in {'.py', '.md', '.json', '.yml', '.txt', '.toml'} or p.name == '.gitignore':
            text = p.read_text(encoding='utf-8-sig')
            if any(pattern.search(text) for pattern in PATTERNS):
                errors.append(f'{rel}: credential-like pattern (value suppressed)')
            if re.search(r'[A-Z]:[\/](?:Users|ORB_NIFTY_50)', text):
                errors.append(f'{rel}: machine-specific source path')
    # Check links in current publication docs. Archived reports describe private
    # artifacts intentionally absent from this public checkout.
    for p in [ROOT / 'README.md', *(ROOT / 'docs').glob('*.md')]:
        text = p.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' in target or target.startswith('#'):
                continue
            if not (p.parent / target.split('#')[0]).exists():
                errors.append(f'{p.name}: missing relative link {target}')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(f'PASS: {len(files)} public files checked; no private data/models, '
          'obvious credential patterns, oversized files or broken publication links.')


if __name__ == '__main__':
    main()
