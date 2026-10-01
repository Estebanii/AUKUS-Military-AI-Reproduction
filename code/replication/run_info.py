"""The record of a run, ``results/run_info.json``, and the check ``compare.py`` makes against it.

``run_all.sh`` writes the record before the first script (``scripts/record_run.py``): a run id (UTC start time,
package commit, data manifest), the commit of the package and where it was read (``git``: HEAD of a git checkout of the
package; ``VERSION``: the commit id that ``git archive`` and GitHub's "Download ZIP" write into the file VERSION; None
if neither is available), a fingerprint of the code that determines the results (sha256 over code/, scripts/,
run_all.sh and environment/requirements.lock) and the sha256 of the data bundle's MANIFEST.sha256. ``compare.py``
prints the record and reports a problem when it is missing, or when the commit, the code or the data bundle differs
from the run's. A tree without .git is not a problem in itself.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import subprocess
from pathlib import Path

from . import data

NAME = 'run_info.json'
CODE_GLOBS = ('code/**/*.py', 'code/**/*.R', 'scripts/*.py', 'run_all.sh', 'environment/requirements.lock')
COMMIT_ID = re.compile(r'^[0-9a-f]{40}$')


def git_head(package: Path = data.PACKAGE) -> str | None:
    """HEAD of the package's own git checkout (None if the package is not the top of a git checkout)."""
    try:
        top = subprocess.run(['git', '-C', str(package), 'rev-parse', '--show-toplevel'], capture_output=True,
                             text=True, timeout=10)
        if top.returncode != 0 or Path(top.stdout.strip()).resolve() != package.resolve():
            return None
        head = subprocess.run(['git', '-C', str(package), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                              timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return head.stdout.strip() if head.returncode == 0 else None


def version_file(package: Path = data.PACKAGE) -> str | None:
    """The commit id in VERSION (written by ``git archive`` and GitHub's "Download ZIP" through export-subst; a git
    checkout holds the unexpanded placeholder, and None is returned)."""
    try:
        text = (package / 'VERSION').read_text(encoding='utf-8').strip()
    except OSError:
        return None
    return text if COMMIT_ID.match(text) else None


def package_commit(package: Path = data.PACKAGE) -> tuple:
    """(commit id or None, source): git HEAD of a git checkout of the package, else the commit id in VERSION."""
    head = git_head(package)
    if head:
        return head, 'git'
    version = version_file(package)
    if version:
        return version, 'VERSION'
    return None, None


def code_sha256(package: Path = data.PACKAGE) -> str:
    digest = hashlib.sha256()
    files = sorted({p for pattern in CODE_GLOBS for p in package.glob(pattern)
                    if p.is_file() and '__pycache__' not in p.parts})
    for path in files:
        digest.update(path.relative_to(package).as_posix().encode('utf-8'))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def manifest_sha256(bundle: Path) -> str | None:
    path = Path(bundle) / 'MANIFEST.sha256'
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def current(bundle: Path) -> dict:
    commit, source = package_commit()
    return {'package_commit': commit, 'package_commit_source': source, 'code_sha256': code_sha256(),
            'data_manifest_sha256': manifest_sha256(bundle)}


def write(results: Path, bundle: Path) -> dict:
    now = datetime.datetime.now(datetime.timezone.utc)
    state = current(bundle)
    run_id = '-'.join([now.strftime('%Y%m%dT%H%M%SZ'), (state['package_commit'] or 'no-commit')[:12],
                       (state['data_manifest_sha256'] or 'no-manifest')[:12]])
    record = {'run_id': run_id, 'started_utc': now.isoformat(timespec='seconds'), **state}
    Path(results).mkdir(parents=True, exist_ok=True)
    (Path(results) / NAME).write_text(json.dumps(record, indent=1) + '\n', encoding='utf-8')
    return record


def check(results: Path, bundle: Path) -> tuple:
    """(record or None, problems): the run record of ``results`` against the current checkout, code and bundle."""
    path = Path(results) / NAME
    if not path.is_file():
        return None, [f'{path} is missing: these results were not produced by ./run_all.sh in this checkout '
                      '(or were copied from elsewhere); rerun ./run_all.sh']
    record = json.loads(path.read_text(encoding='utf-8'))
    now, problems = current(bundle), []
    if record.get('package_commit') != now['package_commit']:
        problems.append(f'the results were produced at package commit {record.get("package_commit")}; this copy of the '
                        f'package is at {now["package_commit"]}')
    if record.get('code_sha256') != now['code_sha256']:
        problems.append('the code (code/, scripts/, run_all.sh, environment/requirements.lock) differs from the code '
                        'that produced the results')
    if record.get('data_manifest_sha256') != now['data_manifest_sha256']:
        problems.append('the data bundle manifest differs from the one the results were produced with')
    return record, problems
