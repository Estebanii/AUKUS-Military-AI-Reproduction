#!/usr/bin/env python3
"""Record the run in results/run_info.json (run id, package commit, code fingerprint, data manifest sha256).

run_all.sh calls it before the first script; compare.py checks the results against it (see README section 5).
"""
import _common  # noqa: F401

from replication import data, run_info


def main() -> int:
    record = run_info.write(data.results_dir(), data.root())
    print(f'[record_run] run {record["run_id"]} (package commit {record["package_commit"]} from '
          f'{record["package_commit_source"]}, '
          f'code {record["code_sha256"][:12]}, data manifest {record["data_manifest_sha256"]})', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
