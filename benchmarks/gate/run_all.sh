#!/bin/bash
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-../../.venv/bin/python}"
echo "== ours $(date +%T)"
$PY run_ours.py --sizes 1e4 1e5 1e6 1e7 3e7 2>&1 | grep -v RuntimeWarning
echo "== competitors $(date +%T)"
$PY run_scaling.py --sizes 1e5 1e6 1e7 3e7 --libs rdrobust R diff_diff 2>&1 | grep -v RuntimeWarning
echo "== gate 4 $(date +%T)"
$PY gate4_run.py 2>&1 | grep -v RuntimeWarning
echo "== done $(date +%T)"
