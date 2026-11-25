#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAPERM="$(cd "$HERE/../../.." && pwd)"
REPOS="$(cd "$SAPERM/.." && pwd)"
export PYTHONPATH="$HERE:$SAPERM/benchmark_methods:$REPOS/RFPerm/Python:$REPOS/CFPerm/Python/src:${PYTHONPATH:-}"
echo "PYTHONPATH=$PYTHONPATH"
