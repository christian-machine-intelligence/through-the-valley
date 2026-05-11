#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
set -a
[ -f .env ] && source .env
set +a
python3 src/generate_scenarios.py --pilot
