#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/raw
hf download HuggingFaceH4/no_robots --repo-type dataset --local-dir data/raw/no_robots
printf '\nDataset downloaded to data/raw/no_robots\n'
printf 'Review the dataset license and card before use.\n'
