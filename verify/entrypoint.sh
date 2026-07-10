#!/usr/bin/env bash
set -euo pipefail

exec python3 /opt/verify/verify_npu_models.py "$@"
