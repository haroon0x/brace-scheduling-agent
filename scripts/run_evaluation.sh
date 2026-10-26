#!/usr/bin/env bash
set -euo pipefail

echo "== Unit tests =="
uv run pytest -q

echo
echo "== Scenario list =="
uv run scheduler scenarios

echo
echo "== Deterministic local eval =="
uv run scheduler eval

if [[ "${RUN_LLM_CHECKS:-0}" == "1" ]]; then
  echo
  echo "== Optional LLM eval =="
  uv run scheduler eval --llm
fi

echo
echo "Evaluation smoke complete."
