#!/usr/bin/env bash
set -euo pipefail

read_secret() {
  local name="$1"
  local path="/run/secrets/$name"
  if [[ -f "$path" ]]; then
    tr -d '\r\n' < "$path"
  fi
}

if [[ -z "${MODAL_TOKEN_ID:-}" ]]; then
  modal_token_id="$(read_secret modal_token_id)"
  if [[ -n "$modal_token_id" ]]; then
    export MODAL_TOKEN_ID="$modal_token_id"
  fi
fi

if [[ -z "${MODAL_TOKEN_SECRET:-}" ]]; then
  modal_token_secret="$(read_secret modal_token_secret)"
  if [[ -n "$modal_token_secret" ]]; then
    export MODAL_TOKEN_SECRET="$modal_token_secret"
  fi
fi

if [[ -z "${HUGGINGFACE_TOKEN:-}" ]]; then
  huggingface_token="$(read_secret huggingface_token)"
  if [[ -n "$huggingface_token" ]]; then
    export HUGGINGFACE_TOKEN="$huggingface_token"
  fi
fi

if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_TOKEN:-}" ]]; then
  export HF_TOKEN="$HUGGINGFACE_TOKEN"
fi

if [[ -z "${MODAL_HF_SECRET_NAME:-}" ]]; then
  modal_hf_secret_name="$(read_secret modal_hf_secret_name)"
  if [[ -n "$modal_hf_secret_name" ]]; then
    export MODAL_HF_SECRET_NAME="$modal_hf_secret_name"
  fi
fi

exec "$@"
