#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${CODE_RAG_URL:-http://localhost:8081}"
JSON_OUTPUT=0

_die() { echo "error: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_urlencode() {
  local s="${1//'%'/%25}"
  s="${s//' '/%20}"
  s="${s//$'\n'/%0A}"
  echo -n "$s"
}

_get() {
  local url="${BASE_URL%/}$1" qp=() sep="?"
  shift
  while [[ $# -gt 0 ]]; do
    qp+=("${sep}$(_urlencode "$1")=$(_urlencode "$2")")
    sep="&"
    shift 2
  done
  local _ifs="$IFS"
  IFS=''; url+="${qp[*]}"; IFS="$_ifs"
  curl -sSf -- "$url" || {
    local ec=$?
    [[ $ec -eq 22 ]] && _die "HTTP error from server"
    [[ $ec -eq 26 ]] && _die "could not read response"
    exit $ec
  }
}

_post() {
  local url="${BASE_URL%/}$1" body="$2" qp=() sep="?"
  shift 2
  while [[ $# -gt 0 ]]; do
    qp+=("${sep}$(_urlencode "$1")=$(_urlencode "$2")")
    sep="&"
    shift 2
  done
  local _ifs="$IFS"
  IFS=''; url+="${qp[*]}"; IFS="$_ifs"
  curl -sSf -X POST -H "Content-Type: application/json" -d "$body" -- "$url" || {
    local ec=$?
    [[ $ec -eq 22 ]] && _die "HTTP error from server"
    [[ $ec -eq 26 ]] && _die "could not read response"
    exit $ec
  }
}

_emit() {
  # Reads JSON from stdin. If --json was set, pretty-prints and returns 0
  # (caller should `return` after calling). Else returns 1 so caller formats.
  if [[ "$JSON_OUTPUT" == "1" ]]; then
    jq '.'
    return 0
  fi
  return 1
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_search() {
  local query="" top_k=5 glibs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --top-k) top_k="$2"; shift 2 ;;
      --glob)  glibs+=("$2"); shift 2 ;;
      *)       query="$1"; shift ;;
    esac
  done
  [[ -z "$query" ]] && _die "search requires a query"
  local qparams=(q "$query" top_k "$top_k")
  for g in "${glibs[@]}"; do qparams+=(globs "$g"); done
  local out
  out="$(_get "/api/search" "${qparams[@]}")"
  echo "$out" | _emit && return
  echo "$out" | jq -r '.[] | "\(.score)\t\(.path)\n  \(.summary)"'
}

cmd_unit() {
  local path="${1:-}"
  [[ -z "$path" ]] && _die "unit requires a path"
  local out
  out="$(_get "/api/unit" path "$path")"
  echo "$out" | _emit && return
  echo "# $(echo "$out" | jq -r '.path')"
  local summary
  summary="$(echo "$out" | jq -r '.summary // empty')"
  [[ -n "$summary" ]] && echo "# $summary"
  echo
  echo "$out" | jq -r '.content'
}

cmd_fetch() {
  [[ $# -eq 0 ]] && _die "fetch requires at least one path"
  local paths_json
  paths_json="$(printf '%s\n' "$@" | jq -R -s -c 'split("\n") | map(select(length > 0)) | {paths: .}')"
  local out idx=0 count
  out="$(_post "/api/units/fetch" "$paths_json")"
  echo "$out" | _emit && return
  count="$(echo "$out" | jq -r 'length')"
  while [[ $idx -lt $count ]]; do
    [[ $idx -gt 0 ]] && printf '\n---\n\n'
    echo "# $(echo "$out" | jq -r ".[$idx].path")"
    local s
    s="$(echo "$out" | jq -r ".[$idx].summary // empty")"
    [[ -n "$s" ]] && echo "# $s"
    echo
    echo "$out" | jq -r ".[$idx].content"
    idx=$((idx + 1))
  done
}

cmd_units() {
  local limit=100 glibs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --limit) limit="$2"; shift 2 ;;
      --glob)  glibs+=("$2"); shift 2 ;;
      *) shift ;;
    esac
  done
  local qparams=(limit "$limit")
  for g in "${glibs[@]}"; do qparams+=(globs "$g"); done
  local out
  out="$(_get "/api/units" "${qparams[@]}")"
  echo "$out" | _emit && return
  echo "$out" | jq -r '.[] | "\(.path)\t\(.summary)"'
}

cmd_files() {
  local glibs=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --glob) glibs+=("$2"); shift 2 ;;
      *) shift ;;
    esac
  done
  local qparams=()
  for g in "${glibs[@]}"; do qparams+=(globs "$g"); done
  local out
  out="$(_get "/api/files" "${qparams[@]}")"
  echo "$out" | _emit && return
  echo "$out" | jq -r '.[] | "\(.repo)/\(.path)\t\(.indexed_at)"'
}

cmd_repos() {
  local out
  out="$(_get "/api/repos")"
  echo "$out" | _emit && return
  echo "$out" | jq -r '.[] | "\(.name)\t\(.root)\t\(.added_at)"'
}

cmd_status() {
  local out
  out="$(_get "/api/status")"
  echo "$out" | _emit && return
  echo "total_units: $(echo "$out" | jq -r '.total_units')"
  echo "embed_count: $(echo "$out" | jq -r '.embed_count')"
  echo "$out" | jq -r '.repos[] | "  \(.repo)\tfiles=\(.file_count)\tunits=\(.unit_count)\tlast_indexed=\(.last_indexed_at // "never")"'
}

cmd_browse() {
  local out
  out="$(_get "/api/browse" path "${1:-}")"
  echo "$out" | _emit && return
  echo "$out" \
    | jq -r '.[] | [.type, .name, .path, (.unit_type // empty), (.summary // empty)] | map(select(. != "")) | join("\t")'
}

cmd_index() {
  local reindex="false" paths=() wait_flag=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --reindex) reindex="true"; shift ;;
      --wait)    wait_flag=1; shift ;;
      *) paths+=("$1"); shift ;;
    esac
  done
  [[ ${#paths[@]} -eq 0 ]] && _die "index requires at least one path"
  local body
  body="$(printf '%s\n' "${paths[@]}" | jq -R -s -c --argjson r "$reindex" 'split("\n") | map(select(length > 0)) | {paths: ., reindex: $r}')"
  local status
  status="$(_post "/api/index" "$body")"
  if [[ "$wait_flag" -eq 0 ]]; then
    _print_job_status "$status"
    return
  fi
  while [[ "$(echo "$status" | jq -r '.running')" == "true" ]]; do
    sleep 2
    printf '.' >&2
    status="$(_get "/api/index/status")"
  done
  printf '\n' >&2
  _print_job_status "$status"
  local last_result
  last_result="$(echo "$status" | jq -r '.last_result // empty')"
  [[ "$last_result" != "ok" ]] && exit 1
}

cmd_index_status() {
  _print_job_status "$(_get "/api/index/status")"
}

cmd_index_cancel() {
  _print_job_status "$(_post "/api/index/cancel" "{}")"
}

cmd_clear_repo() {
  local repo="${1:-}"
  [[ -z "$repo" ]] && _die "clear-repo requires a repo name"
  local out
  out="$(_post "/api/clear_repo" "{}" repo "$repo")"
  echo "$out" | _emit && return
  if [[ "$(echo "$out" | jq -r '.ok // false')" == "true" ]]; then
    echo "cleared: $(echo "$out" | jq -r '.repo')"
  else
    echo "$out" | jq '.'
  fi
}

cmd_staleness() {
  local out
  out="$(_get "/api/repos/staleness")"
  echo "$out" | _emit && return
  echo "$out" | jq -r '.[] | "\(.repo)\tindexed=\(.last_indexed_at // "never")\thead=\(.last_commit_at // "unknown")\tstale=\(.stale)\t\(.reason)"'
}

cmd_ls() {
  local path="${1:-}"
  local out
  if [[ -n "$path" ]]; then
    out="$(_get "/api/ls" path "$path")"
  else
    out="$(_get "/api/ls")"
  fi
  echo "$out" | _emit && return
  local marker=""
  [[ "$(echo "$out" | jq -r '.is_git')" == "true" ]] && marker=" *"
  echo "$(echo "$out" | jq -r '.path')$marker"
  echo "$out" | jq -r '.dirs[].name'
}

_print_job_status() {
  local data="$1"
  echo "$data" | _emit && return
  echo "running: $(echo "$data" | jq -r '.running')"
  local lr lf
  lr="$(echo "$data" | jq -r '.last_result // empty')"
  lf="$(echo "$data" | jq -r '.last_finished_at // empty')"
  [[ -n "$lr" ]] && echo "last_result: $lr"
  [[ -n "$lf" ]] && echo "last_finished_at: $lf"
}

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

usage() {
  cat <<'EOF'
Usage: code-rag.sh [--base-url URL] [--json] COMMAND [ARGS...]

Commands:
  search  QUERY                Search indexed code
          --top-k N            (default: 5)
          --glob GLOB          (repeatable)

  unit    PATH                 Get a single unit by qualified path
  fetch   PATH...              Fetch multiple units by qualified path

  units                        List semantic units
          --limit N            (default: 100)
          --glob GLOB          (repeatable)

  files                        List indexed files
          --glob GLOB          (repeatable)

  repos                        List indexed repositories
  status                       Index health check
  staleness                    Per-repo index freshness vs git HEAD
  browse  [PATH]               Browse the index tree
  ls      [PATH]               List filesystem directories (server-side)

  index   PATH...              Start an indexing job
          --reindex
          --wait               Block until job completes; exit non-zero on failure

  index-status                 Poll indexing job state
  index-cancel                 Cancel running indexing job

  clear-repo REPO              Remove indexed data for a repository

Options:
  --base-url URL               Base URL (env: CODE_RAG_URL, default: http://localhost:8081)
  --json                       Emit raw JSON instead of formatted output
  -h, --help                   Show this help
EOF
}

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

[[ $# -eq 0 ]] && { usage; exit 1; }

# Parse top-level flags (--base-url, --json, --help) in any order before the subcommand
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h)   usage; exit 0 ;;
    --base-url)  BASE_URL="$2"; shift 2 ;;
    --json)      JSON_OUTPUT=1; shift ;;
    *) break ;;
  esac
done

case "${1:-}" in
  search)       shift; cmd_search "$@" ;;
  unit)         shift; cmd_unit "$@" ;;
  fetch)        shift; cmd_fetch "$@" ;;
  units)        shift; cmd_units "$@" ;;
  files)        shift; cmd_files "$@" ;;
  repos)        shift; cmd_repos "$@" ;;
  status)       shift; cmd_status "$@" ;;
  browse)       shift; cmd_browse "$@" ;;
  index)        shift; cmd_index "$@" ;;
  index-status) shift; cmd_index_status "$@" ;;
  index-cancel) shift; cmd_index_cancel "$@" ;;
  clear-repo)   shift; cmd_clear_repo "$@" ;;
  staleness)    shift; cmd_staleness "$@" ;;
  ls)           shift; cmd_ls "$@" ;;
  *)            _die "unknown command: ${1:-}" ;;
esac
