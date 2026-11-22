#!/usr/bin/env bash
# Archive HEAD, version the private Kaggle source dataset AND the Telco
# data dataset (downloaded from blastchar/telco-customer-churn and
# sha256-verified), push the kernel, poll until it finishes, and download
# outputs to artifacts/kaggle/<ts>/. Kernel and dataset ids come from
# kaggle/kernel-metadata.json so this script is portable across repos.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

KAGGLE_BIN="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
SMOKE=0
if [ "${1:-}" = "--smoke" ]; then
  SMOKE=1
fi
META="${KAGGLE_META:-}"
if [ -z "$META" ]; then
  if [ "$SMOKE" -eq 1 ]; then
    META="kaggle_smoke/kernel-metadata.json"
  else
    META="kaggle/kernel-metadata.json"
  fi
fi
KERNEL_ID="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["id"])' "$META")"
SOURCES="$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["dataset_sources"]))' "$META")"
SRC_DATASET="$(echo "$SOURCES" | awk '{print $1}')"
DATA_DATASET="$(echo "$SOURCES" | awk '{print $2}')"
SRC_TITLE="$(basename "$SRC_DATASET")"
TS="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="artifacts/kaggle/${TS}"
EXPECTED_SHA="88be4b93fbe0cc83421af1c503794c97c342eca914c1576db7c276e61d61358a"

SHA="$(git rev-parse HEAD)"
if [ -n "$(git status --porcelain)" ]; then
  echo "warning: working tree is dirty; the dataset reflects HEAD ($SHA) only" >&2
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# --- Data dataset: download blastchar Telco churn, verify sha256, version. ---
mkdir -p "$TMP/dl" "$TMP/data-ds"
"$KAGGLE_BIN" datasets download -d blastchar/telco-customer-churn -p "$TMP/dl" --unzip -q
CSV_COUNT="$(find "$TMP/dl" -name '*.csv' | wc -l | tr -d ' ')"
if [ "$CSV_COUNT" -ne 1 ]; then
  echo "error: expected exactly 1 CSV in blastchar/telco-customer-churn, found $CSV_COUNT" >&2
  exit 1
fi
CSV_PATH="$(find "$TMP/dl" -name '*.csv' | head -1)"
ACTUAL_SHA="$(shasum -a 256 "$CSV_PATH" | awk '{print $1}')"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "error: downloaded CSV sha256 $ACTUAL_SHA != committed $EXPECTED_SHA" >&2
  exit 1
fi
cp "$CSV_PATH" "$TMP/data-ds/WA_Fn-UseC_-Telco-Customer-Churn.csv"
cat > "$TMP/data-ds/dataset-metadata.json" <<EOF
{"title": "telco-churn-data", "id": "${DATA_DATASET}", "licenses": [{"name": "other"}]}
EOF
if "$KAGGLE_BIN" datasets status "$DATA_DATASET" >/dev/null 2>&1; then
  echo "versioning data dataset $DATA_DATASET"
  "$KAGGLE_BIN" datasets version -p "$TMP/data-ds" -m "telco csv ${ACTUAL_SHA:0:12}" -r tar -q
else
  echo "creating data dataset $DATA_DATASET"
  "$KAGGLE_BIN" datasets create -p "$TMP/data-ds" -r tar -q
fi

# --- Source dataset: git archive of HEAD. ---
mkdir -p "$TMP/ds"
git archive HEAD | tar -x -C "$TMP/ds"

cat > "$TMP/ds/dataset-metadata.json" <<EOF
{"title": "${SRC_TITLE}", "id": "${SRC_DATASET}", "licenses": [{"name": "other"}]}
EOF
cat > "$TMP/ds/run_info.json" <<EOF
{"git_sha": "${SHA}", "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF

if "$KAGGLE_BIN" datasets status "$SRC_DATASET" >/dev/null 2>&1; then
  echo "versioning existing dataset $SRC_DATASET"
  "$KAGGLE_BIN" datasets version -p "$TMP/ds" -m "src ${SHA}" -r tar -q
else
  echo "creating dataset $SRC_DATASET"
  "$KAGGLE_BIN" datasets create -p "$TMP/ds" -r tar -q
fi

# Wait for both dataset versions to finish processing.
for DS in "$SRC_DATASET" "$DATA_DATASET"; do
  READY=0
  for _ in $(seq 1 60); do
    STATUS="$("$KAGGLE_BIN" datasets status "$DS" 2>/dev/null || echo unknown)"
    echo "dataset $DS status: $STATUS"
    case "$STATUS" in
      *ready*|*complete*) READY=1; break ;;
    esac
    sleep 10
  done
  if [ "$READY" -ne 1 ]; then
    echo "error: dataset $DS did not become ready" >&2
    exit 1
  fi
done

# "ready" precedes mount propagation by minutes; kernels pushed too early
# mount the previous dataset version and fail the git_sha guard.
sleep 300

echo "pushing kernel $KERNEL_ID"
PUSH_DIR="$(dirname "$META")"
"$KAGGLE_BIN" kernels push -p "$PUSH_DIR"

# Poll kernel status until a terminal state; bounded by the loop count.
TERMINAL=""
for _ in $(seq 1 720); do
  STATUS="$("$KAGGLE_BIN" kernels status "$KERNEL_ID" 2>&1 || echo "status-poll-error")"
  echo "kernel status: $STATUS"
  case "$STATUS" in
    *COMPLETE*|*ERROR*|*CANCEL*) TERMINAL="$STATUS"; break ;;
  esac
  sleep 60
done

mkdir -p "$OUT_DIR"
echo "downloading kernel output to $OUT_DIR"
"$KAGGLE_BIN" kernels output "$KERNEL_ID" -p "$OUT_DIR"

# Guard against the dataset-version race: the run must reflect HEAD.
MANIFEST="$OUT_DIR/benchmark/manifest.json"
if [ -f "$MANIFEST" ]; then
  RUN_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("git_sha",""))' "$MANIFEST")"
  if [ "$RUN_SHA" != "$SHA" ]; then
    echo "error: run manifest git_sha=$RUN_SHA does not match HEAD $SHA" >&2
    exit 1
  fi
fi

case "$TERMINAL" in
  *COMPLETE*) ;;
  *)
    echo "error: kernel did not complete (status: ${TERMINAL:-poll-timeout})" >&2
    exit 1
    ;;
esac

echo "done: kernel=$KERNEL_ID sha=$SHA out=$OUT_DIR"
