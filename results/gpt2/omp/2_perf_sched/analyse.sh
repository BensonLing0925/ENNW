#!/usr/bin/bash
set -euo pipefail

if [ -z "${1:-}" ]; then
    printf "No input file specified!\n"
    printf "Usage: %s <timehist.txt> [process_name]\n" "$0"
    exit 1
fi

INPUT_FILE="$1"
TARGET="${2:-gpt2_io_test}"

FILTER="NR>3 && NF>=6 && \$3 ~ /^${TARGET}/"

TOTAL_THREAD=$(awk -v tgt="$TARGET" 'NR>3 && $3 ~ ("^" tgt) {print $3}' "$INPUT_FILE" \
               | sed 's/.*\[//; s/\].*//' | sort -u | wc -l)
echo "Threads observed      : $TOTAL_THREAD"

read -r WALL RUNTIME <<< $(awk -v tgt="$TARGET" '
    NR>3 && NF>=6 && $3 ~ ("^" tgt) {
        t = $1 + 0
        if (n == 0) { min = t; max = t }
        if (t > max) max = t
        if (t < min) min = t
        run += $(NF)
        n++
    }
    END {
        if (n > 0) printf "%.6f %.6f", max - min, run / 1000
        else       printf "0 0"
    }' "$INPUT_FILE")

echo "Wall clock            : ${WALL} s"
echo "Total CPU time        : ${RUNTIME} s"

AVG_CORES=$(awk "BEGIN {printf \"%.2f\", ($WALL > 0) ? $RUNTIME / $WALL : 0}")
echo "Avg cores busy        : $AVG_CORES"

SORTED=$(awk -v tgt="$TARGET" 'NR>3 && NF>=6 && $3 ~ ("^" tgt) && $3 ~ /\// {print $(NF)}' \
         "$INPUT_FILE" | sort -n)
TOTAL_LINES=$(printf '%s\n' "$SORTED" | grep -c .)

if [ "$TOTAL_LINES" -eq 0 ]; then
    echo "No worker-thread slices found."
    exit 0
fi

percentile() {
    local pct=$1
    local line
    line=$(awk "BEGIN {n = int($TOTAL_LINES * $pct / 100 + 0.5); print (n < 1) ? 1 : n}")
    printf '%s\n' "$SORTED" | sed -n "${line}p"
}

echo "---------- worker slice distribution ----------"
echo "slices                : $TOTAL_LINES"
echo "p25                   : $(percentile 25) ms"
echo "p50 (median)          : $(percentile 50) ms"
echo "p99                   : $(percentile 99) ms"
echo "max                   : $(printf '%s\n' "$SORTED" | tail -n1) ms"

SHORT=$(printf '%s\n' "$SORTED" | awk '$1 < 0.1' | grep -c . || true)
awk "BEGIN {printf \"slices < 0.1 ms       : %d (%.1f%%)\n\", $SHORT, 100 * $SHORT / $TOTAL_LINES}"
