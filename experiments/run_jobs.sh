#!/usr/bin/env bash
# Run a job list with PAR jobs at a time.
# usage: bash experiments/run_jobs.sh mart|baselines <jobs file> [PAR]
# Each non-comment line is "<market> <seed> <runner arguments...>". MART jobs get the development test years
# (2006-2025, Nasdaq 2014-2025) unless the line sets --years, and always save test predictions, which several
# analysis scripts read. Logs go to logs/.
#
# Each runner builds a data cache per market and data setting on first use. Two processes building the same cache
# at once can collide, so build the caches first (see README) or start with PAR=1.
set -eo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}

years_of() { if [ "$1" = nasdaq ]; then seq -s ' ' 2014 2025; else seq -s ' ' 2006 2025; fi; }
runner_of() {
  case $1 in
    mart) echo code/paper_final2/run_top20_amount_ablation.py ;;
    baselines) echo code/paper_final2/run_figure4_baseline_comparison.py ;;
    *) echo "unknown job kind: $1" >&2; exit 1 ;;
  esac
}

if [ "${1:-}" = job ]; then
  kind=$2; market=$3; seed=$4; shift 4
  extra=()
  if [ "$kind" = mart ]; then
    case " $* " in *" --years "*) ;; *) extra=(--years $(years_of "$market")) ;; esac
    extra+=(--save_predictions)
  fi
  mkdir -p logs
  log="logs/${kind}_${market}_seed${seed}_$(echo "$*" | tr -c 'A-Za-z0-9_.-' '_' | cut -c1-80).log"
  exec "$PY" "$(runner_of "$kind")" --market "$market" --seed "$seed" --seed_dir "${extra[@]}" "$@" > "$log" 2>&1
fi

kind=${1:?usage: run_jobs.sh mart|baselines <jobs file> [PAR]}
jobs=${2:?usage: run_jobs.sh mart|baselines <jobs file> [PAR]}
runner_of "$kind" > /dev/null
sed 's/\r$//' "$jobs" | grep -v -e '^[[:space:]]*#' -e '^[[:space:]]*$' | xargs -P "${3:-1}" -L 1 bash "$0" job "$kind"
