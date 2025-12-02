#!/usr/bin/env bash
# reproduce.sh
# Minimal reproduce pipeline: generate -> smooth -> analyze
set -euo pipefail

OUTDIR=outputs/reproduce_$(date +%Y%m%d_%H%M%S)
mkdir -p "$OUTDIR"

echo "Generating samples (20, seed 42) into $OUTDIR"
python generate.py --num_samples 20 --seed 42 --outdir "$OUTDIR"

echo "Smoothing outputs (window 5)"
python batch_smooth_and_render.py --input_dir "$OUTDIR" --window_size 5 --outdir "$OUTDIR/smoothed"

echo "Analyzing outputs"
python analyze_training.py --pred_dir "$OUTDIR" --out "$OUTDIR/metrics.json"

echo "Done. Results in $OUTDIR"
