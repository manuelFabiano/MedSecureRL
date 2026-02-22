#!/bin/bash
# Run full experimental evaluation for MedSecure
# Usage: ./scripts/run_experiments.sh

set -e

DATASET=${1:-"pathmnist"}
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"

echo "========================================"
echo "MedSecure - Experimental Evaluation"
echo "========================================"

# Find victim model and agent checkpoints
VICTIM_PATH=$(ls -t $CHECKPOINT_DIR/*_${DATASET}_best.pth 2>/dev/null | head -1)
AGENT_PATH="$RESULTS_DIR/agent/agent_best.pth"

if [ -z "$VICTIM_PATH" ]; then
    echo "Error: No victim model found for $DATASET"
    echo "Run ./scripts/train_all.sh first"
    exit 1
fi

echo "Victim model: $VICTIM_PATH"
echo "Agent: $AGENT_PATH"
echo "========================================"

# Step 1: Full evaluation
echo ""
echo "[1/3] Running comprehensive evaluation..."
echo "----------------------------------------"
python experiments/evaluate.py \
    --victim-model $VICTIM_PATH \
    --agent $AGENT_PATH \
    --dataset $DATASET \
    --n-samples 1000 \
    --epsilon 0.03 \
    --output-dir $RESULTS_DIR/evaluation

# Step 2: Epsilon sweep
echo ""
echo "[2/3] Running epsilon sweep..."
echo "----------------------------------------"
python experiments/run_baselines.py \
    --model-path $VICTIM_PATH \
    --dataset $DATASET \
    --attack pgd \
    --epsilon-sweep \
    --n-samples 500 \
    --output-dir $RESULTS_DIR/baselines

# Step 3: Ablation studies (optional, takes time)
if [ "$2" == "--ablation" ]; then
    echo ""
    echo "[3/3] Running ablation studies..."
    echo "----------------------------------------"
    python experiments/ablation.py \
        --victim-model $VICTIM_PATH \
        --dataset $DATASET \
        --episodes 20000 \
        --eval-samples 500 \
        --output-dir $RESULTS_DIR/ablation
else
    echo ""
    echo "[3/3] Skipping ablation studies (use --ablation flag to enable)"
fi

echo ""
echo "========================================"
echo "Experiments complete!"
echo "========================================"
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Output files:"
echo "  - Evaluation report: $RESULTS_DIR/evaluation/report.md"
echo "  - Comparison figure: $RESULTS_DIR/evaluation/comparison.png"
echo "  - Raw results: $RESULTS_DIR/evaluation/evaluation_results.json"
echo "========================================"
