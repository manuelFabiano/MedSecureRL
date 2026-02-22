#!/bin/bash
# Train victim models and RL agent for MedSecure
# Usage: ./scripts/train_all.sh [dataset] [model]

set -e

DATASET=${1:-"pathmnist"}
MODEL=${2:-"resnet18"}
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"

echo "========================================"
echo "MedSecure - Full Training Pipeline"
echo "========================================"
echo "Dataset: $DATASET"
echo "Model: $MODEL"
echo "========================================"

# Create directories
mkdir -p $CHECKPOINT_DIR $RESULTS_DIR/agent $RESULTS_DIR/baselines

# Step 1: Train victim model
echo ""
echo "[1/3] Training victim model..."
echo "----------------------------------------"
python experiments/train_victim.py \
    --model $MODEL \
    --dataset $DATASET \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --checkpoint-dir $CHECKPOINT_DIR \
    --early-stopping 10

VICTIM_PATH="$CHECKPOINT_DIR/${MODEL}_${DATASET}_best.pth"
echo "Victim model saved to: $VICTIM_PATH"

# Step 2: Run baseline attacks
echo ""
echo "[2/3] Running baseline attacks..."
echo "----------------------------------------"
python experiments/run_baselines.py \
    --model-path $VICTIM_PATH \
    --dataset $DATASET \
    --n-samples 1000 \
    --epsilon 0.03 \
    --output-dir $RESULTS_DIR/baselines

# Step 3: Train RL agent
echo ""
echo "[3/3] Training RL agent..."
echo "----------------------------------------"
python experiments/train_agent.py \
    --model-path $VICTIM_PATH \
    --dataset $DATASET \
    --episodes 50000 \
    --output-dir $RESULTS_DIR/agent \
    --eval-interval 1000 \
    --warmup-steps 10000

echo ""
echo "========================================"
echo "Training complete!"
echo "========================================"
echo "Victim model: $VICTIM_PATH"
echo "Agent checkpoint: $RESULTS_DIR/agent/agent_best.pth"
echo "Baseline results: $RESULTS_DIR/baselines/"
echo "========================================"
