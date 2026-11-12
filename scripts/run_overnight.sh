#!/bin/bash
# Overnight pipeline: BC edge training → PPO edge fine-tuning
# Logs: ppo_gnn/cache/bc_edge_train.log  and  ppo_gnn/cache/ppo_edge_train.log

set -e
cd /Users/alexchilton/DataspellProjects/orbit_wars

echo "========================================="
echo "PHASE 1: BC Edge Training"
echo "Started: $(date)"
echo "========================================="

python3 -u -m ppo_gnn.train_bc_edge \
  --replay-dir kaggle_replays \
  --epochs 30 \
  --batch-size 64 \
  --lr 3e-4 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 3 \
  --patience 8 \
  --device mps \
  2>&1 | tee ppo_gnn/cache/bc_edge_train.log

echo ""
echo "========================================="
echo "PHASE 2: PPO Edge Fine-Tuning"
echo "Started: $(date)"
echo "========================================="

python3 -u -m ppo_gnn.train_ppo_edge \
  --checkpoint ppo_gnn/cache/checkpoint_bc_edge.pt \
  --mode mixed \
  --num-episodes 50000 \
  --episodes-per-update 8 \
  --lr 3e-5 \
  --d-model 128 \
  --n-heads 4 \
  --n-layers 3 \
  --entropy-coef 0.02 \
  --gamma 0.997 \
  --gae-lambda 0.95 \
  --max-steps 500 \
  --eval-every 200 \
  --pid-lr \
  --target-kl 0.01 \
  --device cpu \
  --update-device mps \
  2>&1 | tee ppo_gnn/cache/ppo_edge_train.log

echo ""
echo "========================================="
echo "DONE: $(date)"
echo "BC checkpoint:  ppo_gnn/cache/checkpoint_bc_edge.pt"
echo "PPO best:       ppo_gnn/cache/checkpoint_ppo_edge_best.pt"
echo "PPO latest:     ppo_gnn/cache/checkpoint_ppo_edge_latest.pt"
echo "========================================="
