"""Quick smoke test for the edge-based policy."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ppo_gnn.edge_policy import (
    EdgePolicy, compute_candidate_edges, MAX_CANDIDATES, EDGE_INPUT_DIM,
    FRACTION_BUCKETS, MAX_ACTIONS, NUM_FRACTIONS,
)
import math
import random


def make_fake_planets(n=48):
    """Create fake planet data matching Kaggle format."""
    planets = []
    for i in range(n):
        angle = random.uniform(0, 2 * math.pi)
        r = random.uniform(15, 45)
        x = 50 + r * math.cos(angle)
        y = 50 + r * math.sin(angle)
        owner = random.choice([-1, -1, -1, 0, 0, 1, 1])  # mostly neutral
        ships = random.uniform(0, 50) if owner >= 0 else random.uniform(5, 30)
        prod = random.uniform(1, 5)
        orbit_vel = random.uniform(0, 2)
        planets.append([i, owner, x, y, orbit_vel, ships, prod])
    return planets


def make_fake_fleets(planets, n_fleets=10):
    """Create fake fleet data."""
    fleets = []
    for i in range(n_fleets):
        owner = random.choice([0, 1])
        src = random.choice(planets)
        x = float(src[2]) + random.uniform(-10, 10)
        y = float(src[3]) + random.uniform(-10, 10)
        angle = random.uniform(0, 2 * math.pi)
        ships = random.uniform(5, 30)
        fleets.append([i, owner, x, y, angle, int(src[0]), ships])
    return fleets


def test_candidate_generation():
    print("=== Test: Candidate Edge Generation ===")
    planets = make_fake_planets()
    fleets = make_fake_fleets(planets)

    edge_features, edge_indices, edge_mask, num_valid = compute_candidate_edges(
        planets=planets,
        fleets=fleets,
        player_id=0,
        num_players=2,
        step=50,
        max_steps=500,
    )

    print(f"  Planets: {len(planets)}")
    print(f"  My planets: {sum(1 for p in planets if p[1] == 0)}")
    print(f"  Valid candidates: {num_valid} / {MAX_CANDIDATES}")
    print(f"  Edge features shape: {edge_features.shape}")
    print(f"  Edge indices shape: {edge_indices.shape}")
    print(f"  Edge mask sum: {edge_mask.sum().item():.0f}")
    assert edge_features.shape == (MAX_CANDIDATES, EDGE_INPUT_DIM), \
        f"Expected ({MAX_CANDIDATES}, {EDGE_INPUT_DIM}), got {edge_features.shape}"
    assert edge_mask.sum().item() == num_valid
    print("  PASSED ✓\n")


def test_policy_forward():
    print("=== Test: Policy Forward Pass ===")
    model = EdgePolicy(d_model=128, n_heads=4, n_layers=3, separate_critic=True)

    params = model.count_parameters()
    print(f"  Parameters:")
    for k, v in params.items():
        print(f"    {k}: {v:,}")

    B, K = 4, MAX_CANDIDATES
    edge_feats = torch.randn(B, K, EDGE_INPUT_DIM)
    edge_mask = torch.ones(B, K)
    # Mask out last 50 as padding
    edge_mask[:, -50:] = 0.0

    noop_logit, edge_logits, frac_logits, value = model(edge_feats, edge_mask)

    print(f"  Noop logit shape: {noop_logit.shape}")        # (B, 1)
    print(f"  Edge logits shape: {edge_logits.shape}")       # (B, K)
    print(f"  Fraction logits shape: {frac_logits.shape}")   # (B, K, 10)
    print(f"  Value shape: {value.shape}")                   # (B, 1)

    assert noop_logit.shape == (B, 1)
    assert edge_logits.shape == (B, K)
    assert frac_logits.shape == (B, K, NUM_FRACTIONS)
    assert value.shape == (B, 1)
    # Padded positions should have -inf edge logits
    assert (edge_logits[:, -50:] == float('-inf')).all(), "Padded edge logits should be -inf"
    print("  PASSED ✓\n")


def test_sample_action():
    print("=== Test: Action Sampling ===")
    model = EdgePolicy(d_model=64, n_heads=4, n_layers=2, separate_critic=True)

    planets = make_fake_planets()
    fleets = make_fake_fleets(planets)

    edge_features, edge_indices, edge_mask, num_valid = compute_candidate_edges(
        planets, fleets, player_id=0, num_players=2, step=50, max_steps=500,
    )

    selected, fractions, is_noop, log_prob, value = model.sample_action(
        edge_features, edge_mask, deterministic=False,
    )

    print(f"  Selected edges: {selected}")
    print(f"  Fractions: {fractions} = {[FRACTION_BUCKETS[f] for f in fractions]}")
    print(f"  Is noop: {is_noop}")
    print(f"  Log prob: {log_prob:.4f}")
    print(f"  Value: {value:.4f}")

    if not is_noop:
        for i, (edge_idx, frac_idx) in enumerate(zip(selected, fractions)):
            src, tgt = edge_indices[edge_idx].tolist()
            src_p = planets[src]
            tgt_p = planets[tgt]
            ships_to_send = int(src_p[5] * FRACTION_BUCKETS[frac_idx])
            print(f"  Action {i+1}: planet {src_p[0]} ({src_p[5]:.0f} ships) "
                  f"→ planet {tgt_p[0]} ({tgt_p[5]:.0f} ships, owner={tgt_p[1]}) "
                  f"send {ships_to_send} ({FRACTION_BUCKETS[frac_idx]*100:.0f}%)")

    assert len(selected) <= MAX_ACTIONS
    assert len(selected) == len(fractions)
    print("  PASSED ✓\n")


def test_evaluate_action():
    print("=== Test: Action Evaluation (PPO) ===")
    model = EdgePolicy(d_model=64, n_heads=4, n_layers=2, separate_critic=True)

    B, K = 8, MAX_CANDIDATES
    edge_feats = torch.randn(B, K, EDGE_INPUT_DIM)
    edge_mask = torch.ones(B, K)
    edge_mask[:, -30:] = 0.0

    # Simulate actions: some with 1, 2, or 3 edges; some noop
    action_edges = torch.zeros(B, MAX_ACTIONS, dtype=torch.long)
    action_fractions = torch.zeros(B, MAX_ACTIONS, dtype=torch.long)
    action_counts = torch.tensor([3, 2, 1, 0, 3, 1, 2, 0])

    for b in range(B):
        for a in range(action_counts[b].item()):
            action_edges[b, a] = random.randint(0, K - 31)  # valid range
            action_fractions[b, a] = random.randint(0, NUM_FRACTIONS - 1)

    log_prob, entropy, value = model.evaluate_action(
        edge_feats, edge_mask, action_edges, action_fractions, action_counts,
    )

    print(f"  Log prob shape: {log_prob.shape}, values: {log_prob.tolist()}")
    print(f"  Entropy shape: {entropy.shape}, values: {[f'{e:.3f}' for e in entropy.tolist()]}")
    print(f"  Value shape: {value.shape}")

    assert log_prob.shape == (B,)
    assert entropy.shape == (B,)
    assert value.shape == (B,)
    assert (entropy >= 0).all(), "Entropy should be non-negative"
    print("  PASSED ✓\n")


def test_gradient_flow():
    print("=== Test: Gradient Flow ===")
    model = EdgePolicy(d_model=64, n_heads=4, n_layers=2, separate_critic=True)

    B, K = 4, MAX_CANDIDATES
    edge_feats = torch.randn(B, K, EDGE_INPUT_DIM, requires_grad=False)
    edge_mask = torch.ones(B, K)

    action_edges = torch.randint(0, K, (B, MAX_ACTIONS))
    action_fractions = torch.randint(0, NUM_FRACTIONS, (B, MAX_ACTIONS))
    action_counts = torch.tensor([2, 3, 1, 3])

    log_prob, entropy, value = model.evaluate_action(
        edge_feats, edge_mask, action_edges, action_fractions, action_counts,
    )

    # PPO-style loss
    loss = -log_prob.mean() + 0.5 * (value - 1.0).pow(2).mean() - 0.05 * entropy.mean()
    loss.backward()

    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    print(f"  Parameters with gradient: {has_grad}/{total}")
    print(f"  Loss: {loss.item():.4f}")
    assert has_grad > 0, "No gradients flowed!"
    print("  PASSED ✓\n")


def test_planet_count_reward_shaping():
    """Planet count bonus should give higher reward when holding more planets.

    The dense planet_count_bonus (per step) should:
    1. Scale with number of owned planets
    2. Be small relative to win/loss (±10) — around 0.005-0.01 per planet
    3. Give strictly more reward for more planets
    """
    print("=== Test: Planet Count Reward Shaping ===")
    from ppo_gnn.train_ppo_edge import compute_planet_count_bonus

    # Scenario 1: holding 1 planet
    bonus_1 = compute_planet_count_bonus(my_planet_count=1)
    # Scenario 2: holding 5 planets
    bonus_5 = compute_planet_count_bonus(my_planet_count=5)
    # Scenario 3: holding 10 planets
    bonus_10 = compute_planet_count_bonus(my_planet_count=10)

    print(f"  1 planet:  bonus = {bonus_1:.4f}")
    print(f"  5 planets: bonus = {bonus_5:.4f}")
    print(f"  10 planets: bonus = {bonus_10:.4f}")

    # More planets = more bonus (monotonically increasing)
    assert bonus_5 > bonus_1, f"5 planets ({bonus_5}) should give more than 1 ({bonus_1})"
    assert bonus_10 > bonus_5, f"10 planets ({bonus_10}) should give more than 5 ({bonus_5})"

    # Bonus should be small relative to win/loss (±10)
    # Over 500 steps with 10 planets, total bonus should be < win_bonus
    max_total = bonus_10 * 500
    assert max_total < 10.0, f"Total bonus over 500 steps ({max_total:.2f}) should be < win_bonus (10)"

    # Bonus for 0 planets should be 0 or negative (no reward for holding nothing)
    bonus_0 = compute_planet_count_bonus(my_planet_count=0)
    assert bonus_0 <= 0.0, f"0 planets should give no bonus, got {bonus_0}"

    print("  PASSED ✓\n")


def test_fraction_head_freezing():
    """Fraction-only training must freeze everything except fraction_head."""
    print("=== Test: Fraction Head Freezing ===")
    from ppo_gnn.train_fraction_head import fraction_only_loss

    model = EdgePolicy(d_model=64, n_heads=4, n_layers=2, separate_critic=True)

    # Freeze everything, unfreeze fraction head (as train_fraction_head does)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fraction_head.parameters():
        param.requires_grad = True

    # Snapshot weights before
    edge_enc_before = model.edge_encoder[0].weight.clone()
    sel_head_before = model.selection_head[0].weight.clone()
    frac_head_before = model.fraction_head[0].weight.clone()

    # Fake batch
    B, K = 4, MAX_CANDIDATES
    batch = {
        "edge_features": torch.randn(B, K, EDGE_INPUT_DIM),
        "edge_mask": torch.ones(B, K),
        "action_edges": torch.randint(0, K, (B, MAX_ACTIONS)),
        "action_fractions": torch.randint(0, NUM_FRACTIONS, (B, MAX_ACTIONS)),
        "action_counts": torch.tensor([2, 3, 1, 3]),
    }

    optimizer = torch.optim.Adam(model.fraction_head.parameters(), lr=1e-2)
    for _ in range(5):
        loss, metrics = fraction_only_loss(model, batch, torch.device("cpu"))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Check: frozen weights unchanged
    assert torch.equal(model.edge_encoder[0].weight, edge_enc_before), \
        "edge_encoder should be frozen!"
    assert torch.equal(model.selection_head[0].weight, sel_head_before), \
        "selection_head should be frozen!"

    # Check: fraction head DID change
    assert not torch.equal(model.fraction_head[0].weight, frac_head_before), \
        "fraction_head should have updated!"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    print(f"  Fraction loss after 5 steps: {metrics['frac_loss']:.4f}")
    print(f"  Fraction acc: {metrics['frac_acc']:.1%}")
    print("  PASSED ✓\n")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    test_candidate_generation()
    test_policy_forward()
    test_sample_action()
    test_evaluate_action()
    test_gradient_flow()
    test_planet_count_reward_shaping()
    test_fraction_head_freezing()

    print("=" * 50)
    print("ALL TESTS PASSED ✓")

    # Compare parameter counts
    print("\n=== Parameter Comparison ===")
    edge_model = EdgePolicy(d_model=128, n_heads=4, n_layers=3, separate_critic=True)
    params = edge_model.count_parameters()
    print(f"Edge-based (d=128, L=3): {params['total']:,} total")

    edge_small = EdgePolicy(d_model=64, n_heads=4, n_layers=2, separate_critic=True)
    params_s = edge_small.count_parameters()
    print(f"Edge-based (d=64, L=2):  {params_s['total']:,} total")
    print(f"Current GNN policy:      ~1,240,000 total (for comparison)")
