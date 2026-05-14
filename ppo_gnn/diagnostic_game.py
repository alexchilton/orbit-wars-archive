"""Play a single diagnostic game and print step-by-step summary."""
import sys, math, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ppo_gnn.edge_policy import (
    EdgePolicy, MAX_ACTIONS, MAX_CANDIDATES, EDGE_INPUT_DIM,
    FRACTION_BUCKETS, compute_candidate_edges,
)


def run_diagnostic(checkpoint_path: str, opponent_name: str = "random", mode: str = "2p"):
    from kaggle_environments import make
    from kaggle_environments.envs.orbit_wars.orbit_wars import random_agent, starter_agent

    # Load model
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    d_model = ckpt.get("d_model", 128)
    separate_critic = ckpt.get("separate_critic", True)

    model = EdgePolicy(d_model=d_model, n_heads=4, n_layers=3,
                       max_actions=MAX_ACTIONS, separate_critic=separate_critic)
    # Filter out shape-mismatched weights (e.g. fraction head 10->16)
    model_sd = model.state_dict()
    filtered_sd = {k: v for k, v in sd.items()
                   if k not in model_sd or v.shape == model_sd[k].shape}
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()

    # Pick opponent
    root = Path(__file__).parent.parent
    opp_map = {"random": random_agent, "starter": starter_agent}
    file_map = {
        "bully": "submission/pool_bully.py",
        "rage": "submission/pool_rage.py",
        "dual": "submission/pool_dual.py",
        "prospector": "submission/pool_prospector.py",
        "nearest_sniper": "submission/ext/pool_baseline_nearest_sniper.py",
        "baseline": "submission/pool_baseline.py",
        "sig_starter": "submission/ext/pool_sigmaborov_starter.py",
        "pascal_v14": "submission/ext/pool_pascal_orbitwork_v14.py",
        "shunlite": "submission/main_fc_rl_shunlite.py",
        "v131_2p": "submission/main_v131_plus_2p.py",
    }
    if opponent_name in opp_map:
        opponent = opp_map[opponent_name]
    elif opponent_name in file_map:
        from ppo_gnn.train_ppo_edge import load_agent_from_file
        opponent = load_agent_from_file(str(root / file_map[opponent_name]))
    else:
        # Try loading from file path directly
        from ppo_gnn.train_ppo_edge import load_agent_from_file
        opponent = load_agent_from_file(opponent_name)

    env = make("orbit_wars", debug=False)
    num_players = 2 if mode == "2p" else 4
    if mode == "2p":
        trainer = env.train([None, opponent])
    else:
        trainer = env.train([None] + [opponent] * 3)

    obs = trainer.reset()
    max_steps = 400

    print(f"Diagnostic: {checkpoint_path} vs {opponent_name} ({mode})")
    print(f"{'Step':>4} | {'Planets':>7} | {'Ships':>6} | {'Prod':>5} | {'Actions':>7} | Details")
    print("-" * 80)

    total_launches = 0
    total_noops = 0
    total_ships_sent = 0
    frac_counts = [0] * len(FRACTION_BUCKETS)
    max_planets_held = 0

    for step_idx in range(max_steps):
        planets = obs.get("planets", [])
        fleets = obs.get("fleets", [])
        player = obs.get("player", 0)

        if not planets:
            obs, reward, done, info = trainer.step([])
            if done:
                break
            continue

        # Count stats
        my_planets = [p for p in planets if int(p[1]) == player]
        enemy_planets = [p for p in planets if int(p[1]) >= 0 and int(p[1]) != player]
        neutral_planets = [p for p in planets if int(p[1]) < 0]
        my_ships = sum(float(p[5]) for p in my_planets)
        my_prod = sum(float(p[6]) for p in my_planets)
        my_fleets = [f for f in fleets if int(f[1]) == player]
        fleet_ships = sum(float(f[2]) for f in my_fleets)

        has_planets = len(my_planets) > 0
        if not has_planets:
            obs, reward, done, info = trainer.step([])
            if done:
                break
            continue

        ef, edge_indices, em, num_valid = compute_candidate_edges(
            planets=planets, fleets=fleets, player_id=player,
            num_players=num_players, step=step_idx, max_steps=max_steps,
        )
        ef = torch.nan_to_num(ef, nan=0.0, posinf=1.0, neginf=-1.0)

        with torch.no_grad():
            selected, fracs, is_noop, log_prob, value = model.sample_action(ef, em)

        actions = []
        action_details = []
        if not is_noop and num_valid > 0:
            for cand_idx, frac_idx in zip(selected, fracs):
                if cand_idx >= num_valid:
                    continue
                src_pidx = edge_indices[cand_idx, 0].item()
                tgt_pidx = edge_indices[cand_idx, 1].item()
                src_p = planets[src_pidx]
                tgt_p = planets[tgt_pidx]
                sx, sy = float(src_p[2]), float(src_p[3])
                tx, ty = float(tgt_p[2]), float(tgt_p[3])
                angle = math.atan2(ty - sy, tx - sx)
                src_fleet = int(float(src_p[5]))
                tgt_ships = float(tgt_p[5])
                tgt_owner = int(tgt_p[1])

                # Ship reserve — keep some ships on planet
                reserve = min(5, int(src_fleet * 0.2))
                available = src_fleet - reserve
                if available < 1:
                    continue

                ships = max(1, int(available * FRACTION_BUCKETS[frac_idx]))

                # Min capture viability
                min_send = 12
                if ships < min_send:
                    if available >= min_send:
                        ships = min_send
                    else:
                        continue

                # Travel-aware viability
                dist = math.sqrt((tx - sx)**2 + (ty - sy)**2)
                travel_time = dist / 6.0
                if tgt_owner != player:
                    tgt_prod = float(tgt_p[6])
                    ships_on_arrival = tgt_ships + tgt_prod * travel_time
                    needed = int(ships_on_arrival * 1.05) + 2
                    if ships < needed:
                        continue

                # Max ETA filter
                max_eta = 14 if step_idx < 50 else (20 if step_idx < 150 else 30)
                if travel_time > max_eta:
                    continue

                # Early-game patience
                num_my = len(my_planets)
                if step_idx < 30 and num_my == 1:
                    if tgt_owner >= 0 and tgt_owner != player:
                        continue
                    if len(actions) >= 1:
                        break

                actions.append([int(src_p[0]), angle, ships])
                frac_counts[frac_idx] += 1
                tgt_label = "own" if tgt_owner == player else ("neutral" if tgt_owner < 0 else "enemy")
                action_details.append(f"P{int(src_p[0])}->{tgt_label}(P{int(tgt_p[0])}) {ships}s @{FRACTION_BUCKETS[frac_idx]:.0%}")

        # Print every 10 steps or when something happens
        if step_idx % 10 == 0 or actions:
            detail_str = ", ".join(action_details) if action_details else ("NOOP" if not actions else "")
            planet_str = f"{len(my_planets)}m/{len(enemy_planets)}e/{len(neutral_planets)}n"
            print(f"{step_idx:4d} | {planet_str:>7} | {my_ships:6.0f} | {my_prod:5.1f} | {len(actions):>7} | {detail_str}")

        total_launches += len(actions)
        total_ships_sent += sum(a[2] for a in actions)
        max_planets_held = max(max_planets_held, len(my_planets))
        if not actions:
            total_noops += 1

        obs, reward, done, info = trainer.step(actions)
        if done:
            break

    # Final state
    final_planets = obs.get("planets", []) if obs else []
    my_final = sum(1 for p in final_planets if int(p[1]) == player)
    enemy_final = sum(1 for p in final_planets if int(p[1]) >= 0 and int(p[1]) != player)

    print("-" * 80)
    outcome = "WIN" if (reward and reward > 0) else ("LOSS" if (reward and reward < 0) else "DRAW/TIMEOUT")
    print(f"Result: {outcome} at step {step_idx}")
    print(f"Final planets: {my_final} mine, {enemy_final} enemy")
    print(f"Total launches: {total_launches}, Total NOOPs: {total_noops}")
    print(f"Launch rate: {total_launches/(step_idx+1):.2f}/step")
    mean_fleet = total_ships_sent / max(total_launches, 1)
    print(f"Mean fleet per send: {mean_fleet:.1f} ships")
    print(f"Total ships sent: {total_ships_sent}")
    print(f"Max planets held: {max_planets_held}")
    print(f"Fraction distribution: {', '.join(f'{FRACTION_BUCKETS[i]:.0%}:{frac_counts[i]}' for i in range(len(FRACTION_BUCKETS)) if frac_counts[i] > 0)}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "ppo_gnn/cache/checkpoint_ppo_edge_latest.pt"
    opp = sys.argv[2] if len(sys.argv) > 2 else "random"
    run_diagnostic(ckpt, opp)
