# Orbit Wars (Kaggle): archive

Private archive of my work on the Kaggle Orbit Wars competition (May to August 2026), taken from the laptop on 2026-11-12.
The project never had a GitHub remote. The first 3 commits are the original local history; the last commit holds everything that was uncommitted on disk.

Start here:

- `LESSONS_LEARNED.md`: what worked and what did not.
- `CLAUDE.md`: project rules and context for agent sessions.
- `docs/` and `ppo_gnn/tutorial.md`: design notes and the PPO + GNN tutorial.

Layout:

- `ppo_gnn/`: PPO with a graph network policy (stable-baselines3), behaviour cloning and offline RL trainers, tests.
- `orbit_wars_jax/`: the JAX version of the environment and trainer.
- `submission/`, `submission_edge/`, `submission_gnn/`: submission builds, with the trained weights (`*.pt`, force-added; they are git-ignored).
- `scripts/`: genome search runners, replay download and analysis tools.
- `agents/`, `snapshots/`, `notebooks/`, `genome test/`, `rl midgame/`, `rl_fundamentals/`: opponents, policy snapshots and experiments.

Not archived: `__pycache__`, IDE settings, session traces (`.claude/`).
