hpriweno di yeahdsyeas# RL Lessonwei/
s — Isaiah Pressman / "IsaiahP" (1st place, Orbit Wars)
Source: Kaggle writeup "Scaling Reinforcement Learning to the Stars" + open-source repo
(github.com/IsaiahPressman/kaggle-orbit-wars)

## The headline
Single **200M-param transformer**, **15B steps** of pure self-play PPO, plays both 2P and 4P.
~2400 B200-hours (4×8×B200, 8192 parallel envs).
This is the Bitter-Lesson opposite of Lin Myat Ko's ~600K-param / $150 run — same recipe, 330× the scale.
See [[rl_lessons_lin_myat_ko]] for the lean counterpoint.

## Core thesis: scale beats engineering
Deliberately kept obs/action encoding as low-level as possible ("no crutches") and let the big
model learn the dynamics. Only concession that mattered: **action = target planet, not raw angle**.
Rejected planet-fleet cross-attention and discrete fleet bins as unnecessary.
Every time he grew the model, performance jumped — he scaled until submission limits stopped him.

**Same core recipe as everyone at the top: entity transformer + PPO + self-play + naked +1/-1 reward.**
The differentiator was purely scale + the compute to feed it.

## Architecture specifics worth stealing
- Per-entity-type MLP projection into a shared **768-dim** space (planets/comets/fleets).
- **17 special tokens**: 4 player-summary + 1 global (projected from features) + 4 actor-plan +
  4 value + 4 **scratch** tokens (learned embeddings, no assigned job — a shared attention workspace).
- 38-block residual self-attention, 16 heads, MLP hidden 1536.
- Action head: source stream → Bernoulli launch logit; target via attention `Q(src)·K(tgt)/√d`;
  target's V folded into source stream → **truncated discretized logistic mixture (8 components)**
  for fleet size in [3, num_ships].
- Critic: value tokens → MLP → softmax over remaining players = win probability.

## The efficiency trick: one forward pass for ALL players
Concatenate each player's summary+plan tokens with the shared entity tokens; compute every player's
action in a single forward pass. **2-4× compute saving** vs the standard one-observation-per-player loop.

## Counterintuitive findings
- **Removing the action mask made the model BETTER.** Letting it launch ships into the sun forced it to
  internally model the physics, which helped elsewhere. Only re-added the mask for end-of-run
  fine-tuning + test time.
- **gamma = 1.0 (no discounting)** to keep the 4P win-prob head well-defined. Side effect: the model had
  no reason to win *now*, so it grabs a ship lead then **stalls** for the rest of the game. Fine at
  inference, wasteful in training (compute burned on already-decided states).
  → next time: add early truncation / surrender to raise the density of relevant states.

## Self-play stabilization
- 2P and 4P run simultaneously (evenly weighted early). Pure self-play for throughput.
- **Best-checkpoint gating**: new model must win **>70%** of eval games (1v1 + 2v2) to replace best.
- Added **policy-KL + cross-entropy-value loss vs previous best** on top of GAE-λ, clipped PG w/
  advantage norm, and entropy bonus.

## Deployment under Kaggle limits (1s/turn slow CPU, 100 MiB)
- **int8 dynamic quant** on linear layers for serving; cap visible fleets (keep largest).
- **4-bit NormalFloat codebook quant, group size 128, one fp16 scale/group** to fit 200M in 100 MiB.
  Kept most perf (won ~40% head-to-head vs full precision). 3-bit lost too much → stopped at 200M.
- **Fallback model**: on a too-slow CPU, play normally until 1s overage remains, then hand off to a
  fast **5M model** to finish. Critic said most games were already decided; the 5M converted 100% of wins.

## Infra
- Rewrote the slow Python env in **Rust**, verified with extensive **replay parity tests**.
- Rust computes launch angles from (source, target) pairs, avoiding sun/planets.
- Preallocated pinned CPU-GPU buffers; many envs in parallel; PPO chosen over IMPALA because it
  scales by "just add GPUs" (linear throughput, multi-node DDP).

## Agentic development style (the actually-novel part)
Wrote **no code by hand** (Codex), reviewed as little as possible — but the repo is disciplined, not slop:
- **Docs mapped to code, enforced by CI.** `just docs-fresh` fails the build if mapped code changes
  without updating its doc (`DOCS_CURRENT=1` to acknowledge a no-op). He reviewed only the docs —
  the freshness gate is what made that safe.
- `AGENTS.md` = a "repository map": which doc to read before touching which module.
- `just prepare` = format + lint + mypy + tests as one gate; PR checklist modeled on OpenAI's
  harness-engineering writeup.
- mypy-forward "pseudo-typed" style, fail-fast errors, no `getattr`/`setattr` in hot paths.
- **Discard codebases between experiments** (a competitor who reused one codebase for the agentic
  approach reported it "blew up" near the end).

## What he'd do differently
- **2P/4P balance.** Assumed the leaderboard favored 2P and went 90% 2P + skipped league play.
  Post-deadline matchmaking inverted to mostly 4P and burned him. Would keep rates balanced and add
  **league play vs past checkpoints** to prevent strategic cycles / self-overfitting.
- Add surrender/truncation to fix the stalling compute waste (see gamma=1.0 above).

## Takeaways for our project
- Our stack is closer to Lin Myat Ko's lean end; the transferable ideas here are the *cheap* ones:
  single-forward-pass-all-players, best-checkpoint gating at 70%, KL-to-previous-best for stability,
  and the doc-mapping trick for keeping agentic dev honest.
- The stalling/gamma=1.0 warning is directly relevant if we ever move to a win-prob value head in 4P.
- The "no action mask trained a better model" result is worth an ablation before we assume masking helps.