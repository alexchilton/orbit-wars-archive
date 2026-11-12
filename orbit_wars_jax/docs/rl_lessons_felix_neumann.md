# RL Lessons — Felix M Neumann / "felixmneumann" (3rd place, Orbit Wars)
Source: Kaggle writeup "[3rd Place] Ab in den Orbit". (No public repo linked.)

## Why this one matters most for US
His single most important training knob was the ENTROPY SCHEDULE — which is exactly the failure our
surviving logs show (entropy collapse to ~0.02 -> frozen mediocre policy). See
[[rl_postmortem_and_fix_plan]]. He also confirms terminal-only -1/+1 (4th independent winner to do so),
and that self-play RL is brutally noisy with mostly-failing runs and crippling late bugs — validating our
whole post-mortem. Cross-refs: [[rl_lessons_simjeg]], [[rl_lessons_isaiah_pressman]],
[[rl_lessons_lin_myat_ko]].

## The headline
Pure self-play RL (PPO + PFSP), NO imitation learning, first-ever RL attempt. Env + feature engineering
rewritten in JAX; model is a 6.2M-param transformer in Torch. 2×RTX 6000 Pro, ~19k SPS (2p) / 15k (4p),
8.4B steps (2p) / 2.7B (4p). Separate models for 2p and 4p.

## ENTROPY was THE knob (his words: "by far the most important")
- Spent the most time on entropy annealing schedules.
- Categoricals are simple, so no tricks needed: simply annealing entropy was enough for the agent to stop
  spamming tiny fleets and converge — for 2p.
- 4p is different: "the entropy seemingly needs to stay high."
- OUR runs died here: entropy collapsed to 0.02 (run10_scratch_nonorm), kl~0, clip~0 = frozen. The
  3rd-place finisher says the entropy schedule was the whole game. This is the knob to obsess over.

## Terminal rewards only (4th confirmation of the -1/+1 rule)
"I relied on terminal rewards only, although I experimented with reward shaping during the very early
training phases." Winner-take-all: +1 winner, -1 losers (-1/3 each in 4p to keep zero-sum). Tried shaping
early, dropped it. All four top solutions = terminal -1/+1, no meaningful shaping. STOP SHAPING.

## Self-play is brutally noisy — even 3rd place mostly failed
- "you can realistically only test whether something SPEEDS UP LEARNING, not whether it changes the final
  ceiling." Hidden dim / depth effects are ~impossible to establish rigorously.
- "at least three times as many runs that either plateaued or crashed to 0" than the ones shown.
- A 4p winrate jump "occurred around 1.8B after a critical bug fix"; "a crucial bug I only unearthed 36
  hours before the end severely crippled my 4p performance."
- Blames coding agents: "extremely easy to add bells and whistles faster than you can understand their
  implications." (Matches our finding that our failures were substantially BUGS, not reward design.)
=> Our inconclusive ablations + buggy runs were the NATURE of the problem, not incompetence.

## Action space: dropped fixed fractions -> "semantic" actions (big speedup)
- Most of the comp: 4 fixed fractions (0.25/0.5/0.75/1.0 of a planet's garrison).
- Last week: switched to SEMANTIC actions, ship count computed from the reachability tensor to satisfy an
  intent instead of a fixed fraction. "increased learning speed by a lot." The 4 semantic actions:
  - Send-all: launch entire garrison.
  - Sortie: send as much as possible without losing this planet to fleets already in flight.
  - Hold: send exactly enough to conquer + hold the target vs already-launched fleets for 8 rounds.
  - Kill-at-arrival: send exactly enough to conquer on arrival, accounting for in-flight fleets.
- Per-planet independent action head: each owned planet emits one categorical over {no-op} U {44 targets x
  4 actions} = 177-dim (same per-planet family as simjeg). Two heads: a no-op head + a launch head.
- Fixed fractions were a drag on learning for him too (echoes simjeg dropping fractions entirely).

## The "reachability tensor" (his key engineering idea)
Shape (B, P, P, S, 3): for every (source, target) pair and each of S=4 semantic actions, stores
[ship count, launch angle, arrival time]. Drives features, action head, semantic ship counts, and masking —
almost everything reads from it. "Computing it efficiently was almost as much work as the model itself."
Did it in JAX (vectorized collision checks + long XLA compile times; some 4p games timed out on compile).
Would do it in Rust/C++ next time to avoid uncontrolled JIT compile inside the per-turn limit. Collision
horizon matters: on some maps not being able to fly >16 turns = losing.

## Architecture / training details
- Pre-norm transformer, 48 tokens (4 player + 44 planet/comet), dim=192, expansion 4, 8 trunk layers.
  Critic = 2 layers + MLP on player tokens only (per-player value baseline, ego + opponents in one pass).
  Actor = 2 layers on planet tokens + the action head.
- Launch head = 3 additive terms: (a) launch-count-aware bilinear (factorizable source·target), (b) a
  full-rank per-(source,target,action) edge MLP run ONLY on reachable edges via gather/scatter (breaks
  torch.compile but far cheaper backward), (c) per-semantic-action bias.
- Features: embeddings NOT scalars for garrison/fleet sizes ("transformers are bad at math"; exact to 384,
  sqrt-bins to 768, clamp above). "Arrival calendar" grid of incoming fleets per player over the horizon
  (24 in 2p / 16 in 4p). FiLM conditioning from global scalars -> per-block scale/shift. Graphormer-style
  attention biases from the reachability tensor so attention respects who-can-reach-whom.
- PPO + GAE, pure self-play, PFSP (Prioritized Fictitious Self-Play). DDP but only 2 GPUs.
- Hyperparams: GAE lambda 0.95; gamma 0.993 (2p) / 0.99 (4p); **adaptive KL-targeted LR** (controller keeps
  mean approx-KL near a budget instead of fixed decay); rollout 256; **PPO epochs=1** ("more epochs mostly
  bought instability rather than sample efficiency" with fresh self-play data each update); minibatch 8192;
  num_envs 1024; clip 0.2.
- Eval: 1024 parallel games vs prior checkpoints; 2p near-monotonic winrate; 4p evaluated 1v3 vs clones
  (regrets not adding checkpoint variety). T=0.1 sampling temp in 4p submission.

## Takeaways for our project
- ENTROPY SCHEDULE is the knob we lost on and the one 3rd place says matters most. Anneal it deliberately;
  for 4p keep it higher/longer. Watch entropy every update; a crash to ~0 = dead run.
- Terminal -1/+1, no shaping (4th confirmation). Zero-sum it in 4p (-1/3 losers).
- Compute: he ran 8.4B steps; we ran ~100K-230K. ~40,000x short (see [[rl_postmortem_and_fix_plan]]).
- Adaptive KL-targeted LR + PPO epochs=1 are cheap stability wins vs our kl-death / instability.
- Per-planet no-op-or-(target x semantic action) head; drop fixed fractions (both simjeg and Felix did).
- Bugs and dead runs are normal even at podium level (3x more failed runs, crippling late bugs). Snapshot
  the whole pipeline (he kept a zoo/ folder) so you can eval old versions faithfully.