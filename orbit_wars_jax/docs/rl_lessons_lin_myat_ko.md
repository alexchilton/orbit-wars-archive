# RL Lessons — Lin Myat Ko (2nd place, Orbit Wars)
Source: Kaggle discussion post

## Core lessons

**Fast env is non-negotiable.** If the environment is slow, don't attempt RL. Target: ~10K SPS
(though their best submission was ~600M steps over 3 days, so effective wall-time SPS was lower).

**Architecture: entity transformer, ~600K params.** Same as our setup.

**Reward: +1/-1 is enough for 2P mode.** No shaping needed.

**One architecture delta at a time. Always.** Shipping 7 changes at once = can't diagnose what broke.
The "stupider" architecture that trains is worth more than a "smarter" one that doesn't.
A working baseline's limitations may be doing free regularization nobody notices.

## Explained variance is the key diagnostic
> "It should go up to at least 0.8 in 100 iters. I checked my earliest run. It got to 0.9 in 20 iters.
> If explained variance never gets past 0.5, check your obs representation or architecture.
> I would suspect obs representations. Start with simple features and simple model."

**If ev stays at 0.000, obs or arch is broken.**

## clip_frac is the earliest warning sign
> "Before entropy_fire collapses or KL spikes, clip_frac starts creeping up monotonically
> (typically 0.10 → 0.30+ over a few million samples). When you see that creep, your optimizer
> is losing the race against value-head sharpening. Cut lr or revert capacity."

## Transformer-PPO requires proper LR schedule
Vanilla PPO (constant LR, single ent_coef) breaks with transformers.
Required: **warmup + cosine decay + careful entropy schedules**.
This should be step 1, not an afterthought.

## Action heads
Separate target head and fraction head. argmax during inference, softmax (categorical) during training.
RL policy is robust enough to learn the fraction to send.

## Self-play
Self-play from the start works fine for this game with +1/-1 reward.
If win rate against eval set collapses: suspect distribution shift when new self-play snapshots enter
the opponent pool.

## AI assistant notes (Opus / Claude)
- Useful for: code ports, parity tests, analysis scripts, mechanical refactors
- Bad at: separating "interesting research direction" from "what your project needs", knowing when to stop,
  remembering corrections from yesterday
- Architecture-level calls and "is this run dead?" decisions belong with the human
- clip_frac is the warning sign the AI missed until it was too late

## Budget reference
- Claude Code: $100, Vast.ai 5090: ~$150 total
- Best submission (F12): ~3 days training, ~600M steps, ~600K params
