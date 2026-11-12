# RL Lessons — Billy Bradley / "sinkingpoint" — "Ender" (8th place, Orbit Wars)
Source: Kaggle writeup "8th Place: How I Made Ender for <$200".
OPEN SOURCE CODE: https://github.com/sinking-point/ender  <-- we can read/fork this.

## WHY THIS IS OUR BLUEPRINT
Built in "compute poverty" on the SAME hardware class we have: an ageing 3080 10GB + 11 days of a rented
4090, ~$170 total. Final 2p run = 3.4 days on the 4090 (~$51). The 4p model was trained ENTIRELY on the
3080. And it solves the TWO exact failures our runs died on: tiny-fleet-spam / entropy collapse, and the
action-space explosion. Cross-refs: [[rl_postmortem_and_fix_plan]], [[rl_lessons_simjeg]],
[[rl_lessons_felix_neumann]].

## Compute (the encouraging part)
- 2p: ~3.1B samples over ~1B env steps. 4p: ~1.5B samples over ~413M env steps.
- Even the compute-POOREST finisher ran ~1B steps. We ran ~200K. Still ~5,000x more — BUT reachable on a
  3080/4090 in DAYS (~$50-170), not a frontier cluster. This is the realistic target for us.
- ~15k micro-steps/s on the 4090, peaking ~4k env-steps/s incl. PPO. JAX env port.

## Fix #1 — tiny-fleet spam / entropy: KL-to-prior INSTEAD of an entropy bonus
He had our EXACT problem: "agents insisting on sending large numbers of small fleets." Higher halt-bias
init "quickly regressed." Key insight: a standard entropy bonus is the WRONG tool — it pushes toward all
options equally likely, but you do NOT want halt and 44x5 launches equally likely.
- FIX: replace the entropy bonus with a **KL penalty for divergence from an initial prior**, on the
  launch/halt and fraction dims. `halt_init_prob=0.9`, `fraction_init_ratio=1:1:1:1:10` (favor full-send).
  "This resulted in a huge immediate improvement."
- Reduce this coefficient later in training -> significant further improvement (even 0 worked in some cases
  but generalized slightly worse to the ladder).
- TAKEAWAY FOR US: regularize the policy toward a SENSIBLE PRIOR (mostly-halt, mostly-full-send), not toward
  uniform. Our entropy collapse / tiny-fleet issues are exactly what this targets.

## Fix #2 — action-space explosion: autoregressive "micro-steps"
Instead of one giant joint action (our MultiDiscrete([1921]*5) that drowned in no-op), build the turn
AUTOREGRESSIVELY, each micro-step = one RL trajectory step:
1. Decide globally: launch or HALT.
2. If launch: sample origin + send-fraction from a joint distribution.
3. Compute reachable targets & ETAs for that origin/fraction.
4. Sample a target (+ an 'abort' option).
5. Add the new fleet to the observation AS IF already in play (abort -> mask that origin-fraction for the
   rest of the turn).
6. Repeat up to 16x until halt.
Prunes candidates from 44*43*5 = 9,460 per turn down to ~43 per micro-step. Key trick: "making the next
decision after committing to a launch is strategically the same as if the new fleet were already in play."

NOT A CONTRADICTION with "candidate filtering caps the ceiling" ([[rl_postmortem_and_fix_plan]] Concept 5):
Ender's 43 = ALL legal targets for the chosen origin. The 9460->43 reduction is FACTORIZATION (the LEARNER
picks origin+fraction first, then keeps every reachable target), NOT a heuristic quality top-K. Nothing good
is discarded. Our old top-192 was reduced by v131's QUALITY score (removes moves by a guess = ceiling).
Rule: narrow candidates by the learner's own choices + legality (fine); never by an external quality
heuristic (ceiling). Reward note: Ender's RL is per-MICRO-STEP (trajectory) but the reward is still TERMINAL
(0/1/2) — do not confuse per-step trajectory (fine) with per-step reward (dense = reward-shaping trap).
RL is done entirely in micro-steps; the game turn index is part of the observation.
Result: Ender learned to launch MULTIPLE fleets from the same origin, to great effect — expressive action
space, cheaply factorized.

## Fix #3 — reward & stalling (still terminal, positive-only)
Reward: **0 loss / 1 draw / 2 win**. AVOIDED negative rewards: with gamma<1, negative rewards encourage
"delaying losing" (stalling) which wastes throughput. Tried gamma==1 -> Isaiah-style do-nothing-once-ahead
(no decisive victories) AND does nothing when behind -> most samples are two idle seats wasting throughput;
also worse at holding a lead vs an opponent who fights back. So: gamma=0.998, positive-only terminal reward.
Still simple, still terminal, still winner-take-all — just shifted to dodge the gamma<1 delay-loss trap.

## Cheap high-value ideas
- **Future features**: projected future garrisons/owners "assuming ceasefire" fed to the target MLP AND
  planet tokens. Fixed launching-too-early/late (policy can ABORT if projected outcome is bad) and let the
  value head respond confidently to good/bad launches in training. (Same family as simjeg roll-forward /
  Felix arrival-calendar.) 24 bins.
- **Incoming fleets**: resolve inter-fleet combat FIRST, encode NET effect per planet as 24 bins (planets
  don't care which origin / how many fleets — only the net). 4p adds 24 one-hot survivor-owner bins.
- **4p early-game bias fix**: 4p games run to 500 turns -> late-game bias. Truncate HALF the envs at 50
  turns (with value bootstrap) so there's always early-game data.
- **League**: self-play vs past + a few strong prior checkpoints, prioritized by recent winrate (harder
  played more). ELO from checkpoint games = free "am I still improving?" signal. Esp. important for 4p.
- Position encoded via **2D RoPE**. Observations normalized to p0 perspective.

## Architecture / hyperparameters
- Transformer encoder, **4 layers, d_model 192**, one CLS + one token per planet. "Larger models learn
  slower, no meaningful benefit" (maybe more compute would change this). Value/launch-halt/origin-fraction/
  abort heads = linear from planet hidden states. Target scorer = MLP(483,192,96,1) per candidate target
  (consumes target hidden state + launch-candidate features; NOT the origin hidden state).
- PPO + GAE, JAX env. lr 3e-5 (low), periodically reduced; ent/KL coef 0.01, periodically reduced;
  gamma 0.998; lambda 0.95; clip 0.2; PPO epochs 2; minibatch 2048; max grad norm 0.5;
  parallel envs 2048 (2p) / 384 (4p); rollout 256 (2p) / 512 micro-steps (4p). Separate 2p and 4p models.
- Env optimization: don't simulate fleet movement — precompute paths at launch, keep incoming bins per
  planet, shift left each turn.
- Test-time search (sampled-action search + launch/halt search) using a small distilled opponent policy;
  competition-harness launch geometry via expanding-circle tangency + sextic roots (closed form).

## Direct action items for us
1. READ THE CODE (github.com/sinking-point/ender) — it's the closest match to our constraints.
2. Adopt KL-to-prior instead of entropy bonus (halt~0.9, favor full-send) to kill tiny-fleet spam / entropy
   collapse.
3. Adopt micro-step autoregressive action construction to escape our action-space blow-up.
4. Keep reward terminal + simple (0/1/2 or -1/+1); gamma ~0.995-0.998; do NOT shape per-turn.
5. Target ~1B env steps on the 3080/4090 (days, ~$50-170) — NOT 200K steps on an M4.
6. Add future/ceasefire-projection features + league checkpoints + 4p early-truncation.