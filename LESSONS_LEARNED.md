
https://www.kaggle.com/competitions/orbit-wars/writeups/2nd-place-solution-for-orbit-wars# Orbit Wars — Lessons Learned

A retrospective written after the competition closed, comparing our effort against the
"Distributed Orbit Wars" solution by Cleway (final rank **110 / ~5000** — a strong silver,
*not* the winner).
The winner's writeup is still to come; notes on what even this silver solution may be missing
are at the end.

The headline finding is the encouraging one:
**our RL design was right.**
We lost on engineering throughput and observation design, not on ideas.

---

## 1. What was identical between us and the silver solution

These were never the problem, so they are not worth revisiting next time.

**Reward.**
Both used a sparse terminal signal: `+1` to the ship-count leader at game end, `-1` to everyone else.
The silver repo *built* a full dense-shaping system (capture bonuses, planet-lead-at-step-30, per-step relative ship count) and shipped it all **disabled by default** (`orbit_constants` / CLI all `0.0`).
We reached the same conclusion independently.
Sparse reward + self-play was correct; obsessing over reward shaping would have been a dead end.

**PPO algorithm.**
Theirs: γ0.995, λ0.95, clip0.2, 3 epochs.
Ours: γ0.99, λ0.95, clip0.1, 1 epoch with a `target_kl` early stop.
Both are reasonable.
We had even internalised the subtle bit — that extra PPO epochs blow up `clip_frac` on a large factorised action space — which is exactly why we ran 1 epoch.

**Policy architecture.**
Both used an entity transformer over per-planet / per-fleet tokens with factorised action heads (launch / target / ship-amount).
Ours was D=128, 3 layers, 4 heads, ~600K params.
Theirs was the same family plus a "target-plan" head (a soft global target-importance vector appended to each source token).
Same idea; theirs slightly richer.

---

## 2. Where we actually lost — all engineering, not RL

### 2a. Throughput → sample budget (this is ~80% of the gap)

| | Ours (`orbit_wars_jax`) | Silver |
|---|---|---|
| Parallel envs | **`N_ENVS = 16`** | **4,096 / GPU (up to 16,384)** |
| Samples per PPO update | 16 × 128 = **2,048** | ~**2,100,000** |
| Env implementation | JAX per-step interpreter, `vmap`'d | hand-written CUDA kernel, parity-tested |

With a sparse ±1 reward, 2,048 transitions per update contain almost no win/loss gradient.
A from-scratch policy cannot climb past a decent hand-coded heuristic on that diet — which is exactly what happened to us: the trained net never overtook the heuristics, so the final submission effectively ran heuristics.
The tell is already in our own `train.py`: a comment that **explained variance (`ev`) stays near 0**.
That is the critic failing to learn, which is the expected symptom of too little signal per update, not a bug to chase.

**The important nuance about "the CUDA was just faster":**
it was, but the deeper reason is **memory per environment**, not raw speed — and this is the part we learned the hard way.
We could not simply raise `N_ENVS`: the GPU fell over (OOM) well before we reached a useful env count.

Why JAX fell over where their CUDA did not:
our env detects fleet/planet collisions with a **dense matrix**, built every step (`env.py`):

```python
hits = jax.vmap(fleet_vs_all_planets)(...)   # [MAX_FLEETS, MAX_PLANETS] = [512, 64]
```

`vmap`'d over environments, that is a `[N_ENVS, 512, 64]` tensor, and `swept_pair_hit` materialises ~a dozen intermediates of that shape.
At `N_ENVS = 2048` each one is `2048 × 512 × 64 × 4 ≈ 270 MB`, several alive simultaneously → multiple GB *before* the rollout buffer is even stored.
Memory scales as **O(N_ENVS × MAX_FLEETS × MAX_PLANETS)**, and `MAX_FLEETS = 512` makes the constant brutal.

A hand-written CUDA kernel loops per-fleet in registers and **never materialises that matrix**.
Its memory is **O(N_ENVS × state size)**, so it scales to 16k envs on the same card.
JAX/XLA's pure-functional model gives you far fewer escape hatches here: no in-place mutation, no per-thread loops, and (as we had it) fp32 rollout storage with no microbatching or activation checkpointing — all knobs the silver repo used explicitly (`fp16` obs, `--update-microbatch-size`, `--activation-checkpointing`).

So the corrected lesson is **not** "saturate the tool you have" — we tried, and it broke.
It is: **the env's memory profile, not the training language, set the ceiling.**
Scaling in pure JAX would have meant real surgery — replace the dense `[F, P]` collision matrix with a sparse / `segment_sum` formulation, shrink `MAX_FLEETS`, store rollouts in fp16, and microbatch the PPO update.
That is a comparable amount of work to what the CUDA rewrite cost them — which is exactly why a custom kernel was a legitimate (arguably correct) call, not over-engineering.
The mistake was not *choosing JAX*; it was not recognising early that a dense-collision env caps env-count by memory, and that sparse-reward self-play needs the env-count more than almost anything else.

### 2b. The forecast ledger (observation engineering)

The silver solution feeds each planet token a **16-step look-ahead**, computed only from currently-visible state (no hidden-future cheating):
friendly / enemy ships already inbound per future step, first-enemy-arrival ETA, projected capture amounts.
One of the ship-amount action bins is literally "exactly enough to capture this target given the forecast."

This converts a hard long-horizon credit-assignment problem into a nearly-shallow one.
The network reads fleet trajectories instead of having to imagine them from sparse reward.

Our observation was current state only — planet (12 features) + fleet (9 features) tokens.
We made the net learn all temporal reasoning implicitly.
With a tiny sample budget on top, that was never going to converge.

A JAX port of this ledger is in `orbit_wars_jax/forecast.py` (reference only, not wired in).
It re-uses the env's exact fleet-speed, straight-line travel, planet rotation, and swept-collision dynamics.
Wiring it into `obs.py` would be additive — gather it with the same `planet_slots` already used for planet tokens:

```python
# in encode_obs(), after planet_slots is computed:
from forecast import forecast_ledger, HORIZON, LEDGER_FEAT
ledger = forecast_ledger(state, player_id, HORIZON)      # [MAX_PLANETS, H, F]
planet_ledger = ledger[safe_idx]                          # align to tokens
planet_ledger = jnp.where(planet_mask[:, None, None], planet_ledger, 0.0)
# return it alongside planet_tokens; model encodes it (small MLP) and
# concatenates to each planet embedding before the transformer.
```

### 2c. Self-play league quality

Ours: a 5-snapshot pool, **uniform** random opponent sampling, new snapshot every 500 iters.
Theirs: a 16-deep pool with **recency-weighted** sampling `P(age=k) ∝ exp(-k/τ)`, plus dedicated **adversary** nets trained to exploit the current main (`reward = -main_reward`), then folded back in as weighted opponents.

A uniform pool of 5 forgets old strategies and is prone to non-transitive cycling (rock-paper-scissors).
The recency weighting + explicit exploit-finders is what keeps a self-play ladder climbing instead of going in circles.
This mattered less than throughput, but it is the next thing after.

---

## 3. Smaller things worth stealing

- **Asymmetric critic.** Their transformer gives the *value head* extra aggregated features the actor does not see (`critic_feature_version = 1`). Cheap way to stabilise value learning under sparse reward.
- **Forecast-capture ship bin.** Making "send exactly enough to capture" a first-class action choice is a strong inductive bias for the action space, not just the observation.
- **Env parity tests.** They have a pure-Python reference and assert the CUDA kernel matches it. This is what let them rewrite the env aggressively without silently breaking the rules.
- **Gradient hygiene.** They `nan_to_num` non-finite gradients so a 10,000-update run does not die at hour 9. Small, boring, decisive.

---

## 4. If we did it again — ranked by leverage

1. **Profile env *memory per environment* on day one**, not just steps/sec. The question that mattered was "how many envs fit before OOM," and the answer was set by the dense `[MAX_FLEETS, MAX_PLANETS]` collision matrix. Fix that first: sparse / `segment_sum` collisions, smaller `MAX_FLEETS`, fp16 rollouts, microbatched PPO updates — *or* accept that a custom CUDA kernel is the cleaner path and budget for it early. Either way, env-count is the master variable for sparse-reward self-play.
2. **Add the forecast ledger to the observation.** Highest design-to-payoff ratio, and independent of the memory fight. `forecast.py` is the starting point.
3. **Upgrade self-play** to recency-weighted sampling + a deeper pool; add adversary exploit-finders only after the above two are paying off.
4. Keep the reward sparse. Keep the entity transformer. Keep 1 PPO epoch + KL early stop. Those were already right.

The meta-lesson: **we had the ML judgement and lost on systems — specifically on memory.**
For sparse-reward self-play, env-count is the variable that silently decides whether everything else gets a chance to work, and env-count was capped not by our willingness to scale but by an env whose memory grew as `O(N_ENVS × fleets × planets)`.
Next time, the *first* design question is "what is the memory cost of one environment step, and does it let me run thousands in parallel?" — answered before any policy or reward work begins.

---

## 5. What even the silver solution might be missing (watch the winner's writeup)

110th place is strong but not optimal.
Plausible levers the eventual gold solution may show that this silver one does not:

- **Prioritised fictitious self-play (PFSP):** weight opponents by *win-rate against the current policy*, not just recency — focus training on opponents that actually beat you.
- **Search at inference:** a short MCTS / rollout-and-pick on top of the learned policy+value (AlphaZero-style) often adds a lot in deterministic-ish games like this.
- **Population-based training:** several policies with different hyperparameters / styles co-evolving, periodically copying winners.
- **Bigger nets + longer schedules** once throughput is solved — the silver net is small (~1–2 layers) likely because it was compute-bound, not because small was best.
- **Reward normalisation / value rescaling** for more stable critic learning at scale.

Read the winner's writeup against this list — the deltas will tell us what the *next* tier of ideas actually was.
