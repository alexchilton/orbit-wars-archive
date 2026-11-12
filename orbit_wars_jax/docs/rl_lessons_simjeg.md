# RL Lessons — Simon Jégou / "simjeg" (2nd place, Orbit Wars)
Source: Kaggle writeup "2nd Place Solution for Orbit Wars" + repo (github.com/SimJeg/orbit-wars)

NOTE ON NAMING: This is the actual FINAL 2nd place. Our older
[[rl_lessons_lin_myat_ko]] doc is labeled "2nd place" but was likely a
mid-competition snapshot; simjeg credits @lightmk (probably Lin Myat Ko) as an
inspiration, so they are different people. Reconcile the labels if needed.

## The headline
Single net, **ModernBERT on 1D-CNN embeddings, 4.3M params**, both 2p AND 4p, self-play from
scratch for **10B steps** on 8×H100. ~47× smaller than IsaiahP's 200M 1st-place model.
See [[rl_lessons_isaiah_pressman]] for the scale-maximalist counterpoint — they reached the same
conclusions from opposite ends.

## The progression (his AlphaGo -> AlphaZero arc)
Heuristic/search agents (top 50) -> imitation learning / behavioral cloning (top 10) ->
RL finetune of the IL model (top 5) -> **RL from scratch in the last 5 days** (final).
Training from scratch quickly beat the IL-initialized models. His regret: didn't try scratch earlier —
it would have freed him to explore wider architectures/action spaces.
First-ever simulation competition; earned Grandmaster.

## The big idea: brutally simple action space
Per body (planet or comet), only **two actions: no-op OR all-in** (launch all ships) toward a
**short-distance target (ETA < 20)**. No fraction buckets. Game insight from watching top replays:
"the two most common actions are do-nothing and send-everything, launches are usually short-distance."

## Architecture (3 modules)
- **1D-CNN encoder (290K params):** 4 residual blocks [Conv1D(k=5) -> GELU -> Residual -> LayerNorm],
  GlobalAveragePooling1D, Linear 128 -> 256. One embedding per body. (Inspired by cpmpml/pdnartreb writeup.)
- **ModernBERT transformer (3.9M):** Ettin **XXS** config — 7 layers, 4 heads, d=256. Modifications:
  no token-embedding table (CNN embeddings are the input), **no positional encoding** (geometry already
  in the CNN embeddings), **global-attention only** (max input N=44 = 40 planets + 4 comets).
- **2N+1 heads (130K):** per-body launch head (linear on hidden states) + target head (attention over
  other bodies' hidden states) + one global value head (on avg of last hidden states) for PPO.

## The feature trick worth stealing
10 features per (body, timestep): timestep, production, radial+angular coord, player/neutral/opp1-3
ship counts, and `ships_for_capture` (extra ships needed to own the body after combat).
Then build a time series over t..t+T (**T=19**) by **rolling the env forward with no new actions**.
This explicitly encodes geometry and **implicitly captures all in-flight fleets' future impact**
(e.g. player_ships going positive->0 mid-series signals an incoming capture).
4 comet slots always reserved. Weakness he admits: model is blind to fleets arriving beyond T.

## How he beat the no-op collapse (the failure that killed OUR IL/RL)
This is the detail that matters most for us — see [[rl_postmortem_and_fix_plan]].
- **Per-planet Bernoulli launch head.** His repo `ow_runtime/model.py` outputs
  `PlanetTransformerOutput(launch_logits, target_logits)` — `launch_logits` is one independent yes/no
  per planet, NOT a single categorical over {N candidates + noop}. You predict N independent binaries,
  so "always no-op" is no longer the loss-minimising answer. No "when to stop" token needed.
- **IL: upweighted the launch class 5×** (BCE launch weight 5.0 vs target CE weight 1.0).
- **Measured launch Average Precision (~82-84)**, not accuracy — accuracy rewards "always no-op" ~95%.
- **RL anti-collapse = exploration/init levers, never reward-hacking**: (1) shift launch logits up on
  init so it *starts* launching, (2) 5× entropy coef on the launch head so it keeps sampling launches
  until the honest win/lose reward selects the good ones, (3) truncate after 40 no-ops. He never
  rewarded "acting" and never hard-masked no-op to force launches.

## Masking + inference
- launch_mask (own bodies only); target_mask[i,j] valid only if i can reach j in < T with an all-in.
- Features/masks/action conversion in **Rust**; NN inference in **Jax**.
- **Test-time augmentation**: average 4 rotational views (2p) / 8 views (4 rotations x 4 opponent
  permutations) for 3-4p. Worth ~+1-2 pts launch AP / target acc in his IL ablations.
- Launch fires when prob > 53% (sub1) / 56% (sub2); forced for owned comets leaving the board next step.

## Imitation learning (dropped from final, but shaped everything)
BCE launch head (weight 5.0) + CCE target head (weight 1.0), PyTorch Lightning. 5M samples (54% 2p /
46% 4p) filtered from 189K shared episodes: keep players with score>1500 (or who beat one), all-in only.
IL got him to top 10 and — more importantly — narrowed the feature set, architecture, and action space.
Things he tried in IL and dropped: fraction head (25/50/75/100%), higher T (30/50), other CNN/transformer
sizes, target-mask-as-attention-bias, various data filters.

## RL specifics (contrast these with IsaiahP)
- **PPO via PufferLib** (chosen for speed). Ported features/masks Rust->C, model Torch->CUDA via
  Codex/GPT-5.5. 8×H100, **~40K SPS**.
- **gamma = 0.995** (NOT 1.0) + **reward shaping that fixes stalling**: -1 lose, **+0.5 win after the
  500-step limit, +1 win before**. This is exactly the anti-stall fix IsaiahP said he *should* have added.
- "Agent never launches" cures: shift launch logits up on init from IL checkpoint; **5× higher entropy
  coef on launch head vs target head**; stop rollout after 40 consecutive no-ops (he later doubts this
  given pressman's replays).
- 4p mutual-neutralization cure: reward shaping above + a pool of frozen checkpoints for 2 of 4 seats
  (dropped for the final run). Poorly ablated — some tricks may be useless.
- Final: 10B steps, 1M steps/epoch, 3× 24h stages (3B / 3.5B / 3.5B) with LR 1e-3 -> 3e-4 -> 1e-4.
- Key config: four_player_prob=0.4, horizon=128, gae_lambda=0.97, min_lr_ratio=0.01 cosine,
  50M-step warmup (stages 2-3), minibatch 4096, clip_coef=0.2, launch_ent_coef=0.01,
  target_ent_coef=0.002, vf_coef=0.5, total_agents=1024.

## Evaluation (no CV in a sim comp)
Fast **local Rust arena**: run 1v1/2v2 matches or a local leaderboard using **OpenSkill** to mimic
Kaggle's (unknown) matchmaking. 1v1 and 2v2 between checkpoints correlated well with LB score.

## What he'd try next
Separate 2p/4p models; richer action space (fraction buckets); longer horizon than T=20;
**PairFormer** (AlphaFold) to explicitly encode per-pair (i,j) actions as a time series of
[t, min ships, max ships] sendable from i to j. After reading IsaiahP: **scale the transformer** and
replace the 1D-CNN with a plain linear projection on flattened features.

## Takeaways for our project
- His reward shaping (+0.5 for a late/timeout win, gamma=0.995) is the concrete, cheap fix for the
  stalling problem IsaiahP hit with gamma=1.0. Directly usable.
- The "no-op vs all-in, ETA<20" action space is a strong prior — collapsing the action space made a
  4.3M model competitive with a 200M one.
- The rolled-forward time-series feature (run env T steps with no action) is an elegant way to encode
  in-flight fleet dynamics without hand-engineering fleet features.
- OpenSkill local arena for checkpoint eval + TTA at inference are both low-cost wins.
- IL -> RL-finetune -> RL-from-scratch is a viable ladder, but from-scratch won: don't over-invest in IL.