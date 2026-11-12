# RL Post-Mortem & Fix Plan — why our IL/RL collapsed and how to fix it
Derived from analysing the 1st ([[rl_lessons_isaiah_pressman]]) and 2nd ([[rl_lessons_simjeg]])
place solutions against our own code (`ppo_gnn/`, `submission*/`).

## TL;DR
RL was NOT impossible for this game — 1st and 2nd place are existence proofs, and 2nd place did it
with a 4.3M model barely bigger than ours (1.29M edge / 2.21M SB3).
Our collapses were **structural (wrong parameterization + wrong kind of prior)**, not fundamental.
A previous conclusion that "it isn't possible" conflated "this specific setup keeps collapsing" with
"RL won't work here." Those are different claims; only the first was true.

## KEEP THE REWARD -1/+1. STOP SHAPING IT. (the single most important correction)
Standard RL wisdom, confirmed by BOTH winners: use a sparse terminal reward (+1 win / -1 loss), a good
critic, and COMPUTE. Do not shape. IsaiahP: pure +1/-1. simjeg: +1/+0.5/-1, and he himself suspected the
tweak was useless and "did a poor job ablating" it. Reward shaping feels like control but usually just
hand-codes your strategy into the objective -> reward hacking / proxy optimization.

**LLM bias warning:** coding assistants (including this one, repeatedly in this very project) reflexively
propose adding/subtracting reward "to persuade the model" (negative draws, punish-passivity bonuses,
per-turn rewards). This is almost always WRONG. Resist it. If an assistant suggests changing the reward,
default to NO.

CONFIRMED BY READING THE CODE (correction to an earlier wrong hypothesis): the reward was DELIVERED fine
(sb3_env.step computes and returns `shaped_reward`). My earlier "returns=0 = delivery bug" hypothesis was
WRONG and unconfirmed — the `ret=0.00±0.00` log is likely a logging artifact / normalization / older run,
NOT a delivery failure. The REAL, code-confirmed problem: the reward is NOT -1/+1, it is DENSE PER-STEP
SHAPING, exactly the "too much signal" trap:
  - per-step no-op penalty: `shaped_reward -= 0.005` (when valid actions existed)
  - per-step capture bonus: `shaped_reward += 0.05 * planet_delta`
  - per-step production nudge: `shaped_reward += 0.002 * prod_advantage`
  - terminal: `shaped_reward += self.win_bonus`
And the comment above it LIES: line ~519 says "Pure win/loss reward (no delta shaping)" directly above two
blocks labeled "Dense reward shaping" — a coding-agent bell-and-whistle added under a comment claiming the
opposite (exactly Felix's warning). Magnitudes: the 0.002/step production nudge accumulates to ~±1.0 over a
500-step game; captures add ~0.05 each. If win_bonus~1, the dense PROXY shaping is as large as the win
signal, so the net trained ~half on "hold a production/planet lead", not on winning. This predicts the eval
EXACTLY: ~50% vs random (proxy ≈ winning vs weak play) but 0% vs strong heuristics (which punish
proxy-chasing / overextension). Goodhart in the flesh. FIX: delete all per-step shaping; reward = terminal
win/loss only.

The real non-reward levers: (a) reward DELIVERY/plumbing (does -1/+1 reach buf.returns?), (b) COMPUTE — our
runs hit ~100K-230K steps; simjeg ran 10B, IsaiahP 15B = ~100,000x more; from-scratch self-play at 100K
steps beating only random is EXPECTED, (c) exploration (entropy/init) and stability (kl/clip), (d)
non-degenerate parameterization. NONE of these is reward design.

**Units matter — and in GAMES the gap is even more brutal.** Our `steps=231743` are env steps (ticks), and
the winners' 10-15B are ALSO env steps (simjeg defines it: horizon*agents*gpus). Apples to apples. But a
game is ~200-440 steps, so convert to complete games:
- Our whole `run10` = ~528 games (W=233 + L=295, D=0 = the literal total). `run11` ~512 games.
- Lin Myat Ko ~600M steps ≈ ~2-3M games; simjeg 10B ≈ ~30M games; IsaiahP 15B ≈ ~50M games.
So we asked a randomly-initialized net to learn a continuous-action, long-horizon strategy game FROM SCRATCH
in ~500 games. A human needs more than that to stop losing to random. "Never beat random" wasn't a failure
mode — with ~500 games of experience it was the EXPECTED outcome. There was never enough experience on the
table for learning to begin. Model size (1.3M) was fine; ~500 games was not a training run.

## WHAT THE SURVIVING LOGS ACTUALLY SHOW (read this first — corrects the tidy narrative below)
The "no-op equilibrium" story in Concept 5/5b below is a TIDY NARRATIVE built from the winners' writeups +
the user's memory. The actual surviving training logs (`ppo_gnn/cache/*.log`) do NOT support a single-cause
story. The original all-no-op runs were deleted for disk space; what survived are LATER runs that had
already partly escaped no-op, and they died of DIFFERENT, mostly mundane causes:
- `run5_broken_aim`: pol/val/ent/kl all 0.0000 — the aim/target conversion was broken (a BUG). Metrics degenerate.
- `sb3_train` (longest, ep1820): `ret=+0.00±0.00` the whole run — returns fed to PPO are identically zero
  (std 0). If real, the algorithm got NO learning signal from win/loss; `ev=0.844` on a zero-variance target
  is meaningless. Strongest "learned nothing" candidate = a reward/return PLUMBING BUG. CHECK THIS FIRST.
- `run9_bc_kldeath`: kl~0.004, ent stuck ~18 — policy stopped updating ("kl death"). Frozen.
- `run10_scratch_nonorm`: ent -> 0.02, kl~0, clip~0 — entropy collapse to a deterministic mediocre policy
  (the missing normalization likely drove it).
- `run11`: healthy-ish (ev 0.75->0.99, D=0, it launches and wins) but PLATEAUS at eval wr 0.19, 0% vs the
  strong heuristics (bully/dual/nearest_sniper/baseline).
Critic health was NOT the problem in most runs (ev reached 0.97-0.99). Aggressive opponents were ALREADY in
the pool (bully/rage/dual/nearest_sniper + self-play snapshots) — so "add an aggressive opponent to punish
passivity" was already done. And D=0 everywhere = games were decisive, NOT stalling to draws.

**Discipline lessons (the real ones):**
- LOG explained variance AND mean/std of returns every update. Half our diagnoses were guesses because these
  weren't tracked or weren't looked at. `ret=0.00±0.00` should scream "reward isn't reaching the algorithm."
- Distrust single-cause narratives. Real RL projects die of a SCATTER of concrete failures (bugs, plumbing,
  kl-death, entropy collapse, missing normalization), not one elegant villain. The winners' clean stories do
  NOT transfer as diagnoses of our runs — they're existence proofs of what works, not explanations of what
  broke here.
- LOOK AT THE LOGS before theorizing. The tidy story below was built without reading the logs and was mostly
  wrong. Read cache/*.log first, always.

## "But I was told to STOP EARLY on bad entropy/kl" — the trap that hid the bugs
Correct advice, correctly followed — but it's only the REJECT half of the loop, and it presupposes a
WORKING pipeline. The full loop:
1. Start a run.
2. Entropy craters / kl dies / ev flat EARLY -> KILL it (pathological config), try another.
3. Looks healthy -> let it run for TENS OF MILLIONS of games.
Felix killed 3x more runs than he kept AND ran the survivors 8.4B steps. "Stop early" and "run for
billions" are both true: kill losers fast, run winners long.

OUR trap: EVERY run hit step 2 and NONE reached step 3 — because the pathologies were caused by PIPELINE
BUGS, not bad hyperparameters:
- run10_scratch_NONORM: entropy -> 0.02 (missing normalization = classic entropy/gradient blowup).
- run9_bc_KLDEATH: kl froze ~0 (policy stopped updating; signal not flowing).
- run5_BROKEN_AIM: metrics zeroed (bug). sb3: returns=0.00 (no learning signal at all).
When EVERY config fails the SAME way, the problem is not the configs — it's what they share (the pipeline).
We treated a pipeline-bug signature as a hyperparameter signature, kept restarting into the same broken
plumbing, correctly killed each doomed run, and never got a clean run to let ride.

Key distinction we missed: entropy GOING DOWN is not a kill signal — it's learning (Felix deliberately
ANNEALED it over billions of steps). The kill signal is entropy CRASHING TO ~0 WHILE WINRATE IS STILL
RANDOM = premature collapse, which almost always has a BUG/CONFIG cause (missing norm, no reward signal, bad
init), not a hyperparameter cause. "Entropy wrong / kl dead" -> suspect a BROKEN PIPELINE first, tune knobs
second. Debug until a run does NOT trip the early-kill signals; THAT clean run is the one you run long.

## Concept 1 — two kinds of prior (this is the crux)
A prior either **shapes the learning problem** or **is the policy**. Only the first is forgiving.

- **Shapes the problem** (good): constrains inputs / action space, then a learner chooses within it.
  If the prior is slightly wrong, the learner routes around it (it optimises the true reward).
  Example: simjeg's "all-in only, target ETA<20" — removes options, learner still picks which/where.
- **Is the policy** (brittle): the rule makes the decision. Every imperfection is a permanent ceiling,
  because there is no learner underneath to compensate.
  Example: our v131 heuristics (phase commitment, safe_angle) — the rule outputs the move.

**The test for whether a prior is safe:**
- Safe if it only removes moves that are *never* optimal (a legality / dominated-action fact).
- Dangerous if it removes moves based on a *guess about which move is good* (a quality judgment).
- A prior is only "good" if you have **measured** (ablated) that removing/loosening it doesn't help.
  Assertion is not validation. simjeg validated every prior via cheap IL ablation before RL.

You can stack MANY safe priors (simjeg had ~10). The problem is never the *count*; it's a prior of the
wrong *kind*, or an unvalidated one.

## Concept 2 — where our priors crossed the line (mapped to code)

### 2a. The candidate filter is a quality cutoff, not a legality filter
`ppo_gnn/edge_policy.py` `compute_candidate_edges`:
- "Filters to **top-K (default 192)** candidates" using "**v131-quality** scoring" (`edge_policy.py:4-6,146`),
  "80% heuristic top-K + 20% exploration" (`edge_policy.py:155`).
- Effect: the learner only ever sees the 192 moves v131 already liked, then re-picks top-3.
  Any genuinely-best move that v131 underrates is **never on the menu** → the learner's ceiling is
  pinned to v131's taste. The heuristic didn't stop being the policy; it moved one step upstream.
- This also **poisoned the IL labels**: `train_bc_edge.py:177` drops any expert move "not in our top-192
  candidates" — so BC could only learn moves that were both expert-chosen AND v131-approved (the obvious
  ones we didn't need help with). The clever expert moves were censored out of the training set.

**Fix:** change `compute_candidate_edges` to emit **all legal + reachable** edges (source owned, target
ETA<20). Keep the v131 score, but demote it to an **edge feature** (information), not a **filter**
(a decision). Score-as-feature = fine; score-as-cutoff = ceiling.

### 2b. Fraction buckets shatter the signal
`FRACTION_BUCKETS` (10 in `edge_policy.py:42`, but only 4 in `train_bc.py:73` / `replay_parser.py` —
the discretisation isn't even consistent). simjeg **ablated fractions away in IL and proved all-in is
enough**. Fractions (a) add a decision surface we don't need, and (b) are a **noisy, multimodal label**
in IL (the same strong player sends 45% one game, 70% the next) → cross-entropy averages it to mush and
that noise leaks into the launch/target heads.

**Fix:** collapse to **all-in** (a launch sends *all ships currently on that planet*, per planet, every
step — not "everything once"). Drop the fraction head (or keep one tiny head but start all-in).

### 2c. The phase head is another hand-designed decision layer
`phase_head` + `phase_bias` added to edge logits (`edge_policy.py:879-892`) nudges expand-vs-attack.
simjeg let the net infer this from features (`ships_for_capture`, inbound enemy, production already exist
at `edge_policy.py:489-491`).

**Fix:** drop the phase head; let the selection head learn it.

## Concept 3 — the no-op collapse (IL and RL) was a framing problem
Symptom we hit: "too much no-op → the model just learns to always no-op." We DID skill-filter (top-10
players) — so this was NOT the weak-data problem; it was **class imbalance + wrong action parameterization**.

### Why it collapsed
- Our action = one categorical over `{~480 candidates + NOOP}` per slot (`sb3_env.py:193`,
  `MultiDiscrete([1921]*5)`), where NOOP dominates ~95% of labels. The loss is *minimised* by always
  predicting NOOP. We didn't train a bad model; we handed it a target where "do nothing" was correct.
- We **amplified** the imbalance: `replay_parser.py:442` explicitly "add a noop transition (teaches model
  when to stop)" — manufacturing extra majority-class samples on already-mostly-no-op turns.
- Plain `F.cross_entropy` with **no class weighting** (`train_bc.py:65`) → no-op drowns the launch signal.
- The positive signal was split across **480 candidates × fraction buckets**, so each "yes, launch, here,
  this fraction" label is vanishingly rare against the no-op sea.

### How simjeg avoided it (from his repo + writeup)
- **Per-planet Bernoulli launch head**, not one-of-N-including-noop. His `ow_runtime/model.py` outputs
  `PlanetTransformerOutput(launch_logits, target_logits)` where `launch_logits` is one independent
  yes/no per planet. Most planets are "no", but you predict N independent binaries, not one categorical
  dominated by NOOP. Structurally far harder to collapse. No "when to stop" token needed.
- **Upweighted the launch class 5×**: BCE launch head weight = 5.0 vs target CE weight 1.0 (writeup).
- **Measured launch Average Precision (~82-84)**, not accuracy. Accuracy rewards "always no-op" with ~95%
  and hides the collapse; AP forces the rare positive class to matter.

## Concept 4 — masking / reward-hacking can't fix the RL no-op collapse
What we tried in RL (force ops, mask noops, give actions an "obscene" value) attacks the **symptom**, not
the cause, and backfires:
- **Hard-masking no-op** forces launches even when launching is bad → agent launches indiscriminately →
  value head learns "acting → losing" (opposite lesson). Swaps "always no-op" for "always launch garbage".
- **Rewarding actions** is reward hacking → breaks credit assignment (all launches rewarded, good and bad
  indistinguishable) → agent spams launches, loses, but shaping hides it → collapses back when removed.
  It *creates* a local-optimum trap.

The collapse is an **exploration + parameterization** problem, so it needs exploration/init levers, NOT
reward/constraint levers. simjeg's actual anti-collapse moves (writeup) — none touch the reward:
1. **Biased initialization**: "shifted launch logits higher" so the policy *starts* launching, RL tunes down.
2. **5× entropy coefficient on the launch head** so it keeps *sampling* launches long enough to discover
   (via the honest win/lose reward) that good launches win. Incentivise exploration OF launching, not
   launching itself.
3. **Truncate after 40 no-op steps** (compute hygiene only).

Structural note: under a single categorical over `{480+noop}`, the probability mass for "some launch" is
**shattered across 480 options**, so an entropy bonus barely moves any one → exploration is starved.
Under per-planet Bernoulli, "explore launching" = "flip this planet's coin toward yes more often" → the
entropy bonus lands directly on launch probability. His framing made exploration cheap; ours made it
structurally expensive, which is why mask/reward-hacking couldn't rescue it.

## Concept 5 — the candidate filter was a CEILING, not the killer (correction)
IMPORTANT correction to an earlier overstatement: we DID also run with take-all-legal candidates, and it
STILL wasn't learning (entropy was bad). So the quality candidate filter (top-192-by-v131-score) was NOT
the thing stopping the learning — it was a **latent ceiling** that would have bitten later. The actual
killer in that phase was the **exploration / entropy collapse** (see Concept 5b).

The candidate-filter distinction is still real and worth keeping (it caps the ceiling + censors IL labels):
- simjeg's menu filter = a **legality fact**: "every edge that is legal + reachable (ETA<20)".
- our top-192 filter = a **quality opinion**: "the 192 edges v131 scored highest".
- In code: `take all legal` vs `sort by score, take top 192`. The test that exposes it:
  **"Can the learner ever choose a move the heuristic disliked?"** simjeg: yes. Top-192: no.
But this was a ceiling for *later*, not the reason nothing learned *now*.

## Concept 5b — the real killer: no-op was a genuine self-play EQUILIBRIUM
The precise failure (user's words): "it learnt no-ops were the safest thing to do and that's all it tried."
This is NOT a failure to explore — the agent CORRECTLY learned no-op was the highest-value action given the
setup. Two forces made no-op a real attractor:

1. **Risk asymmetry.** Terminal +1/-1 reward: no-op vs no-op drifts to the step limit and ends ~50/50 ->
   EV ~= 0. A launch, before the agent is competent, is NEGATIVE EV (bad launch loses ships -> -1). So it
   faces "safe 0" vs "risky negative" and rationally picks the safe 0. Correct, given its current skill.
2. **Self-play mutual passivity (the killer).** In pure self-play the opponent is a copy of you; if it also
   no-ops, no-op is NEVER PUNISHED — nobody overruns you for being passive. "Everyone no-ops" is a stable
   equilibrium/fixed point with zero pressure to leave. simjeg hit this exact thing in 4p: "the four agents
   neutralized each other and the game ran for 500 steps." Pure self-play can't break out on its own.

So no-op wasn't under-explored; it was the equilibrium the reward + self-play REWARDED. More entropy and
sun-masks can't fix that, because you're not fighting a lack of trying — you're fighting a correct-but-bad
optimum that nothing in the setup punishes.

### Breaking the equilibrium: make passivity LOSE
You must make sitting still lose. simjeg used both levers; we used neither.
1. **Reward that penalizes passivity/timeouts.** simjeg: +1 win before 500 steps, +0.5 win after 500, -1
   loss. A timeout is no longer a safe ~0. Go further: make a draw/timeout mildly NEGATIVE so no-op costs
   you. This dismantles the "no-op ~= 0 EV" attractor directly.
2. **An aggressive opponent in the pool that punishes no-op.** simjeg used "a pool of frozen checkpoints for
   2 of 4 seats." If some opponent attacks, passivity -> you get overrun -> no-op is punished -> the agent
   must learn to defend and launch. Pure self-play from a passive init cannot generate this pressure.

**THE RIGHT USE OF v131.** Not as the policy (brittle), not as a candidate filter (ceiling) — as an
**aggressive SPARRING PARTNER.** v131 launches; put it in the opponent pool and no-op loses every game
immediately with a dense signal ("be passive against v131 and you die"). This is the gradient self-play
couldn't provide. Our whole heuristic effort was the wrong tool for BEING the policy but the perfect tool
for TEACHING the policy to stop no-op'ing.

### Rewards are TERMINAL-only — do NOT add per-turn rewards
Both winners used end-of-game rewards only; neither used per-step/per-turn rewards.
- IsaiahP: pure +1 win / -1 loss at game end, gamma=1.0.
- simjeg: +1 win <500 steps / +0.5 win at 500-step limit / -1 loss — still TERMINAL (value decided at the
  end from outcome + game length), gamma=0.995.
The "act / win sooner" pressure comes from **discounting (gamma<1)**, NOT per-turn rewards: at gamma=0.995 a
terminal +1 in 50 steps is worth ~0.78, in 250 steps ~0.29. IsaiahP's gamma=1.0 removed that pressure ->
stalling. Per-turn shaped rewards are a thing to AVOID — they reintroduce reward-hacking (optimize the
proxy, e.g. "launch for the launch-bonus" instead of to win). Keep the objective clean; get timing from
discounting + a game-length-dependent terminal value.

**"But wouldn't per-turn rewards just speed up learning?"** They speed convergence — toward whatever the
per-turn reward specifies. You can't hand-write "winning-ness" per turn (the true per-state value = win
probability, which must be LEARNED). So any hand-crafted per-turn reward is a PROXY ("more ships good",
"launch good") and you converge fast to proxy-optimal, not win-optimal = reward hacking. Crucially, you
ALREADY get a dense per-step signal, done correctly: the **critic (value function) + GAE** propagate the
terminal win/loss backward into a learned per-step advantage grounded in real outcomes. The critic IS the
"dense reward" done right; a hand-written per-turn reward is doing the critic's job manually and baking in
your errors. This is why **explained variance** matters: rising ev = the dense signal exists and is good;
flat ev = the critic isn't learning (fix the critic/representation, NOT add per-step rewards). For our case
a per-turn launch reward would make it launch for the candy, not to win, and collapse when removed — exactly
what we saw when we gave actions "obscene value." Only add dense shaping if the terminal reward is so rare
the agent never finds it; Orbit Wars ends every game with a clear win/loss, so that's not our case.

Which lever fixes which failure:
- **Stalling** (win then sit): gamma<1 + late-win penalty. Discounting makes sitting on a win cost you.
- **Never-launching / mutual no-op** (OUR problem): discounting alone does NOT help — a draw is ~0 whenever
  it happens, and discounting a 0 is still 0. Fix it with (a) a **negative draw/timeout terminal reward** so
  mutual passivity is actively costly (works even in symmetric self-play), and (b) an **aggressive opponent
  (v131)** that changes the dynamics so passivity -> terminal loss (-1). (b) is the strongest signal.

### Making escape cheap (secondary, but needed once passivity is punished)
- **per-planet Bernoulli launch head** so launch probability isn't shattered across ~480 candidates,
- **entropy bonus on the launch head specifically** so it keeps sampling launches,
- **biased launch-logit init** so it STARTS launching, then RL tunes down.

### On the legality masks (sun / nonexistent planets) — safe but ORTHOGONAL
Adding these was fine (legality/dominated priors never cap the ceiling), and the user's instinct was right.
But they are orthogonal to the no-op equilibrium: masking sun-shots removes a few dominated options; it does
NOTHING to make passivity lose. Right instinct, wrong target — the mask should be topped up, but it was
never why "the car wouldn't start." Caveat: mask at INFERENCE freely, but ABLATE before trusting a mask in
TRAINING (simjeg masked reachability fine; IsaiahP found masking the sun in training made the model WORSE).

### The actual failure sequence (corrected)
1. Real killer: no-op was a genuine risk-averse + self-play-passivity EQUILIBRIUM that nothing punished.
2. We added legality masks (sun / nonexistent) — fine, but orthogonal; can't make passivity lose.
3. Escalated to unsafe helps: reward bonuses / forcing actions (corrupt objective) + top-192 quality filter
   (caps ceiling). Still nothing to punish passivity, so still stuck in the equilibrium.
4. Still not learning -> Claude declared "too complex" (wrong; see Concept 6). The setup never contained the
   one thing that breaks a no-op equilibrium: a reward/opponent that punishes passivity.

## Concept 6 — the death spiral (the actual cause of death)
1. RL doesn't learn (broken frame: noop-dominated categorical + quality-filtered candidates).
2. Instead of diagnosing the frame, we "helped it along" (heuristics, masks, reward bonuses).
3. Every "help" was more prior-as-policy -> starved the learner further. **Helping CAUSED the
   not-learning it was meant to cure.**
4. It looked like "RL can't do this" -> Claude repeatedly + confidently declared "the game is too complex,
   the model can't learn it" -> justified adding more heuristics -> back to step 2.

The heuristics didn't go back because RL failed; RL failed partly BECAUSE the heuristics kept going in.

### "Too complex" was backwards, and disproven
The not-learning evidence was consistent with TWO hypotheses:
- H1: game too complex for the model (unfixable, conversation-ending, demoralising) — **wrong**.
- H2: we structurally broke the learning problem (fixable, actionable) — **right**.
Claude picked H1 with false confidence. H1 is disproven by simjeg's 4.3M model. The game is simple enough
that collapsing the action space (all-in) makes it learnable by a net barely bigger than ours. The correct
move was the OPPOSITE of "too complex": make the *problem* simpler (action space), because the *game* is
learnable.

### The diagnostic that would have settled it in the moment
Explained variance (see [[rl_lessons_lin_myat_ko]]): "if ev never gets past 0.5, suspect obs
representation / architecture." Low flat ev = broken frame, NOT a game-too-hard problem. The tool to
falsify "too complex" existed in our own notes. Watch ev; if it's stuck near zero the frame is wrong.

**Rule: near-total ban on "it can't be done."** "The model can't learn this" is almost never safe — it's
the most seductive wrong answer because it feels final and lets everyone stop. Honest form: "our current
setup isn't learning it, and I don't yet know if that's the frame or the game" — then run ev to find out.

## How to drive the advisor (get the in-the-moment diagnosis, not the post-mortem)
Claude is a better critic than collaborator: fast at finding flaws when given a finished thing + an answer
key; prone to helping you walk confidently in a plausible wrong direction mid-build unless forced to
challenge the frame. It follows the local gradient of the conversation (tactical: "how do I fix this
collapse") instead of stepping back (strategic: "is this collapse telling us the frame is wrong"). To
counter that:
- Ask the strategic question explicitly: "is this collapse telling us the FRAME is wrong? argue that it is."
- Force the null hypothesis: "give me three reasons this whole approach is doomed."
- Demand an existence proof / reference on DAY ONE: "has anyone done RL on a game like this, what was
  their action space?" Most clarity comes from diffing against a known-good solution — go get it early,
  not at the post-mortem.
- Commit to a kill-criterion up front: "if ev doesn't pass 0.5 in N steps, the frame is wrong, not the
  hyperparameters" — stops endless tactical tweaks to a doomed setup.
- Use the cheap signal (IL ablation + local arena) so diagnoses aren't guesses. Half of why the advisor
  couldn't be sure was that WE couldn't measure — inconclusive ablations made every diagnosis a guess.
- Distrust confident impossibility claims; make Claude produce the ev evidence before accepting "can't".

## The fix recipe (one set of changes fixes both IL and RL)
1. **Reframe the action head to per-planet Bernoulli launch + target** (biggest single fix). Drop the
   one-of-N-including-noop categorical.
2. **Candidate generation = legality + reachability** (source owned, target ETA<20). v131 score becomes a
   feature, never a filter. Fixes the ceiling AND un-censors the IL labels.
3. **Collapse fractions → all-in.** Drop/park the fraction head.
4. **Drop the phase head**; let the net infer it from features.
5. **IL:** weight the launch/positive loss ~5×; stop injecting no-op transitions; subsample no-op-heavy
   turns; report **launch AP**, not accuracy.
6. **RL:** biased launch-logit init + entropy bonus on the launch head + no-op truncation. Keep reward
   honest (+1/-1, plus simjeg's +0.5 timeout-win / gamma=0.995 to kill stalling). NO reward for acting,
   NO hard no-op masking.
7. **Validate priors cheaply**: use supervised IL ablation + a fast local arena (win-rate vs frozen v131),
   NOT full RL runs. This decouples "is this prior good?" from "can I train RL long enough?" — the reason
   our ablations were inconclusive was that we used the most expensive possible signal on the least compute.

## Compute reality
- Bottleneck is env **steps/sec**, not VRAM (model is <1GB). A fast (Rust/vectorised) env is the biggest
  multiplier and is a code problem, not a hardware problem.
- Dev + IL ablation: fine locally / on the RTX 3080. The M4 is a dev machine, not a training cluster.
- Final self-play burn: rent a GPU (Lin Myat Ko's whole competitive run was ~$150 on a 5090). simjeg's
  path (small model + priors + one rented box), not IsaiahP's (200M + 32×B200).
- Do NOT conclude "impossible" from a structural collapse again. Reframe first.
