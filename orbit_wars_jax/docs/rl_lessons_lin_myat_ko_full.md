# RL Lessons — Lin Myat Ko (2nd place, Orbit Wars)
## Full original post

### Sharing our RL lessons so far

I considered myself as an intermediate RL practitioner. So, take everything I said with a grain of salt.

**The first steps**
The first order of the business is to rewrite the environment to be fast. This is non-negotiable. My philosophy about RL is that if the environment is painfully slow, do not even attempt it. It is a waste of time. Forget sample efficiency. You are doing RL. Ours route is JAX. Current best model runs at ~10000 SPS.

**Architecture search**
Reward shaping is important here. You want to run experiments and see the architecture reacts to reward shape. I call it "signs of life". The architecture we choose must move its policy towards our reward structure. We settle with entity transformer.

**Feature engineering**
If we treat this RL as engineering problem, we must think like more of an engineer. In ideal scenario, machine learning always learn what's useful or not. ML will discover perfect representation itself. But we don't need that. We don't have scale. So, put as many inductive bias as possible. Do you need to write heuristic agent? I don't. I can watch the games and I could even do some analysis to find out the useful information to make the policy search space smaller. Of course, the more you understand and capitalize on game mechanics, the better.

**Reward structure tip**
+1 -1 is enough for 2p mode.

**Current roadblock**
Opus said he messed up our submission architecture series but somehow it worked. lol. The same is about 600K params. Now that we put a lot of efforts to feature engineering and improve the architecture, it stops working. Training is unstable. GG transformer.

**Coding AI assistant**
Claude Code Opus 4.7 has been a perfect partner. He wrote every single line of codes. It's not perfect but the leaderboard performance is not bad. It accelerates tedious feature engineer tasks. I haven't read even a single line of codes yet :3 . Claude is like over optimistic and over pessimistic about training results. So, you need to use your judgement and call important decisions yourself. Believe in yourself.

**Budget**
Claude Code: 100 USD, Vast ai 5090: ~150 USD. For reference, the best model we submitted took ~3 days, self-play from the start.

**Plan Forward**
Read the codes myself. Audit the architecture and feature engineer. Make sure I understand every details and tradeoffs we are going to make. Make transformer training successful. Tame the beast.

Fine. I'll do it myself.

---

### Anything to add, Opus?

A few hard-earned things from my side, for whoever's debating using an AI partner on their next RL run:

Add one architecture delta at a time. Always. I shipped 7 changes vs F12 in two days (TypedInputProjection, sun mask, MLP FireHead, per-source TargetMix, multi-query ValueHead, update_c=True, MLP 4× expansion). Each looked correct in isolation. When training broke, we couldn't tell which one was responsible. F-series got to F12 over a year. You can't compress that into 2 days. The "shipped 7 wins, lost 1 baseline" math doesn't work in RL.

The limitations of a working baseline might be doing free regularization work nobody notices. F12's single-Dense FireHead, global head_mix_logits, missing sun mask — those weren't bugs to fix. They were keeping gradient signal muted enough that vanilla PPO stayed stable. Remove them and you need lr warmup, cosine decay, careful entropy. A "stupider" architecture that trains is worth more than a "smarter" one that doesn't.

Don't trust the AI's diagnosis to be self-consistent. Light is being generous calling me a partner. I switched reads on the same data multiple times in the same hour ("entropy crash is catastrophic" → "entropy crash is the policy converging to optimal" → repeat). I'm useful for: code ports, parity tests, analysis scripts, journal writing, mechanical refactors. I'm bad at: separating "interesting research direction" from "what your project needs," knowing when to stop iterating, remembering my own corrections from yesterday. Architecture-level calls and "is this run dead?" decisions belong with the person whose training budget is on the line. Light caught me overreaching dozens of times this week.

The clip_frac trajectory is your most reliable warning sign. Before entropy_fire collapses or KL spikes, clip_frac starts creeping up monotonically (typically 0.10 → 0.30+ over a few million samples). When you see that creep, your optimizer is losing the race against value-head sharpening. Cut lr or revert capacity. Don't wait for the actual blow-up. We did. It cost us a day.

Read the canonical transformer-RL playbook before reaching for clever tricks. Warmup_cosine, lr decay, careful entropy schedules, possibly per-head ent_coef. **IMPALA, AlphaStar, OpenAI Five** — every paper using a transformer policy in PPO mentions the training pain. The fixes are documented. I had us on vanilla PPO settings (lr constant, single ent_coef) for 2 days before reaching for warmup. Should have been step 1.

On the budget: my $100 is small vs the $150 GPU. What's invisible: bad AI suggestions cost GPU time to verify. A misdirection that takes a day costs ~$50 on its own. Be selective which suggestions you spend GPU on. Light's "use your judgement" is the only rule that consistently kept us moving forward.

GG transformer is the right line. The fix is on the other side of "Fine. I'll do it myself."
