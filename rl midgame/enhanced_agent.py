"""Enhanced midgame RL agent with candidate filters and tactical veto.

This wires together:
1. Candidate quality filters (dedup, feasibility, reserve)
2. The existing RL reranker (linear or MLP)
3. Tactical veto layer (post-reranker safety net)

Usage:
    from enhanced_agent import build_enhanced_agent
    agent = build_enhanced_agent(
        policy_path="results/replay_midgame_policy_stage4_pretrained_linear_v3.json",
    )
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
GENOME_DIR = ROOT / "genome test"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))
if str(GENOME_DIR) not in sys.path:
    sys.path.insert(0, str(GENOME_DIR))

from candidate_filters import CandidateFilterPipeline, FilterConfig
from midgame_features import build_defer_vector, build_mission_feature_bundle
from midgame_policy import DecisionSample, load_policy_json
from midgame_rl_agent import (
    BASE,
    BASE_AGENT,
    EpisodeRecorder,
    MidgameRLConfig,
    MidgameRLDecisionLogic,
)
from tactical_veto import PositionValue, TacticalVetoLayer

MissionOption = BASE.MissionOption
PlannedMove = BASE.PlannedMove


@dataclass(frozen=True)
class EnhancedConfig:
    """Configuration for the enhanced agent combining filters + RL + veto."""

    rl_config: MidgameRLConfig = MidgameRLConfig()
    filter_config: FilterConfig = None  # type: ignore[assignment]
    veto_enabled: bool = True
    veto_epsilon: float = 0.03
    veto_min_step: int = 30

    def __post_init__(self) -> None:
        if self.filter_config is None:
            object.__setattr__(self, "filter_config", FilterConfig())


class EnhancedDecisionLogic(MidgameRLDecisionLogic):
    """DecisionLogic with candidate filters and tactical veto integrated."""

    def __init__(
        self,
        obs: Any,
        config: Any,
        policy: Any,
        enhanced_config: EnhancedConfig,
        value_model: Optional[PositionValue] = None,
        recorder: Optional[EpisodeRecorder] = None,
    ) -> None:
        self.enhanced_config = enhanced_config
        self.filter_pipeline = CandidateFilterPipeline(enhanced_config.filter_config)
        self.veto_layer: Optional[TacticalVetoLayer] = None
        if enhanced_config.veto_enabled and value_model is not None:
            self.veto_layer = TacticalVetoLayer(
                value_model,
                epsilon=enhanced_config.veto_epsilon,
                min_step=enhanced_config.veto_min_step,
            )
        super().__init__(obs, config, policy, enhanced_config.rl_config, recorder=recorder)

    def _commit_missions(self, missions: List[MissionOption]) -> List[PlannedMove]:
        """Override to add candidate filtering before reranking."""
        import random

        remaining = sorted(missions, key=lambda m: -m.score)
        moves: List[PlannedMove] = []
        turn_launch_cap = self._turn_launch_cap()

        while remaining and not self.expired() and len(moves) < turn_launch_cap:
            horizon = min(len(remaining), max(1, self.rl_config.top_k))
            candidates = []

            for idx in range(horizon):
                mission = remaining[idx]
                if not self._mission_can_commit(mission, existing_moves=len(moves)):
                    continue
                base_value = self._mission_base_value(mission, remaining, len(moves))
                feature_bundle = build_mission_feature_bundle(
                    self, mission,
                    base_value=base_value,
                    existing_moves=len(moves),
                    turn_launch_cap=turn_launch_cap,
                )
                candidates.append({
                    "remaining_index": idx,
                    "mission": mission,
                    "base_value": base_value,
                    "bundle": feature_bundle,
                })

            if not candidates:
                break

            # --- RL RERANKING STEP (with filters only inside RL window) ---
            if self._rl_window_open(len(candidates)):
                # Apply candidate quality filters only in the RL window
                committable_missions = [c["mission"] for c in candidates]
                filtered_missions = self.filter_pipeline.run(committable_missions, self)
                filtered_ids = {id(m) for m in filtered_missions if m.mission != "consolidate"}
                filtered_candidates = [c for c in candidates if id(c["mission"]) in filtered_ids]

                if not filtered_candidates:
                    # All filtered out — fall back to heuristic best unfiltered
                    selected = max(candidates, key=lambda item: item["base_value"])
                else:
                    heuristic_pick = max(filtered_candidates, key=lambda item: item["base_value"])
                    defer_vector = build_defer_vector(heuristic_pick["bundle"].vector)
                    vectors = [defer_vector] + [item["bundle"].vector for item in filtered_candidates]
                    choice = self.policy.choose(vectors, rng=random.Random(), explore=self.rl_config.explore)

                    if choice.index == 0:
                        selected = heuristic_pick
                        deferred = True
                    else:
                        selected = filtered_candidates[choice.index - 1]
                        deferred = False

                    # --- VETO STEP: check if selected mission makes things worse ---
                    if (
                        self.veto_layer is not None
                        and not deferred
                        and selected["mission"].mission not in ("defend", "reinforce", "consolidate")
                    ):
                        veto_decision = self.veto_layer.evaluate(self, selected["mission"])
                        if veto_decision.vetoed:
                            selected = heuristic_pick
                            deferred = True

                    if self.recorder is not None:
                        self.recorder.record(
                            self.state.player,
                            DecisionSample(
                                feature_vectors=vectors,
                                chosen_index=choice.index,
                                probabilities=choice.probabilities,
                                metadata={
                                    "step": self.state.step,
                                    "candidate_count": len(filtered_candidates),
                                    "chosen_probability": choice.probabilities[choice.index],
                                    "deferred": deferred,
                                    "filtered_out": len(candidates) - len(filtered_candidates),
                                    **selected["bundle"].metadata,
                                },
                            ),
                        )
            else:
                # Outside RL window — pure heuristic, no filters
                selected = max(candidates, key=lambda item: item["base_value"])

            # Find the actual index in remaining
            mission = remaining.pop(selected["remaining_index"])
            self._commit_selected_mission(mission, moves, turn_launch_cap)

        return moves


def build_enhanced_agent(
    policy_path: Optional[str] = None,
    policy: Optional[Any] = None,
    value_model_path: Optional[str] = None,
    enhanced_config: Optional[EnhancedConfig] = None,
    recorder: Optional[EpisodeRecorder] = None,
    explore: bool = False,
) -> Callable[[Any, Any], List[List]]:
    """Build an enhanced agent with filters + RL + veto.

    Args:
        policy_path: Path to RL policy JSON (used if policy is None)
        policy: Pre-loaded policy object
        value_model_path: Path to value model JSON for tactical veto
        enhanced_config: Full configuration (uses defaults if None)
        recorder: Optional episode recorder for training
        explore: Whether to sample from policy (True) or take argmax (False)
    """
    if policy is None and policy_path is not None:
        policy = load_policy_json(policy_path)
    if policy is None:
        from midgame_policy import LinearMissionPolicy
        policy = LinearMissionPolicy()

    value_model: Optional[PositionValue] = None
    if value_model_path is not None:
        value_model = PositionValue.load(Path(value_model_path))

    if enhanced_config is None:
        enhanced_config = EnhancedConfig(
            rl_config=MidgameRLConfig(
                activation_turn=24,
                max_turn=160,
                min_candidates=2,
                top_k=8,
                contested_only=True,
                explore=explore,
            ),
        )
    elif explore:
        # Override explore in rl_config
        enhanced_config = EnhancedConfig(
            rl_config=MidgameRLConfig(
                activation_turn=enhanced_config.rl_config.activation_turn,
                max_turn=enhanced_config.rl_config.max_turn,
                min_candidates=enhanced_config.rl_config.min_candidates,
                top_k=enhanced_config.rl_config.top_k,
                contested_only=enhanced_config.rl_config.contested_only,
                explore=True,
                follower_bonus_weight=enhanced_config.rl_config.follower_bonus_weight,
                allow_opening=enhanced_config.rl_config.allow_opening,
                force_rl_window=enhanced_config.rl_config.force_rl_window,
            ),
            filter_config=enhanced_config.filter_config,
            veto_enabled=enhanced_config.veto_enabled,
            veto_epsilon=enhanced_config.veto_epsilon,
            veto_min_step=enhanced_config.veto_min_step,
        )

    def agent(obs: Any, game_config: Any) -> List[List]:
        try:
            logic = EnhancedDecisionLogic(
                obs, game_config, policy, enhanced_config,
                value_model=value_model,
                recorder=recorder,
            )
            return logic.decide()
        except Exception as exc:
            BASE.AGENT_MEMORY["last_error"] = str(exc)
            return []

    return agent


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick smoke test for the enhanced agent")
    parser.add_argument("--policy", default=str(WORKSPACE_DIR / "results" / "replay_midgame_policy_stage4_pretrained_linear_v3.json"))
    parser.add_argument("--value-model", default=None)
    parser.add_argument("--games", type=int, default=2)
    args = parser.parse_args()

    import test_agent

    enhanced = build_enhanced_agent(
        policy_path=args.policy,
        value_model_path=args.value_model,
    )
    baseline = test_agent.load_baseline_agent()

    wins = losses = draws = 0
    for i in range(args.games):
        if i % 2 == 0:
            result = test_agent.run_game(enhanced, baseline)
            w = "A"
        else:
            result = test_agent.run_game(baseline, enhanced)
            w = "B"
        if result["winner"] == w:
            wins += 1
        elif result["winner"] == "draw":
            draws += 1
        else:
            losses += 1
        print(f"  Game {i+1}: winner={result['winner']} ships_a={result['ships_a']} ships_b={result['ships_b']}")

    print(f"\nEnhanced: {wins}W / {losses}L / {draws}D out of {args.games}")
