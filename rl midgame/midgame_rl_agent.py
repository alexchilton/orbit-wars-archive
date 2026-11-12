from __future__ import annotations

import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

WORKSPACE_DIR = Path(__file__).resolve().parent
ROOT = WORKSPACE_DIR.parent
BASE_AGENT_PATH = ROOT / "snapshots" / "stage3_search_base.py"
_BASE_MODULE_NAME = "_orbit_wars_midgame_rl_base"


def _load_base_module() -> Any:
    if _BASE_MODULE_NAME in sys.modules:
        return sys.modules[_BASE_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, BASE_AGENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load base agent from {BASE_AGENT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_BASE_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_module()
BASE_AGENT = getattr(BASE, "agent", None) or getattr(BASE, "_base_agent_entrypoint", None)
if BASE_AGENT is None:
    raise RuntimeError("Base agent module does not expose an entrypoint")
MissionOption = BASE.MissionOption
PlannedMove = BASE.PlannedMove

from midgame_features import MissionFeatureBundle, build_defer_vector, build_mission_feature_bundle
from midgame_policy import DecisionSample, LinearMissionPolicy, load_policy_json


@dataclass(frozen=True)
class MidgameRLConfig:
    activation_turn: int = 24
    max_turn: int = 160
    min_candidates: int = 2
    top_k: int = 8
    contested_only: bool = True
    explore: bool = False
    follower_bonus_weight: float = 0.55
    allow_opening: bool = False
    force_rl_window: bool = False


class EpisodeRecorder:
    """Collect sampled midgame decisions so a trainer can update the policy afterward."""

    def __init__(self) -> None:
        self._last_step: Dict[int, int] = {}
        self._episodes: Dict[int, List[DecisionSample]] = {}

    def start_turn(self, player: int, step: int) -> None:
        if step <= 0 or step < self._last_step.get(player, -1):
            self._episodes[player] = []
        self._last_step[player] = step

    def record(self, player: int, sample: DecisionSample) -> None:
        self._episodes.setdefault(player, []).append(sample)

    def pop(self, player: int) -> List[DecisionSample]:
        return list(self._episodes.pop(player, []))


class MidgameRLDecisionLogic(BASE.DecisionLogic):
    def __init__(
        self,
        obs: Any,
        config: Any,
        policy: Any,
        rl_config: MidgameRLConfig,
        recorder: Optional[EpisodeRecorder] = None,
    ) -> None:
        self.policy = policy
        self.rl_config = rl_config
        self.recorder = recorder
        super().__init__(obs, config)
        if self.recorder is not None:
            self.recorder.start_turn(self.state.player, self.state.step)

    def _rl_window_open(self, candidate_count: int) -> bool:
        if candidate_count < self.rl_config.min_candidates:
            return False
        if self.state.step < self.rl_config.activation_turn or self.state.step > self.rl_config.max_turn:
            return False
        if self.rl_config.force_rl_window:
            return True
        if self.state.is_opening and not self.rl_config.allow_opening:
            return False
        if self.state.is_very_late:
            return False
        if not self.rl_config.contested_only:
            return True

        modes = getattr(self, "modes", {})
        if self.state.num_players <= 2:
            my_total = float(modes.get("my_total", 0.0))
            enemy_total = float(modes.get("enemy_total", 0.0))
            share = my_total / max(1.0, my_total + enemy_total)
            return 0.34 <= share <= 0.72 and not modes.get("is_finishing", False)

        my_rank = int(modes.get("my_rank", 1))
        return my_rank <= 2 and not modes.get("is_dominating", False) and not modes.get("is_cleanup", False)

    def _mission_base_value(
        self,
        mission: MissionOption,
        remaining: List[MissionOption],
        existing_moves: int,
    ) -> float:
        pending_sources = set(mission.source_ids)
        pending_targets = {mission.target_id} if self._mission_blocks_target(mission) else set()
        follower_bonus = 0.0
        follower_horizon = min(len(remaining), 10)
        for follower in remaining[:follower_horizon]:
            if follower is mission:
                continue
            if self._mission_can_commit(
                follower,
                extra_used_sources=pending_sources,
                extra_target_ids=pending_targets,
                existing_moves=existing_moves + len(mission.source_ids),
            ):
                follower_bonus = max(follower_bonus, follower.score)
        return mission.score + self.rl_config.follower_bonus_weight * follower_bonus

    def _commit_selected_mission(
        self,
        mission: MissionOption,
        moves: List[PlannedMove],
        turn_launch_cap: int,
    ) -> bool:
        if len(mission.source_ids) == 1:
            src_id = mission.source_ids[0]
            src = self.state.planets_by_id.get(src_id)
            if src is None or self._effective_planet(src).ships < mission.ships[0]:
                return False
            move = PlannedMove(
                src_id,
                mission.target_id,
                mission.angles[0],
                mission.ships[0],
                mission.etas[0],
                mission.mission,
            )
            moves.append(move)
            self._commit_move(move)
            return True

        if len(moves) + len(mission.source_ids) > turn_launch_cap:
            return False

        for index, source_id in enumerate(mission.source_ids):
            src = self.state.planets_by_id.get(source_id)
            if src is None or self._effective_planet(src).ships < mission.ships[index]:
                return False

        for index, source_id in enumerate(mission.source_ids):
            move = PlannedMove(
                source_id,
                mission.target_id,
                mission.angles[index],
                mission.ships[index],
                mission.etas[index],
                mission.mission,
            )
            moves.append(move)
            self._commit_move(move)
        return True

    def _commit_missions(self, missions: List[MissionOption]) -> List[PlannedMove]:
        remaining = sorted(missions, key=lambda mission: -mission.score)
        moves: List[PlannedMove] = []
        turn_launch_cap = self._turn_launch_cap()

        while remaining and not self.expired() and len(moves) < turn_launch_cap:
            horizon = min(len(remaining), max(1, self.rl_config.top_k))
            candidates: List[Dict[str, Any]] = []

            for idx in range(horizon):
                mission = remaining[idx]
                if not self._mission_can_commit(mission, existing_moves=len(moves)):
                    continue
                base_value = self._mission_base_value(mission, remaining, len(moves))
                feature_bundle = build_mission_feature_bundle(
                    self,
                    mission,
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

            if self._rl_window_open(len(candidates)):
                heuristic_pick = max(candidates, key=lambda item: item["base_value"])
                defer_vector = build_defer_vector(heuristic_pick["bundle"].vector)
                vectors = [defer_vector] + [item["bundle"].vector for item in candidates]
                choice = self.policy.choose(vectors, rng=random.Random(), explore=self.rl_config.explore)
                if choice.index == 0:
                    # Defer: use heuristic ranking
                    selected = heuristic_pick
                    deferred = True
                else:
                    selected = candidates[choice.index - 1]
                    deferred = False
                extra_metadata: Dict[str, Any] = {}
                if hasattr(self.policy, "decision_metadata"):
                    try:
                        extra_metadata = dict(self.policy.decision_metadata(vectors, choice))
                    except Exception:
                        extra_metadata = {}
                if self.recorder is not None:
                    self.recorder.record(
                        self.state.player,
                        DecisionSample(
                            feature_vectors=vectors,
                            chosen_index=choice.index,
                            probabilities=choice.probabilities,
                            metadata={
                                "step": self.state.step,
                                "candidate_count": len(candidates),
                                "chosen_probability": choice.probabilities[choice.index],
                                "deferred": deferred,
                                **selected["bundle"].metadata,
                                **extra_metadata,
                            },
                        ),
                    )
            else:
                selected = max(candidates, key=lambda item: item["base_value"])

            mission = remaining.pop(selected["remaining_index"])
            self._commit_selected_mission(mission, moves, turn_launch_cap)

        return moves


def build_agent(
    policy: Optional[Any] = None,
    rl_config: Optional[MidgameRLConfig] = None,
    recorder: Optional[EpisodeRecorder] = None,
    explore: Optional[bool] = None,
) -> Callable[[Any, Any], List[List[float | int]]]:
    policy = policy or LinearMissionPolicy()
    config = rl_config or MidgameRLConfig()
    if explore is not None:
        config = MidgameRLConfig(
            activation_turn=config.activation_turn,
            max_turn=config.max_turn,
            min_candidates=config.min_candidates,
            top_k=config.top_k,
            contested_only=config.contested_only,
            explore=bool(explore),
            follower_bonus_weight=config.follower_bonus_weight,
            allow_opening=config.allow_opening,
            force_rl_window=config.force_rl_window,
        )

    def agent(obs: Any, game_config: Any) -> List[List[float | int]]:
        try:
            logic = MidgameRLDecisionLogic(obs, game_config, policy, config, recorder=recorder)
            return logic.decide()
        except Exception as exc:
            BASE.AGENT_MEMORY["last_error"] = str(exc)
            return []

    return agent


def load_policy_agent(
    policy_path: str | Path,
    rl_config: Optional[MidgameRLConfig] = None,
    recorder: Optional[EpisodeRecorder] = None,
    explore: Optional[bool] = None,
) -> Callable[[Any, Any], List[List[float | int]]]:
    return build_agent(
        policy=load_policy_json(policy_path),
        rl_config=rl_config,
        recorder=recorder,
        explore=explore,
    )
