from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_DIR))

from genome_agent import (  # noqa: E402
    BASE_AGENT_PATH,
    GENE_SPACE,
    GenomeConfig,
    PRESET_GENOMES,
    build_agent,
    crossover_genomes,
    mutate_genome,
    random_genome,
    resolve_base_agent_path,
    write_agent_wrapper,
)
import genetic_search  # noqa: E402
from genetic_search import (  # noqa: E402
    STAGE3_WRAPPER_PATHS,
    build_covering_population,
    emit_current_wrappers,
    load_reference_agents,
    load_resume_records,
    pareto_front,
    write_summary_file,
)
from weird_opponents import greedy_agent, turtle_agent  # noqa: E402


class GenomeSearchTests(unittest.TestCase):
    def test_presets_are_valid(self) -> None:
        for genome in PRESET_GENOMES.values():
            genome.validate()

    def test_random_mutate_and_crossover_stay_in_gene_space(self) -> None:
        rng = random.Random(7)
        genome_a = random_genome(rng)
        genome_b = random_genome(rng)
        child = crossover_genomes(genome_a, genome_b, rng)
        mutated = mutate_genome(child, rng, mutation_rate=0.8)
        for genome in (genome_a, genome_b, child, mutated):
            payload = genome.to_dict()
            for name, options in GENE_SPACE.items():
                self.assertIn(payload[name], options)

    def test_build_agent_returns_callable(self) -> None:
        agent = build_agent(GenomeConfig())
        self.assertTrue(callable(agent))

    def test_genome_workspace_uses_leaderboard_base(self) -> None:
        self.assertEqual(BASE_AGENT_PATH.name, "stage4_leaderboard_search_base.py")

    def test_resolve_base_agent_path_honors_env_override(self) -> None:
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            override = Path(tmpdir) / "custom_base.py"
            override.write_text("def agent(obs, config):\n    return []\n", encoding="utf-8")
            previous = os.environ.get("ORBIT_WARS_BASE_AGENT_PATH")
            os.environ["ORBIT_WARS_BASE_AGENT_PATH"] = str(override)
            try:
                self.assertEqual(resolve_base_agent_path(), override.resolve())
            finally:
                if previous is None:
                    os.environ.pop("ORBIT_WARS_BASE_AGENT_PATH", None)
                else:
                    os.environ["ORBIT_WARS_BASE_AGENT_PATH"] = previous

    def test_reference_agents_include_live_oldbase_controls(self) -> None:
        references = load_reference_agents()
        self.assertIn("oldbase_balanced", references)
        self.assertIn("oldbase_two_player", references)
        self.assertIn("release_candidate_v2", references)
        self.assertIn("s2_4p_antidogpile", references)

    def test_reference_agents_include_stage3_wrappers_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            wrapper_paths = {
                "stage3_balanced": tmp / "current-balanced.py",
                "stage3_two_player": tmp / "current-two-player.py",
                "stage3_four_player": tmp / "current-four-player.py",
            }
            for path in wrapper_paths.values():
                path.write_text("def agent(obs, config):\n    return []\n", encoding="utf-8")
            original_paths = genetic_search.STAGE3_WRAPPER_PATHS
            genetic_search.STAGE3_WRAPPER_PATHS = wrapper_paths
            try:
                references = load_reference_agents()
            finally:
                genetic_search.STAGE3_WRAPPER_PATHS = original_paths
            self.assertIn("stage3_balanced", references)
            self.assertIn("stage3_two_player", references)
            self.assertIn("stage3_four_player", references)

    def test_covering_population_spreads_gene_values(self) -> None:
        genomes = build_covering_population(12, random.Random(7))
        self.assertGreaterEqual(len(genomes), 12)
        style_profiles = {genome.style_profile for genome in genomes}
        duel_openings = {genome.duel_opening for genome in genomes}
        value_profiles = {genome.value_profile for genome in genomes}
        concentration_profiles = {genome.concentration_profile for genome in genomes}
        transition_profiles = {genome.transition_profile for genome in genomes}
        opening_range_profiles = {genome.opening_range_profile for genome in genomes}
        vulture_profiles = {genome.vulture_profile for genome in genomes}
        conversion_profiles = {genome.conversion_profile for genome in genomes}
        self.assertGreaterEqual(len(style_profiles), 3)
        self.assertGreaterEqual(len(duel_openings), 2)
        self.assertGreaterEqual(len(value_profiles), 3)
        self.assertGreaterEqual(len(concentration_profiles), 2)
        self.assertGreaterEqual(len(transition_profiles), 2)
        self.assertGreaterEqual(len(opening_range_profiles), 2)
        self.assertGreaterEqual(len(vulture_profiles), 2)
        self.assertGreaterEqual(len(conversion_profiles), 2)

    def test_pareto_front_keeps_non_dominated_records(self) -> None:
        records = [
            {"slug": "balanced", "objective_2p": 0.70, "four_player": {"score": 0.70}, "balanced_score": 0.70},
            {"slug": "two-player", "objective_2p": 0.90, "four_player": {"score": 0.50}, "balanced_score": 0.72},
            {"slug": "four-player", "objective_2p": 0.50, "four_player": {"score": 0.90}, "balanced_score": 0.72},
            {"slug": "dominated", "objective_2p": 0.40, "four_player": {"score": 0.40}, "balanced_score": 0.40},
        ]
        front = pareto_front(records)
        slugs = {record["slug"] for record in front}
        self.assertIn("balanced", slugs)
        self.assertIn("two-player", slugs)
        self.assertIn("four-player", slugs)
        self.assertNotIn("dominated", slugs)

    def test_weird_opponents_return_action_lists(self) -> None:
        obs = {"player": 0, "step": 0, "planets": []}
        config = {"shipSpeed": 6.0}
        self.assertEqual(greedy_agent(obs, config), [])
        self.assertEqual(turtle_agent(obs, config), [])

    def test_wrapper_emission_writes_genome_payload(self) -> None:
        genome = GenomeConfig(duel_opening="mtmr", value_profile="economy")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_agent_wrapper(genome, Path(tmpdir) / "wrapper.py")
            content = path.read_text(encoding="utf-8")
        self.assertIn("GenomeConfig.from_dict", content)
        self.assertIn('"duel_opening": "mtmr"', content)
        self.assertIn('"value_profile": "economy"', content)

    def test_resume_records_and_summary_writing(self) -> None:
        record = {
            "record_type": "complete",
            "generation": 0,
            "slug": "balanced-v23",
            "genome": GenomeConfig().to_dict(),
            "fixed_two_player": {},
            "fixed_two_player_score": 0.5,
            "self_play": {"games": 0.0, "score_rate": 0.0, "avg_reward_diff": 0.0, "avg_ship_diff": 0.0},
            "mutant_pool": {"games": 0.0, "score_rate": 0.0, "avg_reward_diff": 0.0, "avg_ship_diff": 0.0, "series": {}},
            "objective_2p": 0.5,
            "four_player": {"score": 0.4, "games": 4.0, "avg_rank": 2.0, "top2_rate": 1.0, "avg_reward": 0.0, "avg_ships": 0.0, "wins": 0.0, "top2": 4.0, "rank_sum": 8.0, "reward_sum": 0.0, "ship_sum": 0.0},
            "balanced_score": 0.455,
            "combined_score": 0.455,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            jsonl = tmp / "search_log.jsonl"
            partial = {
                "record_type": "partial",
                "phase": "fixed_two_player",
                "generation": 0,
                "slug": "balanced-v23",
                "genome": GenomeConfig().to_dict(),
                "opponent": "baseline",
                "result": {"score_rate": 0.5},
                "progress": {"generation": 0, "phase": "fixed_two_player", "slug": "balanced-v23"},
            }
            jsonl.write_text(
                __import__("json").dumps(partial) + "\n" + __import__("json").dumps(record) + "\n",
                encoding="utf-8",
            )
            loaded = load_resume_records(jsonl)
            self.assertIn((0, "balanced-v23"), loaded["complete"])
            self.assertEqual(loaded["partial_genomes"], {})

            class Args:
                population = 4
                generations = 1
                games_per_seat = 1
                self_play_games_per_seat = 1
                mutant_games_per_seat = 0
                champion_mutants = 0
                seed = 7
                skip_four_player = False
                two_player_opponents = ["baseline", "oldbase_balanced"]
                four_player_opponents = ["oldbase_balanced", "baseline", "v23"]
                resume = True
                emit_top = 2

            summary_path = tmp / "search_summary.json"
            payload = write_summary_file(
                summary_path,
                Args,
                [],
                [record],
                current_progress={"phase": "fixed_two_player", "slug": "balanced-v23"},
            )
            self.assertEqual(payload["champions"]["balanced"]["slug"], "balanced-v23")
            self.assertEqual(payload["current_progress"]["slug"], "balanced-v23")
            emitted = emit_current_wrappers(tmp, payload, generation=0, emit_top=2)
            self.assertTrue(any(path.exists() for path in emitted.values()))


if __name__ == "__main__":
    unittest.main()
