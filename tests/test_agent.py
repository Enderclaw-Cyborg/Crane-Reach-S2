"""Template tests. Every composed example inherits these and must pass them in CI.

They encode what every submittable repo should satisfy: the manifest parses and names a
loadable, instantiable agent class with the required interface, and that agent can actually
drive a few steps of the synced environment headlessly. They pass on the bare template because
it ships a small working starting agent, so a fresh clone is green out of the box; they keep
gating composed examples in CI and a student's own edits locally.
"""

from __future__ import annotations

import json
from pathlib import Path

from sandbox.env import META, make_env
from sandbox.harness.environment import resolve_parameters
from sandbox.play import load_agent, play_episode

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_manifest_parses_and_names_a_loadable_class():
    manifest = json.loads((REPO_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"entry_point", "class_name", "template_version"}
    assert isinstance(manifest["template_version"], int)
    agent = load_agent(REPO_ROOT)
    assert agent is not None


def test_agent_has_required_interface():
    agent = load_agent(REPO_ROOT)
    assert callable(getattr(agent, "reset", None))
    assert callable(getattr(agent, "act", None))


def test_three_step_headless_episode_runs():
    agent = load_agent(REPO_ROOT)
    env = make_env(resolve_parameters(META))
    try:
        score = play_episode(agent, env, seed=0, max_steps=3)
    finally:
        env.close()
    assert isinstance(score, float)


def test_agent_prefers_lower_cost_terrain_on_equal_distance():
    from agent import Agent

    grid = [[{"terrain": "grass", "feature": "none"} for _ in range(6)] for _ in range(6)]
    grid[1][3] = {"terrain": "marsh", "feature": "none"}
    grid[2][3] = {"terrain": "grass", "feature": "none"}
    observation = {
        "observation": {
            "self": {
                "unit_id": "red_footman_0",
                "type": "footman",
                "position": {"q": 2, "r": 2},
                "hit_points": 12,
                "movement_points": 2,
                "direction": 2,
            },
            "battlefield": {
                "side": 7,
                "tiles": tuple(tuple(row) for row in grid),
                "zones": (),
            },
        },
        "action_mask": {"path": [0, 1, 1, 0, 0, 0, 0] + [0] * (1555 - 7), "target": [0]},
    }

    step = Agent()._step_toward(observation, {"q": 4, "r": 2})

    assert step == 2
