"""
Works out of the box on your Mac. 2 million steps in 0.2 seconds on CPU alone.

  The key things to notice in the code:

  1. EnvState is a NamedTuple of arrays — no Python objects, no mutability. JAX needs this to trace the computation.
  2. step() has zero if/else — everything is jnp.where(condition, this, that). That's the main mental shift. GPU can't branch, so you compute all outcomes and mask.
  3. jax.vmap(step) — one line turns the single-env step into a 10,000-env batched step. You didn't write any batching logic.
  4. @jax.jit — one decorator compiles it all to XLA. First call is slow (compilation), every subsequent call runs at compiled speed.

  That's the whole pattern. An Orbit Wars JAX env would use exactly the same structure — just the step() function would be 500 lines of tensor ops instead of 10, and
  the state would hold planet positions, fleet arrays, ship counts etc. as fixed-size padded tensors. Same vmap, same jit.

JAX RL Hello World: Vectorized Grid World

A 5x5 grid. Agent starts at (0,0), goal is at (4,4).
Actions: 0=up, 1=right, 2=down, 3=left
Reward: +1 for reaching the goal, 0 otherwise.
Episode resets automatically when goal is reached or after 50 steps.

Key concepts demonstrated:
1. Env state is a pure pytree (no mutable objects)
2. step() is a pure function (no side effects)
3. jax.vmap runs 10,000 envs in parallel with zero code changes
4. jax.jit compiles everything to XLA for C++ speed

Install: pip install jax jaxlib
"""

import jax
import jax.numpy as jnp
import time
from functools import partial
from typing import NamedTuple


# ---------------------------------------------------------------------------
# 1. State is a pytree (just arrays, no Python mutability)
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    x: jnp.ndarray        # agent x position
    y: jnp.ndarray        # agent y position
    step_count: jnp.ndarray  # steps taken this episode
    done: jnp.ndarray     # whether episode is finished


GRID_SIZE = 5
GOAL_X = 4
GOAL_Y = 4
MAX_STEPS = 50


# ---------------------------------------------------------------------------
# 2. reset() and step() are pure functions — no side effects
# ---------------------------------------------------------------------------

def reset(key: jax.Array) -> EnvState:
    """Create a fresh episode. Agent at (0,0)."""
    return EnvState(
        x=jnp.int32(0),
        y=jnp.int32(0),
        step_count=jnp.int32(0),
        done=jnp.bool_(False),
    )


def step(state: EnvState, action: jnp.ndarray) -> tuple[EnvState, float, bool]:
    """
    Pure function: state + action -> (new_state, reward, done)

    No if/else branching — everything is computed with jnp.where (masks).
    This is what makes it GPU-compatible.
    """
    # Movement deltas for [up, right, down, left]
    dx = jnp.array([0, 1, 0, -1])
    dy = jnp.array([1, 0, -1, 0])

    # Compute new position (clipped to grid bounds)
    new_x = jnp.clip(state.x + dx[action], 0, GRID_SIZE - 1)
    new_y = jnp.clip(state.y + dy[action], 0, GRID_SIZE - 1)

    # Check if goal reached
    reached_goal = (new_x == GOAL_X) & (new_y == GOAL_Y)
    reward = jnp.where(reached_goal, 1.0, 0.0)

    new_step_count = state.step_count + 1
    done = reached_goal | (new_step_count >= MAX_STEPS)

    # Auto-reset: if done, snap back to start (no separate reset call needed)
    final_x = jnp.where(done, jnp.int32(0), new_x)
    final_y = jnp.where(done, jnp.int32(0), new_y)
    final_step_count = jnp.where(done, jnp.int32(0), new_step_count)

    new_state = EnvState(
        x=final_x,
        y=final_y,
        step_count=final_step_count,
        done=done,
    )
    return new_state, reward, done


# ---------------------------------------------------------------------------
# 3. vmap: write step() for 1 env, run it on 10,000 simultaneously
# ---------------------------------------------------------------------------

# This takes (batch_of_states, batch_of_actions) -> batch of results
batched_step = jax.vmap(step)


# ---------------------------------------------------------------------------
# 4. jit: compile the batched step to XLA (runs at C++ speed)
# ---------------------------------------------------------------------------

@jax.jit
def batched_step_jit(states: EnvState, actions: jnp.ndarray):
    return batched_step(states, actions)


# ---------------------------------------------------------------------------
# Demo: run 10,000 envs in parallel with random actions
# ---------------------------------------------------------------------------

def main():
    NUM_ENVS = 10_000
    NUM_STEPS = 200

    print(f"JAX backend: {jax.default_backend()}")
    print(f"Running {NUM_ENVS:,} environments in parallel for {NUM_STEPS} steps each\n")

    # Initialize all envs
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, NUM_ENVS)
    states = jax.vmap(reset)(keys)

    # Warm up JIT (first call compiles — ignore its timing)
    dummy_actions = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
    _ = batched_step_jit(states, dummy_actions)

    # Run the actual loop
    total_rewards = jnp.zeros(NUM_ENVS)
    total_goals = 0

    start = time.time()
    for t in range(NUM_STEPS):
        # Random policy: pick a random action for each env
        key, subkey = jax.random.split(key)
        actions = jax.random.randint(subkey, (NUM_ENVS,), 0, 4)

        # Step ALL 10,000 envs in one call
        states, rewards, dones = batched_step_jit(states, actions)

        total_rewards += rewards
        total_goals += int(dones.sum())

    elapsed = time.time() - start

    total_steps = NUM_ENVS * NUM_STEPS
    steps_per_second = total_steps / elapsed

    print(f"Total steps:      {total_steps:,}")
    print(f"Time:             {elapsed:.3f}s")
    print(f"Steps/second:     {steps_per_second:,.0f}")
    print(f"Goals reached:    {total_goals:,}")
    print(f"Avg reward/env:   {float(total_rewards.mean()):.2f}")
    print()
    print("--- For comparison ---")
    print(f"Orbit Wars does ~{total_steps // 30:,} steps in the same time (at 30s/game, ~200 steps/game)")
    print(f"That's a ~{int(steps_per_second / (200/30)):,}x speedup over your current setup")


if __name__ == "__main__":
    main()
