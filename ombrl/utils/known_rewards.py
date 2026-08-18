from typing import Optional, Tuple

import jax.numpy as jnp


MOUNTAIN_CAR_REWARD = "mountain_car_continuous"
CARTPOLE_SPARSE_REWARD = "cartpole_swingup_sparse"
HOPPER_HOP_REWARD = "hopper_hop"


def resolve_known_reward_type(env_name: str, enabled: bool) -> Optional[str]:
    """Return the supported reward map for an experiment environment."""
    if not enabled:
        return None
    reward_types = {
        "MountainCarContinuous-v0": MOUNTAIN_CAR_REWARD,
        "cartpole-swingup_sparse": CARTPOLE_SPARSE_REWARD,
        "hopper-hop": HOPPER_HOP_REWARD,
    }
    try:
        return reward_types[env_name]
    except KeyError as error:
        supported = ", ".join(sorted(reward_types))
        raise NotImplementedError(
            f"known_reward=True is not implemented for {env_name!r}. "
            f"Supported environments: {supported}."
        ) from error


def known_reward_and_termination(
        reward_type: str,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        action_repeat: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate a known task reward on model-generated next observations.

    Cartpole and Hopper expose rewards at every dm-control step. The current
    replay transition spans ``action_repeat`` such steps but stores only the
    final observation, so their outer-step reward is approximated by the final
    stage reward times ``action_repeat``. The discrepancy from the real summed
    reward is logged by the learner.
    """
    if reward_type == MOUNTAIN_CAR_REWARD:
        position = jnp.clip(next_observations[..., 0], -1.2, 0.6)
        velocity = jnp.clip(next_observations[..., 1], -0.07, 0.07)
        reached_goal = (position >= 0.45) & (velocity >= 0.0)
        reward = (
            100.0 * reached_goal.astype(next_observations.dtype)
            - 0.1 * jnp.square(actions[..., 0])
        )
        return reward, reached_goal

    if reward_type == CARTPOLE_SPARSE_REWARD:
        cart_position = next_observations[..., 0]
        pole_cosine = next_observations[..., 1]
        pole_sine = next_observations[..., 2]
        pole_norm = jnp.maximum(
            jnp.sqrt(jnp.square(pole_cosine) + jnp.square(pole_sine)),
            1e-6,
        )
        pole_cosine = pole_cosine / pole_norm
        in_target = (
            (cart_position >= -0.25)
            & (cart_position <= 0.25)
            & (pole_cosine >= 0.995)
        )
        reward = (
            jnp.asarray(action_repeat, dtype=next_observations.dtype)
            * in_target.astype(next_observations.dtype)
        )
        return reward, jnp.zeros_like(in_target)

    if reward_type == HOPPER_HOP_REWARD:
        height = next_observations[..., -2]
        speed = next_observations[..., -1]
        standing = (height >= 0.6) & (height <= 2.0)
        hopping = jnp.clip(speed / 2.0, 0.0, 1.0)
        reward = (
            jnp.asarray(action_repeat, dtype=next_observations.dtype)
            * standing.astype(next_observations.dtype)
            * hopping
        )
        return reward, jnp.zeros_like(standing)

    raise ValueError(f"Unknown known-reward type: {reward_type!r}")
