import copy

import numpy as np


class EnvStateAccessor:
    def __init__(self, env):
        self.env = env
        self.base_env = env.unwrapped
        physics = getattr(self.base_env, "physics", None)
        if physics is not None and hasattr(physics, "get_state") and hasattr(physics, "set_state"):
            self._snapshot = self._snapshot_physics
            self._restore = self._restore_physics
            self._physics = physics
            return
        if hasattr(self.base_env, "state"):
            self._snapshot = self._snapshot_state
            self._restore = self._restore_state
            return
        raise NotImplementedError(
            "input_knowledge=True requires a simulator state snapshot. "
            f"Unsupported env type: {type(self.base_env).__name__}."
        )

    def snapshot(self):
        return self._snapshot()

    def restore(self, state) -> None:
        self._restore(state)

    def _snapshot_physics(self):
        return np.asarray(self._physics.get_state()).copy()

    def _restore_physics(self, state) -> None:
        self._physics.set_state(state)
        if hasattr(self._physics, "after_reset"):
            self._physics.after_reset()

    def _snapshot_state(self):
        return np.asarray(self.base_env.state).copy()

    def _restore_state(self, state) -> None:
        self.base_env.state = state.copy()


class TrueInputEffect:
    """Estimate F(x, u) - F(x, 0) from a simulator with restorable state."""

    def __init__(self, env, action_space, clone_env: bool = False, preserve_state: bool = True):
        self.env = copy.deepcopy(env) if clone_env else env
        self.zero_action = np.zeros(action_space.shape, dtype=action_space.dtype)
        self.preserve_state = preserve_state
        self.state_accessor = EnvStateAccessor(self.env)

    def batch_effect(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        effects = np.empty_like(observations)
        for idx, (observation, action) in enumerate(zip(observations, actions)):
            effects[idx] = self.effect(observation, action)
        return effects

    def batch_effect_from_states(self, states, actions: np.ndarray, observation_shape) -> np.ndarray:
        actions = np.asarray(actions)
        effects = np.empty((len(actions),) + tuple(observation_shape), dtype=np.float32)
        restore = self.state_accessor.restore
        step = self.env.step
        zero_action = self.zero_action
        current_state = self.state_accessor.snapshot() if self.preserve_state else None
        try:
            for idx, (state, action) in enumerate(zip(states, actions)):
                restore(state)
                next_with_action = step(action)[0]
                restore(state)
                next_with_zero_action = step(zero_action)[0]
                effects[idx] = next_with_action - next_with_zero_action
        finally:
            if current_state is not None:
                restore(current_state)
        return effects

    def effect_from_state(self, state, action: np.ndarray) -> np.ndarray:
        next_with_action = self._next_observation_from_state(state, action)
        next_with_zero_action = self._next_observation_from_state(state, self.zero_action)
        return next_with_action - next_with_zero_action

    def effect(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        current_state = self.state_accessor.snapshot() if self.preserve_state else None
        try:
            next_with_action = self._next_observation(observation, action)
            next_with_zero_action = self._next_observation(observation, self.zero_action)
            return next_with_action - next_with_zero_action
        finally:
            if current_state is not None:
                self.state_accessor.restore(current_state)

    def snapshot_state(self):
        return self.state_accessor.snapshot()

    def _next_observation_from_state(self, state, action: np.ndarray) -> np.ndarray:
        self.state_accessor.restore(state)
        next_observation, *_ = self.env.step(action)
        return np.asarray(next_observation, dtype=np.float32)

    def _next_observation(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        self._set_state_from_observation(observation)
        next_observation, *_ = self.env.step(action)
        return np.asarray(next_observation, dtype=observation.dtype)

    def _set_state_from_observation(self, observation: np.ndarray) -> None:
        observation = np.asarray(observation)
        base_env = self.env.unwrapped

        if hasattr(base_env, "set_state_from_observation"):
            base_env.set_state_from_observation(observation)
            return

        if hasattr(base_env, "state"):
            state = np.asarray(base_env.state)
            if state.shape == observation.shape:
                base_env.state = observation.astype(state.dtype, copy=True)
                return

        if self._try_set_pendulum_state(base_env, observation):
            return

        if self._try_set_physics_state(base_env, observation):
            return

        raise NotImplementedError(
            "input_knowledge=True requires restoring the simulator state from "
            "an observation. Add a set_state_from_observation(observation) "
            "method to the unwrapped env, or extend TrueInputEffect for this "
            f"env type: {type(base_env).__name__}."
        )

    @staticmethod
    def _try_set_pendulum_state(base_env, observation: np.ndarray) -> bool:
        if observation.shape != (3,) or not hasattr(base_env, "state"):
            return False
        env_name = base_env.spec.id if getattr(base_env, "spec", None) is not None else ""
        if "Pendulum" not in env_name:
            return False
        theta = np.arctan2(observation[1], observation[0])
        theta_dot = observation[2]
        base_env.state = np.array([theta, theta_dot], dtype=np.asarray(base_env.state).dtype)
        return True

    @staticmethod
    def _try_set_physics_state(base_env, observation: np.ndarray) -> bool:
        physics = getattr(base_env, "physics", None)
        if physics is None or not hasattr(physics, "get_state") or not hasattr(physics, "set_state"):
            return False
        state = np.asarray(physics.get_state())
        flat_observation = observation.reshape(-1)
        if state.shape != flat_observation.shape:
            return False
        physics.set_state(flat_observation.astype(state.dtype, copy=False))
        if hasattr(physics, "after_reset"):
            physics.after_reset()
        return True


def snapshot_env_state(env):
    return EnvStateAccessor(env).snapshot()


def restore_env_state(env, state) -> None:
    EnvStateAccessor(env).restore(state)


class SimulatorStateBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.states = [None] * self.capacity

    def insert(self, index: int, state) -> None:
        self.states[int(index) % self.capacity] = state

    def get(self, indices: np.ndarray):
        sampled_states = [self.states[int(index) % self.capacity] for index in indices]
        if any(state is None for state in sampled_states):
            raise RuntimeError("Sampled a replay entry before its simulator state was recorded.")
        return sampled_states
