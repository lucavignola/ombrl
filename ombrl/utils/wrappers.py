from gymnasium import Wrapper
from typing import Optional
import numpy as np
import gymnasium as gym

class PendulumInitWrapper(Wrapper):
    def __init__(self, env, init_angle: float = np.pi, init_vel: float = 0.0):
        """
        Wraps the pendulum environment to override its reset.
        
        Args:
            env: The original pendulum environment.
            init_angle: The desired initial angle (in radians).
            init_vel: The desired initial angular velocity.
        """
        super().__init__(env)
        self.init_angle = init_angle
        self.init_vel = init_vel

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)

        self.env.unwrapped.state = np.array([self.init_angle, self.init_vel])

        new_obs = self.env.unwrapped._get_obs()
        return new_obs, info


class AdditiveGaussianProcessNoise(Wrapper):
    """Apply Gaussian noise to the simulator state after each transition.

    For MuJoCo environments the disturbance is applied to generalized
    velocities, which avoids invalidating free-joint quaternions. Environments
    exposing a NumPy ``state`` receive additive noise on that state. The old
    observation-noise behavior is retained only as a fallback for simulators
    whose state cannot be mutated.
    """

    def __init__(self, env, noise_std: float = 0.0, seed: Optional[int] = None):
        super().__init__(env)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)
        self.base_env = env.unwrapped
        self._wrapper_chain = []
        current = env
        while isinstance(current, gym.Wrapper):
            self._wrapper_chain.append(current)
            current = current.env

        self._physics = getattr(self.base_env, "physics", None)
        self._dm_control_env = getattr(self.base_env, "_env", None)
        if self._physics is not None and hasattr(self._physics.data, "qvel"):
            self.noise_mode = "simulator_velocity"
        elif (hasattr(self.base_env, "data")
              and hasattr(self.base_env.data, "qvel")
              and hasattr(self.base_env, "set_state")):
            self.noise_mode = "simulator_velocity"
        elif hasattr(self.base_env, "state"):
            self.noise_mode = "simulator_state"
        else:
            self.noise_mode = "observation_fallback"

        self._clip_low = None
        self._clip_high = None
        space = self.observation_space
        if isinstance(space, gym.spaces.Box):
            if np.all(np.isfinite(space.low)) or np.all(np.isfinite(space.high)):
                self._clip_low = space.low
                self._clip_high = space.high

    def _add_observation_noise(self, observation):
        if self.noise_std <= 0.0:
            return observation
        if isinstance(observation, dict):
            return {key: self._add_observation_noise(value) for key, value in observation.items()}
        if not isinstance(observation, np.ndarray) or not np.issubdtype(observation.dtype, np.floating):
            return observation

        noisy_observation = observation + self.rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=observation.shape,
        ).astype(observation.dtype)

        if self._clip_low is not None and self.observation_space.shape == noisy_observation.shape:
            noisy_observation = np.clip(noisy_observation, self._clip_low, self._clip_high)
        return noisy_observation.astype(observation.dtype, copy=False)

    def _apply_simulator_noise(self):
        if self._physics is not None and hasattr(self._physics.data, "qvel"):
            qvel = self._physics.data.qvel
            qvel[:] = qvel + self.rng.normal(0.0, self.noise_std, qvel.shape)
            if hasattr(self._physics, "after_reset"):
                self._physics.after_reset()
            elif hasattr(self._physics, "forward"):
                self._physics.forward()
        elif self.noise_mode == "simulator_velocity":
            qpos = np.asarray(self.base_env.data.qpos).copy()
            qvel = np.asarray(self.base_env.data.qvel).copy()
            qvel += self.rng.normal(0.0, self.noise_std, qvel.shape)
            self.base_env.set_state(qpos, qvel)
        else:
            state = np.asarray(self.base_env.state)
            self.base_env.state = state + self.rng.normal(
                0.0, self.noise_std, state.shape)

        return self._current_observation()

    def _current_observation(self):
        if self._dm_control_env is not None and hasattr(self._dm_control_env, "task"):
            observation = self._dm_control_env.task.get_observation(self._physics)
            if getattr(self._dm_control_env, "_flat_observation", False):
                from dm_control.rl.control import flatten_observation
                observation = flatten_observation(observation)
        elif hasattr(self.base_env, "_get_obs"):
            observation = self.base_env._get_obs()
        elif (hasattr(self.base_env, "state")
              and self.base_env.observation_space.shape == np.asarray(self.base_env.state).shape):
            observation = np.asarray(
                self.base_env.state,
                dtype=self.base_env.observation_space.dtype,
            ).copy()
        else:
            raise RuntimeError(
                f"Cannot reconstruct an observation for {type(self.base_env).__name__}."
            )

        for wrapper in reversed(self._wrapper_chain):
            if isinstance(wrapper, gym.ObservationWrapper):
                observation = wrapper.observation(observation)
            elif hasattr(wrapper, "_frames") and hasattr(wrapper, "_get_obs"):
                # FrameStack already appended the pre-noise observation while
                # stepping. Replace that frame with the actual noisy state.
                wrapper._frames[-1] = observation
                observation = wrapper._get_obs()
        return observation

    def step(self, action):
        step = self.env.step(action)
        if len(step) == 5:
            observation, reward, terminated, truncated, info = step
            if self.noise_std > 0.0:
                if self.noise_mode == "observation_fallback":
                    observation = self._add_observation_noise(observation)
                else:
                    observation = self._apply_simulator_noise()
            info = dict(info)
            info['process_noise_std'] = self.noise_std
            info['process_noise_mode'] = self.noise_mode
            return observation, reward, terminated, truncated, info

        observation, reward, done, info = step
        if self.noise_std > 0.0:
            if self.noise_mode == "observation_fallback":
                observation = self._add_observation_noise(observation)
            else:
                observation = self._apply_simulator_noise()
        info = dict(info)
        info['process_noise_std'] = self.noise_std
        info['process_noise_mode'] = self.noise_mode
        return observation, reward, done, info


# Example usage:
if __name__ == "__main__":
    env = gym.make("Pendulum-v1")
    env = PendulumInitWrapper(env, init_angle=0.0, init_vel=0.0)
    
    obs, info = env.reset(seed=42)
    print("Initial observation:", obs)
    
    n_steps = 5
    for step in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"\nStep {step + 1}:")
        print(" Action:", action)
        print(" Observation:", obs)
        print(" Reward:", reward)
        print(" Terminated:", terminated)
        print(" Truncated:", truncated)
