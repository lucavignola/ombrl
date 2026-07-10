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
    def __init__(self, env, noise_std: float = 0.0, seed: Optional[int] = None):
        super().__init__(env)
        self.noise_std = float(noise_std)
        self.rng = np.random.default_rng(seed)

    def _add_noise(self, observation):
        if self.noise_std <= 0.0:
            return observation
        if isinstance(observation, dict):
            return {key: self._add_noise(value) for key, value in observation.items()}
        if not isinstance(observation, np.ndarray) or not np.issubdtype(observation.dtype, np.floating):
            return observation

        noisy_observation = observation + self.rng.normal(
            loc=0.0,
            scale=self.noise_std,
            size=observation.shape,
        ).astype(observation.dtype)

        space = self.observation_space
        if isinstance(space, gym.spaces.Box) and space.shape == noisy_observation.shape:
            low = space.low
            high = space.high
            if np.all(np.isfinite(low)) or np.all(np.isfinite(high)):
                noisy_observation = np.clip(noisy_observation, low, high)
        return noisy_observation.astype(observation.dtype, copy=False)

    def step(self, action):
        step = self.env.step(action)
        if len(step) == 5:
            observation, reward, terminated, truncated, info = step
            observation = self._add_noise(observation)
            info = dict(info)
            info['process_noise_std'] = self.noise_std
            return observation, reward, terminated, truncated, info

        observation, reward, done, info = step
        observation = self._add_noise(observation)
        info = dict(info)
        info['process_noise_std'] = self.noise_std
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
