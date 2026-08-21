from gymnasium import Wrapper
from typing import Optional
import numpy as np
import gymnasium as gym

from ombrl.utils.known_rewards import known_reward_and_termination_numpy


class QuadrupedPhysicalStateObservation(gym.ObservationWrapper):
    """Expose MuJoCo integration state modulo horizontal translation.

    The standard dm-control quadruped observation mixes an incomplete physical
    state with accelerometer and contact-force sensors. This representation
    instead contains qpos without global x/y, followed by qvel and actuator
    activation state.
    """

    _OMITTED_ROOT_TRANSLATIONS = 2

    def __init__(self, env):
        super().__init__(env)
        self._physics = getattr(env.unwrapped, "physics", None)
        if self._physics is None:
            raise TypeError(
                "QuadrupedPhysicalStateObservation requires a dm-control "
                "environment exposing physics."
            )

        model = self._physics.model
        import mujoco

        joint_types = np.asarray(model.jnt_type)
        joint_qpos_addresses = np.asarray(model.jnt_qposadr)
        free_joint_type = int(mujoco.mjtJoint.mjJNT_FREE)
        if (joint_types.size == 0 or int(joint_types[0]) != free_joint_type
                or int(joint_qpos_addresses[0]) != 0):
            raise ValueError(
                "Expected the quadruped root to be the first MuJoCo free joint."
            )

        observation_dim = (
            int(model.nq) - self._OMITTED_ROOT_TRANSLATIONS
            + int(model.nv)
            + int(model.na)
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )

    def observation(self, observation):
        del observation
        data = self._physics.data
        state = np.concatenate((
            np.asarray(data.qpos)[self._OMITTED_ROOT_TRANSLATIONS:],
            np.asarray(data.qvel),
            np.asarray(data.act),
        ))
        return state.astype(np.float32, copy=False)


class HopperKnownRewardObservation(gym.ObservationWrapper):
    """Append the physical coordinates used by Hopper's reward function."""

    _NUM_REWARD_COORDINATES = 2

    def __init__(self, env):
        super().__init__(env)
        self._physics = getattr(env.unwrapped, "physics", None)
        if (self._physics is None
                or not hasattr(self._physics, "height")
                or not hasattr(self._physics, "speed")):
            raise TypeError(
                "HopperKnownRewardObservation requires dm-control Hopper physics."
            )
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("Hopper observations must use a Box space.")

        low = np.concatenate((
            np.asarray(env.observation_space.low).reshape(-1),
            np.full(self._NUM_REWARD_COORDINATES, -np.inf),
        ))
        high = np.concatenate((
            np.asarray(env.observation_space.high).reshape(-1),
            np.full(self._NUM_REWARD_COORDINATES, np.inf),
        ))
        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation):
        reward_coordinates = np.asarray(
            [self._physics.height(), self._physics.speed()],
            dtype=np.float32,
        )
        return np.concatenate((
            np.asarray(observation, dtype=np.float32).reshape(-1),
            reward_coordinates,
        ))


class KnownRewardTransition(gym.Wrapper):
    """Make real and imagined transitions use the same known reward map.

    For repeated-action dm-control tasks this deliberately defines the outer
    reward from the final noisy state. That is the quantity available to the
    one-step learned model; the original environment instead sums rewards from
    intermediate, pre-noise simulator states.
    """

    def __init__(self, env, reward_type: str, action_repeat: int):
        super().__init__(env)
        self.reward_type = reward_type
        self.action_repeat = int(action_repeat)
        if self.action_repeat < 1:
            raise ValueError("Known rewards require action_repeat >= 1.")
        self._episode_return = 0.0
        self._episode_outer_steps = 0

    def reset(self, **kwargs):
        self._episode_return = 0.0
        self._episode_outer_steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, _, _, truncated, info = self.env.step(action)
        reward, known_termination = known_reward_and_termination_numpy(
            self.reward_type,
            action,
            observation,
            self.action_repeat,
        )
        terminated = bool(known_termination)
        self._episode_return += reward
        self._episode_outer_steps += 1

        info = dict(info)
        info['known_reward_transition'] = True
        if terminated or truncated:
            episode = dict(info.get('episode', {}))
            episode['return'] = self._episode_return
            episode.setdefault(
                'length', self._episode_outer_steps * self.action_repeat
            )
            episode.setdefault('duration', 0.0)
            info['episode'] = episode
        else:
            # MountainCar may have terminated before post-transition process
            # noise moved it away from the goal. The known noisy-state MDP has
            # not terminated in that case.
            info.pop('episode', None)
        return observation, reward, terminated, truncated, info


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
    velocities and any actuator activation state. Optional position noise is
    integrated in MuJoCo's tangent space, preserving valid free-joint
    quaternions. Environments exposing a NumPy ``state`` receive bounded
    additive noise on that state. Observation noise is retained only as a
    fallback for simulators whose state cannot be mutated.
    """

    def __init__(self, env, noise_std: float = 0.0,
                 actuator_noise_std: Optional[float] = None,
                 position_noise_std: float = 0.0,
                 project_velocity_noise: bool = False,
                 seed: Optional[int] = None):
        super().__init__(env)
        self.noise_std = float(noise_std)
        self.position_noise_std = float(position_noise_std)
        self.actuator_noise_std = (
            self.noise_std
            if actuator_noise_std is None
            else float(actuator_noise_std)
        )
        if (self.noise_std < 0.0 or self.position_noise_std < 0.0
                or self.actuator_noise_std < 0.0):
            raise ValueError("Process-noise standard deviations must be non-negative.")
        self.has_process_noise = (
            self.noise_std > 0.0
            or self.position_noise_std > 0.0
            or self.actuator_noise_std > 0.0
        )
        self.project_velocity_noise = bool(project_velocity_noise)
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

        self._mujoco_model = None
        self._mujoco_data = None
        self._mass_solve_rhs = None
        self._mass_solve_result = None
        self._solve_mass_matrix = None
        self._integrate_position = None
        self._equality_constraint_type = None
        if self.project_velocity_noise or self.position_noise_std > 0.0:
            self._initialize_mujoco_state_noise()

        self._clip_low = None
        self._clip_high = None
        space = self.observation_space
        if isinstance(space, gym.spaces.Box):
            if np.all(np.isfinite(space.low)) or np.all(np.isfinite(space.high)):
                self._clip_low = space.low
                self._clip_high = space.high

    def _initialize_mujoco_state_noise(self):
        if self._physics is None or not hasattr(self._physics.data, "qvel"):
            raise TypeError(
                "Position or constraint-projected process noise requires "
                "dm-control MuJoCo physics."
            )

        import mujoco

        self._mujoco_model = getattr(self._physics.model, "ptr", self._physics.model)
        self._mujoco_data = getattr(self._physics.data, "ptr", self._physics.data)
        self._equality_constraint_type = int(
            mujoco.mjtConstraint.mjCNSTR_EQUALITY
        )
        # A single equality (for example, a weld) can occupy multiple rows.
        solve_shape = (
            int(self._mujoco_model.nv),
            int(self._mujoco_model.nv),
        )
        self._mass_solve_rhs = np.empty(
            solve_shape,
            dtype=np.float64,
        )
        self._mass_solve_result = np.empty_like(self._mass_solve_rhs)
        self._solve_mass_matrix = lambda result, rhs: mujoco.mj_solveM(
            self._mujoco_model, self._mujoco_data, result, rhs
        )
        self._integrate_position = lambda position, velocity: mujoco.mj_integratePos(
            self._mujoco_model, position, velocity, 1.0
        )

    def _project_onto_equality_tangent(self, disturbance):
        data = self._mujoco_data
        model = self._mujoco_model
        num_constraints = int(data.nefc)
        if num_constraints == 0:
            return disturbance

        constraint_types = np.asarray(data.efc_type[:num_constraints])
        equality_rows = constraint_types == self._equality_constraint_type
        if not np.any(equality_rows):
            return disturbance

        constraint_jacobian = np.asarray(data.efc_J)
        required_size = num_constraints * model.nv
        if constraint_jacobian.size < required_size:
            raise RuntimeError(
                "Constraint projection requires MuJoCo's dense constraint Jacobian."
            )
        constraint_jacobian = constraint_jacobian.reshape(-1, model.nv)
        equality_jacobian = constraint_jacobian[:num_constraints][equality_rows]

        num_equalities = equality_jacobian.shape[0]
        mass_solve_rhs = self._mass_solve_rhs[:num_equalities]
        mass_solve_result = self._mass_solve_result[:num_equalities]
        mass_solve_rhs[:] = equality_jacobian
        self._solve_mass_matrix(mass_solve_result, mass_solve_rhs)
        inverse_mass_jacobian_t = mass_solve_result.T
        constraint_mass = (
            equality_jacobian @ inverse_mass_jacobian_t
        )
        multiplier = np.linalg.solve(
            constraint_mass, equality_jacobian @ disturbance
        )
        return disturbance - inverse_mass_jacobian_t @ multiplier

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
            if self.noise_std > 0.0:
                disturbance = self.rng.normal(0.0, self.noise_std, qvel.shape)
                if self.project_velocity_noise:
                    disturbance = self._project_onto_equality_tangent(disturbance)
                qvel[:] = qvel + disturbance

            if self.position_noise_std > 0.0:
                position_disturbance = self.rng.normal(
                    0.0, self.position_noise_std, qvel.shape
                )
                if self.project_velocity_noise:
                    position_disturbance = self._project_onto_equality_tangent(
                        position_disturbance
                    )
                self._integrate_position(
                    self._physics.data.qpos,
                    position_disturbance,
                )

            actuator_state = self._physics.data.act
            if self.actuator_noise_std > 0.0 and actuator_state.size:
                actuator_state[:] = actuator_state + self.rng.normal(
                    0.0, self.actuator_noise_std, actuator_state.shape
                )
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
            noisy_state = state + self.rng.normal(
                0.0, self.noise_std, state.shape)
            if (self._clip_low is not None
                    and self.observation_space.shape == noisy_state.shape):
                noisy_state = np.clip(
                    noisy_state, self._clip_low, self._clip_high
                )
            if (noisy_state.size >= 2
                    and hasattr(self.base_env, "min_position")
                    and noisy_state[0] <= self.base_env.min_position
                    and noisy_state[1] < 0.0):
                noisy_state[1] = 0.0
            self.base_env.state = noisy_state.astype(state.dtype, copy=False)

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
            if self.has_process_noise:
                if self.noise_mode == "observation_fallback":
                    observation = self._add_observation_noise(observation)
                else:
                    observation = self._apply_simulator_noise()
            info = dict(info)
            info['process_noise_std'] = self.noise_std
            info['process_position_noise_std'] = self.position_noise_std
            info['process_actuator_noise_std'] = self.actuator_noise_std
            info['process_noise_projected'] = self.project_velocity_noise
            info['process_noise_mode'] = self.noise_mode
            return observation, reward, terminated, truncated, info

        observation, reward, done, info = step
        if self.has_process_noise:
            if self.noise_mode == "observation_fallback":
                observation = self._add_observation_noise(observation)
            else:
                observation = self._apply_simulator_noise()
        info = dict(info)
        info['process_noise_std'] = self.noise_std
        info['process_position_noise_std'] = self.position_noise_std
        info['process_actuator_noise_std'] = self.actuator_noise_std
        info['process_noise_projected'] = self.project_velocity_noise
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
