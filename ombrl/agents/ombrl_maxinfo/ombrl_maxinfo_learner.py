import functools
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import copy
from jaxrl.agents.sac import temperature
from maxinforl_jax.agents.maxinfosac.critic import target_update
from ombrl.utils.pertubation import PerturbationModule
from jaxrl.agents.sac.temperature import update as update_temp

from jaxrl.datasets import Batch
from jaxrl.networks import critic_net, policies
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey

from maxinforl_jax.models import EnsembleState, DeterministicEnsemble, ProbabilisticEnsemble


def get_imagined_batch(
        batch: Batch,
        ens: DeterministicEnsemble,
        ens_state: EnsembleState,
        known_input_effect: Optional[jnp.ndarray],
        input_knowledge: bool,
        predict_rewards: bool,
        predict_diff: bool,
        sample_model: bool,
        internal_noise_std: float,
        internal_noise_samples: int,
        key: PRNGKey, # type: ignore
        dt: float = None,
        action_repeat: int = 1,
):
    if input_knowledge:
        input = batch.observations
    else:
        input = jnp.concatenate([batch.observations, batch.actions], axis=-1)
    ens_mean, ens_std = ens(input=input, state=ens_state, denormalize_output=True)
    noise_key, key = jax.random.split(key, 2)
    if sample_model:
        ens_mean = jax.random.choice(key=key, a=ens_mean, axis=0)
        ens_std = jax.random.choice(key=key, a=ens_std, axis=0)
    else:
        ens_mean = jnp.mean(ens_mean, axis=0)
        ens_std = jnp.mean(ens_std, axis=0)

    if predict_rewards:
        ens_mean = ens_mean[..., :-1]
        ens_std = ens_std[..., :-1]

    if internal_noise_std == 0.0:
        if internal_noise_samples > 1:
            next_state = jnp.broadcast_to(
                ens_mean[jnp.newaxis],
                (internal_noise_samples,) + ens_mean.shape,
            )
        else:
            next_state = ens_mean
    else:
        if internal_noise_std > 0.0:
            if internal_noise_samples > 1:
                noise_shape = (internal_noise_samples,) + ens_std.shape
                next_state = ens_mean[jnp.newaxis] + jax.random.normal(
                    noise_key, shape=noise_shape) * internal_noise_std
            else:
                next_state = ens_mean + jax.random.normal(
                    noise_key, shape=ens_std.shape) * internal_noise_std
        else:
            if internal_noise_samples > 1:
                noise_shape = (internal_noise_samples,) + ens_std.shape
                next_state = ens_mean[jnp.newaxis] + jax.random.normal(
                    noise_key, shape=noise_shape) * ens_std[jnp.newaxis]
            else:
                next_state = ens_mean + jax.random.normal(
                    noise_key, shape=ens_std.shape) * ens_std

    if predict_diff:
        if dt is not None:
            # CT case: The ensemble predicts the derivative of the next_state
            next_state = next_state * dt * action_repeat
        if internal_noise_samples > 1:
            next_state = next_state + batch.observations[jnp.newaxis]
        else:
            next_state = next_state + batch.observations

    if input_knowledge:
        if internal_noise_samples > 1:
            next_state = next_state + known_input_effect[jnp.newaxis]
        else:
            next_state = next_state + known_input_effect

    if internal_noise_samples > 1:
        def repeat_batch_field(x):
            repeated = jnp.broadcast_to(
                x[jnp.newaxis],
                (internal_noise_samples,) + x.shape,
            )
            return repeated.reshape((-1,) + x.shape[1:])

        imagined_batch = batch._replace(
            observations=repeat_batch_field(batch.observations),
            actions=repeat_batch_field(batch.actions),
            rewards=repeat_batch_field(batch.rewards),
            masks=repeat_batch_field(batch.masks),
            next_observations=next_state.reshape((-1,) + next_state.shape[2:]),
        )
    else:
        imagined_batch = batch._replace(next_observations=next_state)
    return imagined_batch


def _policy_actions_and_log_probs(actor: Model,
                                  actor_params: Params,
                                  observations: jnp.ndarray,
                                  key: PRNGKey,
                                  deterministic_policy: bool):
    dist = actor.apply_fn({'params': actor_params}, observations)
    if deterministic_policy:
        if hasattr(dist, 'bijector') and hasattr(dist, 'distribution'):
            actions = dist.bijector.forward(dist.distribution.mean())
        else:
            actions = dist.mean()
        log_probs = jnp.zeros(observations.shape[:-1])
    else:
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
    return actions, log_probs


@functools.partial(jax.jit, static_argnames=('actor_apply_fn',))
def _deterministic_policy_actions(actor_apply_fn,
                                  actor_params: Params,
                                  observations: np.ndarray) -> jnp.ndarray:
    dist = actor_apply_fn({'params': actor_params}, observations)
    if hasattr(dist, 'bijector') and hasattr(dist, 'distribution'):
        return dist.bijector.forward(dist.distribution.mean())
    return dist.mean()


def _ensemble_input(observations: jnp.ndarray,
                    actions: jnp.ndarray,
                    input_knowledge: bool) -> jnp.ndarray:
    if input_knowledge:
        return observations
    return jnp.concatenate([observations, actions], axis=-1)


def update_actor_local(key: PRNGKey,
                       actor: Model,
                       critic: Model,
                       temp: Model,
                       target_actor: Model,
                       dyn_entropy_temp: Model,
                       ens: DeterministicEnsemble,
                       ens_state: EnsembleState,
                       batch: Batch,
                       deterministic_policy: bool,
                       use_action_entropy: bool,
                       input_knowledge: bool) -> Tuple[Model, EnsembleState, InfoDict]:
    key, target_key = jax.random.split(key, 2)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[EnsembleState, InfoDict]]:
        actions, log_probs = _policy_actions_and_log_probs(
            actor=actor,
            actor_params=actor_params,
            observations=batch.observations,
            key=key,
            deterministic_policy=deterministic_policy,
        )
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)

        target_actions, _ = _policy_actions_and_log_probs(
            actor=target_actor,
            actor_params=target_actor.params,
            observations=batch.observations,
            key=target_key,
            deterministic_policy=deterministic_policy,
        )
        target_inp = _ensemble_input(batch.observations, target_actions, input_knowledge)
        inp = _ensemble_input(batch.observations, actions, input_knowledge)
        total_inp = jnp.concatenate([inp, target_inp], axis=0)
        info_gain, new_ens_state = ens.get_info_gain(input=total_inp,
                                                     state=ens_state,
                                                     update_normalizer=True)
        info_gain, target_info_gain = info_gain[:actions.shape[0]], info_gain[actions.shape[0]:]
        dyn_ent_coef, _ = dyn_entropy_temp()
        act_ent_coef, _ = temp()
        total_entropy = dyn_ent_coef * info_gain
        if use_action_entropy:
            total_entropy = total_entropy - act_ent_coef * log_probs
        actor_loss = -(total_entropy + q).mean()
        return actor_loss, (new_ens_state, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'info_gain': info_gain.mean(),
            'target_info_gain': target_info_gain.mean(),
        })

    new_actor, (new_ens_state, info) = actor.apply_gradient(actor_loss_fn)

    return new_actor, new_ens_state, info


def update_critic_local(key: PRNGKey,
                        actor: Model,
                        critic: Model,
                        target_critic: Model,
                        temp: Model,
                        dyn_entropy_temp: Model,
                        ens: DeterministicEnsemble,
                        ens_state: EnsembleState,
                        batch: Batch,
                        discount: float,
                        backup_entropy: bool,
                        deterministic_policy: bool,
                        use_action_entropy: bool,
                        input_knowledge: bool) -> Tuple[Model, EnsembleState, InfoDict]:
    next_actions, next_log_probs = _policy_actions_and_log_probs(
        actor=actor,
        actor_params=actor.params,
        observations=batch.next_observations,
        key=key,
        deterministic_policy=deterministic_policy,
    )

    info_gain, new_ens_state = ens.get_info_gain(
        input=_ensemble_input(batch.next_observations, next_actions, input_knowledge),
        state=ens_state, update_normalizer=False)

    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        dyn_ent_coef, _ = dyn_entropy_temp()
        act_ent_coef, _ = temp()
        total_entropy = dyn_ent_coef * info_gain
        if use_action_entropy:
            total_entropy = total_entropy - act_ent_coef * next_log_probs
        target_q += discount * batch.masks * total_entropy

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean()
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, new_ens_state, info


@functools.partial(jax.jit,
                   static_argnames=('ens',
                                    'backup_entropy',
                                    'update_target',
                                    'use_log_transform',
                                    'predict_rewards',
                                    'predict_diff',
                                    'sample_model',
                                    'update_critic_with_real_data',
                                    'update_policy',
                                    'internal_noise_std',
                                    'internal_noise_samples',
                                    'deterministic_policy',
                                    'use_action_entropy',
                                    'input_knowledge',
                                    ))
def _update_jit(
        rng: PRNGKey, actor: Model, critic: Model, target_actor: Model, target_critic: Model, temp: Model, # type: ignore
        dyn_entropy_temp: Model, ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: Batch, discount: float, tau: float,
        target_entropy: float, backup_entropy: bool, update_target: bool,
        use_log_transform: bool, predict_rewards: bool, predict_diff: bool,
        sample_model: bool, update_critic_with_real_data: bool, update_policy: bool,
        internal_noise_std: float, internal_noise_samples: int, dt: float, action_repeat: int,
        deterministic_policy: bool, use_action_entropy: bool,
        known_input_effect: Optional[jnp.ndarray], input_knowledge: bool,
) -> Tuple[PRNGKey, Model, Model, Model, Model, Model, Model, EnsembleState, InfoDict]: # type: ignore
    rng, key = jax.random.split(rng)
    if update_critic_with_real_data:
        new_critic, ens_state, critic_info = update_critic_local(
            key=key,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            dyn_entropy_temp=dyn_entropy_temp,
            ens=ens,
            ens_state=ens_state,
            batch=batch,
            discount=discount,
            backup_entropy=backup_entropy,
            deterministic_policy=deterministic_policy,
            use_action_entropy=use_action_entropy,
            input_knowledge=input_knowledge,
        )
    else:
        new_critic = critic
        critic_info = {}

    rng, model_sample_key = jax.random.split(rng)
    imagined_batch = get_imagined_batch(
        batch=batch,
        ens_state=ens_state,
        ens=ens,
        known_input_effect=known_input_effect,
        input_knowledge=input_knowledge,
        predict_diff=predict_diff,
        predict_rewards=predict_rewards,
        sample_model=sample_model,
        internal_noise_std=internal_noise_std,
        internal_noise_samples=internal_noise_samples,
        key=model_sample_key,
        dt=dt,
        action_repeat=action_repeat,
    )
    rng, key = jax.random.split(rng)
    new_critic, ens_state, imagined_critic_info = update_critic_local(
        key=key,
        actor=actor,
        critic=new_critic,
        target_critic=target_critic,
        temp=temp,
        dyn_entropy_temp=dyn_entropy_temp,
        ens=ens,
        ens_state=ens_state,
        batch=imagined_batch,
        discount=discount,
        backup_entropy=backup_entropy,
        deterministic_policy=deterministic_policy,
        use_action_entropy=use_action_entropy,
        input_knowledge=input_knowledge,
    )

    imagined_critic_info = {f'imagined_critic_{key}': val for key, val in imagined_critic_info.items()}

    if update_target:
        new_target_critic = target_update(new_critic, target_critic, tau)
    else:
        new_target_critic = target_critic

    if update_policy:
        rng, key = jax.random.split(rng)
        new_actor, ens_state, actor_info = update_actor_local(key=key,
                                                              actor=actor,
                                                              target_actor=target_actor,
                                                              critic=new_critic,
                                                              temp=temp,
                                                              dyn_entropy_temp=dyn_entropy_temp,
                                                              ens=ens,
                                                              ens_state=ens_state,
                                                              batch=batch,
                                                              deterministic_policy=deterministic_policy,
                                                              use_action_entropy=use_action_entropy,
                                                              input_knowledge=input_knowledge,
                                                              )
        if update_target:
            new_target_actor = target_update(new_actor, target_actor, tau)
        else:
            new_target_actor = target_actor

        if use_action_entropy:
            new_temp, alpha_info = update_temp(temp, actor_info['entropy'],
                                               target_entropy, use_log_transform=use_log_transform)
        else:
            new_temp, alpha_info = temp, {}
        new_dyn_entropy_temp, dyn_ent_info = update_temp(dyn_entropy_temp, actor_info['info_gain'],
                                                         actor_info['target_info_gain'],
                                                         use_log_transform=use_log_transform)
        dyn_ent_info = {f'dyn_ent_{key}': val for key, val in dyn_ent_info.items()}
    else:
        new_actor, new_temp, new_dyn_entropy_temp, new_target_actor = actor, temp, dyn_entropy_temp, target_actor
        actor_info, alpha_info, dyn_ent_info = {}, {}, {}

    if predict_diff:
        outputs = batch.next_observations - batch.observations
        if input_knowledge:
            outputs = outputs - known_input_effect
        if dt is not None:
            outputs = outputs / (dt * action_repeat)
    else:
        outputs = batch.next_observations
        if input_knowledge:
            outputs = outputs - known_input_effect
    if predict_rewards:
        outputs = jnp.concatenate([outputs, batch.rewards.reshape(-1, 1)], axis=-1)
    new_ens_state, (loss, mse) = ens.update(
        input=_ensemble_input(batch.observations, batch.actions, input_knowledge),
        output=outputs,
        state=ens_state,
    )
    ens_info = {'ens_nll': loss,
                'ens_mse': mse,
                'ens_inp_mean': ens_state.ensemble_normalizer_state.input_normalizer_state.mean.mean(),
                'ens_inp_std': ens_state.ensemble_normalizer_state.input_normalizer_state.std.mean(),
                # 'ens_inp_num_points': ens_state.ensemble_normalizer_state.input_normalizer_state.num_points,
                'ens_out_mean': ens_state.ensemble_normalizer_state.output_normalizer_state.mean.mean(),
                'ens_out_std': ens_state.ensemble_normalizer_state.output_normalizer_state.std.mean(),
                # 'ens_out_num_points': ens_state.ensemble_normalizer_state.output_normalizer_state.num_points,
                'ens_info_gain_mean': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.mean.mean(),
                'ens_info_gain_std': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.std.mean(),
                # 'ens_info_gain_num_points': ens_state.ensemble_normalizer_state.info_gain_normalizer_state.num_points,
                }

    return rng, \
        new_actor, \
        new_critic, \
        new_target_actor, \
        new_target_critic, \
        new_temp, \
        new_dyn_entropy_temp, \
        new_ens_state, {
        **critic_info,
        **imagined_critic_info,
        **actor_info,
        **alpha_info,
        **dyn_ent_info,
        **ens_info,
    }


class MaxInfoOmbrlLearner(object):
    def __init__(self,
                 seed: int,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 dyn_ent_lr: float = 3e-4,
                 dyn_wd: float = 0.0,
                 ens_lr: float = 3e-4,
                 ens_wd: float = 0.0,
                 hidden_dims: Sequence[int] = (256, 256),
                 model_hidden_dims: Sequence[int] = (256, 256),
                 num_heads: int = 5,
                 predict_reward: bool = True,
                 predict_diff: bool = True,
                 use_log_transform: bool = True,
                 learn_std: bool = False,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 target_update_period: int = 1,
                 target_entropy: Optional[float] = None,
                 backup_entropy: bool = True,
                 init_temperature: float = 1.0,
                 init_temperature_dyn_entropy: float = 1.0,
                 init_mean: Optional[np.ndarray] = None,
                 policy_final_fc_init_scale: float = 1.0,
                 sample_model: bool = True,
                 internal_noise_std: Optional[float] = None,
                 internal_noise_samples: int = 1,
                 critic_real_data_update_period: int = 2,
                 policy_update_period: Optional[int] = None,
                 max_gradient_norm: Optional[float] = None,
                 use_bronet: bool = False,
                 reset_period: Optional[int] = None,
                 reset_models: bool = False,
                 perturb_rate: float = 0.2,
                 perturb_policy: bool = True,
                 perturb_model: bool = True,
                 deterministic_policy: bool = False,
                 deterministic_train_actions: bool = False,
                 use_action_entropy: bool = True,
                 pseudo_ct: bool = False,
                 dt: float = None,
                 action_repeat: int = None,
                 input_knowledge: bool = False,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.predict_reward = predict_reward
        self.predict_diff = predict_diff
        self.num_heads = num_heads
        self.sample_model = sample_model
        self.internal_noise_std = -1.0 if internal_noise_std is None else internal_noise_std
        self.internal_noise_samples = internal_noise_samples
        self.deterministic_policy = deterministic_policy
        self.deterministic_train_actions = deterministic_train_actions
        self.use_action_entropy = use_action_entropy
        self.input_knowledge = input_knowledge
        self.critic_real_data_update_period = critic_real_data_update_period
        self.perturb_rate = perturb_rate
        if policy_update_period:
            self.policy_update_period = policy_update_period
        else:
            self.policy_update_period = critic_real_data_update_period

        action_dim = actions.shape[-1]

        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy

        self.backup_entropy = backup_entropy

        self.tau = tau
        self.target_update_period = target_update_period
        self.discount = discount

        if reset_period is None:
            self.reset_period = 2_500_000
        else:
            self.reset_period = reset_period

        self._reset_models = reset_models

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)
        actor_def = policies.NormalTanhPolicy(
            hidden_dims,
            action_dim,
            init_mean=init_mean,
            final_fc_init_scale=policy_final_fc_init_scale)
        actor_optimizer = optax.adam(learning_rate=actor_lr)
        critic_optimizer = optax.adam(learning_rate=critic_lr)
        temp_optimizer = optax.adam(learning_rate=temp_lr)
        dyn_ent_temp_optimizer = optax.adamw(learning_rate=dyn_ent_lr, weight_decay=dyn_wd)
        model_optimizer = optax.adamw(learning_rate=ens_lr, weight_decay=ens_wd)
        if max_gradient_norm:
            assert max_gradient_norm > 0
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),  # Apply gradient clipping
                actor_optimizer  # Apply Adam optimizer
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                critic_optimizer,
            )

            temp_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                temp_optimizer,
            )

            dyn_ent_temp_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                dyn_ent_temp_optimizer,
            )
            model_optimizer = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                model_optimizer,
            )

        actor = Model.create(actor_def,
                             inputs=[actor_key, observations],
                             tx=actor_optimizer)

        target_actor = Model.create(actor_def,
                                    inputs=[actor_key, observations])

        critic_def = critic_net.DoubleCritic(hidden_dims, use_bronet=use_bronet)
        critic = Model.create(critic_def,
                              inputs=[critic_key, observations, actions],
                              tx=critic_optimizer)
        target_critic = Model.create(
            critic_def, inputs=[critic_key, observations, actions])


        temp = Model.create(temperature.Temperature(init_temperature),
                            inputs=[temp_key],
                            tx=temp_optimizer)

        # information gain kwargs
        dyn_ent_temp_key, rng = jax.random.split(rng, 2)
        dyn_ent_temp = Model.create(temperature.Temperature(init_temperature_dyn_entropy),
                                    inputs=[dyn_ent_temp_key],
                                    tx=dyn_ent_temp_optimizer)

        model_key, rng = jax.random.split(rng, 2)

        output_dim = observations.shape[-1]
        if predict_reward:
            output_dim += 1

        if learn_std:
            model_type = ProbabilisticEnsemble
        else:
            model_type = DeterministicEnsemble
        ensemble = model_type(
            model_kwargs={'hidden_dims': model_hidden_dims + (output_dim,)},
            optimizer=model_optimizer,
            num_heads=self.num_heads,
            use_entropy_for_int_rew=False,  # return model epistemic uncertainty as the intrinsic rew
        )

        if self.input_knowledge:
            ensemble_init_input = observations
        else:
            ensemble_init_input = jnp.concatenate([observations, actions], axis=-1)
        ens_state = ensemble.init(key=model_key, input=ensemble_init_input)

        self.perturb_module = PerturbationModule(
            actor_init_fn=actor_def.init,
            critic_init_fn=critic_def.init,
            model_init_fn=ensemble.init,
            actor_init_opt_state=copy.deepcopy(actor.opt_state),
            critic_init_opt_state=copy.deepcopy(critic.opt_state),
            perturb_rate=perturb_rate,
            perturbation_freq=self.reset_period,
            perturb_policy=perturb_policy,
            perturb_model=perturb_model,
            model_input_uses_actions=not self.input_knowledge,
        )

        self.use_log_transform = use_log_transform

        self.actor = actor
        self.target_actor = target_actor
        self.critic = critic
        self.target_critic = target_critic
        self.temp = temp
        self.dyn_ent_temp = dyn_ent_temp
        self.ens_state = ens_state
        self.ensemble = ensemble
        self.rng = rng

        self.step = 1
        if dt is not None:
            assert pseudo_ct == True, f"continuous-time must be enabled for given dt, got: {pseudo_ct}"
            assert predict_diff == True, \
            f"predict_diff should be True in the pseudo-ct case, got: {predict_diff} for dt={dt}"
        else:
            assert pseudo_ct == False, f"continuous-time must be disabled for given dt, got: {pseudo_ct}"
        self.dt = dt
        self.action_repeat = action_repeat

    def _should_perturb(self) -> bool:
        return self.step >= 1 and self.step % self.reset_period == 0

    def sample_actions(self,
                       observations: np.ndarray,
                       temperature: float = 1.0) -> np.ndarray:
        if self.deterministic_train_actions:
            actions = _deterministic_policy_actions(
                self.actor.apply_fn,
                self.actor.params,
                observations,
            )
        else:
            rng, actions = policies.sample_actions(self.rng, self.actor.apply_fn,
                                                   self.actor.params, observations,
                                                   temperature)
            self.rng = rng

        actions = np.asarray(actions)
        return np.clip(actions, -1, 1)

    def update(self, batch: Batch, known_input_effect: Optional[np.ndarray] = None) -> InfoDict:
        if self.input_knowledge and known_input_effect is None:
            raise ValueError("known_input_effect must be provided when input_knowledge=True")
        if not self.input_knowledge:
            known_input_effect = None

        if self._reset_models:
            rng, self.rng = jax.random.split(self.rng)
            if self._should_perturb():
                actor, critic, target_actor, target_critic, new_ens_state = self.perturb_module.perturb(
                    actor=self.actor,
                    critic=self.critic,
                    target_actor=self.target_actor,
                    target_critic=self.target_critic,
                    ens_state=self.ens_state,
                    observation=batch.observations,
                    action=batch.actions,
                    rng=rng,
                    step=self.step
                )
                self.actor = actor
                self.target_actor = target_actor
                self.critic = critic
                self.target_critic = target_critic
                self.ens_state = new_ens_state

        self.step += 1
        new_rng, new_actor, new_critic, new_target_actor, new_target_critic, \
            new_temp, new_dyn_entropy_temp, new_ens_state, info = _update_jit(
            rng=self.rng,
            actor=self.actor,
            critic=self.critic,
            target_actor=self.target_actor,
            target_critic=self.target_critic,
            temp=self.temp,
            dyn_entropy_temp=self.dyn_ent_temp,
            ens=self.ensemble,
            ens_state=self.ens_state,
            batch=batch,
            discount=self.discount,
            tau=self.tau,
            target_entropy=self.target_entropy,
            backup_entropy=self.backup_entropy,
            update_target=self.step % self.target_update_period == 0,
            use_log_transform=self.use_log_transform,
            predict_rewards=self.predict_reward,
            predict_diff=self.predict_diff,
            sample_model=self.sample_model,
            update_critic_with_real_data=self.step % self.critic_real_data_update_period == 0,
            update_policy=self.step % self.policy_update_period == 0,
            internal_noise_std=self.internal_noise_std,
            internal_noise_samples=self.internal_noise_samples,
            dt=self.dt,
            action_repeat=self.action_repeat,
            deterministic_policy=self.deterministic_policy,
            use_action_entropy=self.use_action_entropy,
            known_input_effect=known_input_effect,
            input_knowledge=self.input_knowledge,
        )

        self.rng = new_rng
        self.actor = new_actor
        self.critic = new_critic
        self.target_actor = new_target_actor
        self.target_critic = new_target_critic
        self.temp = new_temp
        self.dyn_ent_temp = new_dyn_entropy_temp
        self.ens_state = new_ens_state

        return info
