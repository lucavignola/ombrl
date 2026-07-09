import argparse
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.maxinfombsac import experiment as exp
from experiments.utils import dict_permutations, generate_base_command, generate_run_commands


PROJECT_NAME = 'SOMBRL_Fig3_State_MBPO'
ENTITY = 'lvignola-eth-z-rich'

COMMON = {
    'batch_size': [256],
    'seed': list(range(5)),
    'wandb_log': [1],
    'project_name': [PROJECT_NAME],
    'entity_name': [ENTITY],
    'use_tqdm': [0],
    'alg_name': ['maxinfombsac'],
    'ens_lr': [3e-4],
    'lr': [3e-4],
    'ens_wd': [0.0],
    'critic_real_data_update_period': [5],
    'use_bronet': [1],
    'num_hidden_layers': [2],
    'pseudo_ct': [0],
    'predict_diff': [1],
    'eval_episodes': [10],
    'perturb_model': [1],
    'perturb_policy': [0],
}

MBPO_OPTIMISTIC = {
    'exp_hash': ['mbpo_optimistic'],
    'sample_model': [0],
    'dyn_ent_lr': [3e-4],
    'init_temperature_dyn_entropy': [1.0],
} | COMMON

MBPO_MEAN = {
    'exp_hash': ['mbpo_mean'],
    'sample_model': [0],
    'dyn_ent_lr': [0.0],
    'init_temperature_dyn_entropy': [1e-8],
} | COMMON

MOUNTAIN_CAR = {
    'env_name': ['MountainCarContinuous-v0'],
    'max_steps': [40_000],
    'eval_interval': [1_000],
    'action_repeat': [1],
    'hidden_dims': [256],
}

CARTPOLE = {
    'env_name': ['cartpole-swingup_sparse'],
    'max_steps': [400_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'action_cost': [0.0],
    'hidden_dims': [256],
}

HOPPER = {
    'env_name': ['hopper-hop'],
    'max_steps': [1_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'hidden_dims': [256],
}

QUADRUPED = {
    'env_name': ['quadruped-run'],
    'max_steps': [3_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'hidden_dims': [512],
}

HUMANOID = {
    'env_name': ['humanoid-stand', 'humanoid-walk'],
    'max_steps': [3_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'hidden_dims': [512],
}

TASKS = [MOUNTAIN_CAR, CARTPOLE, HOPPER, QUADRUPED, HUMANOID]


def build_flags(project_name=PROJECT_NAME, entity_name=ENTITY):
    flags = []
    for task in TASKS:
        for alg in [MBPO_OPTIMISTIC, MBPO_MEAN]:
            task_flags = task | alg
            task_flags['project_name'] = [project_name]
            task_flags['entity_name'] = [entity_name]
            flags.extend(dict_permutations(task_flags))
    return flags


def main(args):
    command_list = []
    logs_dir = args.logs_dir
    if args.mode == 'euler' and logs_dir is None:
        logs_dir = f'/cluster/scratch/lvignola/{PROJECT_NAME}/'
    elif logs_dir is None:
        logs_dir = os.path.abspath('./logs/sombrl_fig3_state/')

    setup_prefix = ''
    if args.euler_setup:
        setup_prefix = f'. {os.path.abspath(args.euler_setup)} && '

    for flags in build_flags(args.project_name, args.entity_name):
        flags['logs_dir'] = logs_dir
        cmd = setup_prefix + generate_base_command(exp, flags=flags)
        if args.mode == 'euler':
            cmd = f'bash -lc {shlex.quote(cmd)}'
        command_list.append(cmd)

    num_hours = args.hours
    generate_run_commands(command_list,
                          num_cpus=args.num_cpus,
                          num_gpus=args.num_gpus,
                          mode=args.mode,
                          duration=f'{num_hours}:59:00',
                          prompt=not args.yes,
                          dry=args.dry_run,
                          mem=args.mem_per_cpu,
                          gpu_type=args.gpu_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launch SOMBRL Figure 3 state MBPO jobs')
    parser.add_argument('--num_cpus', type=int, default=10)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--mode', type=str, default='euler', choices=['euler', 'local', 'local_async'])
    parser.add_argument('--hours', type=int, default=23)
    parser.add_argument('--mem_per_cpu', type=int, default=10240)
    parser.add_argument('--gpu_type', type=str, default='rtx_4090')
    parser.add_argument('--project_name', type=str, default=PROJECT_NAME)
    parser.add_argument('--entity_name', type=str, default=ENTITY)
    parser.add_argument('--logs_dir', type=str, default=None)
    parser.add_argument('--euler_setup', type=str, default='utility_scripts/setup_sombrl_euler.bash')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--yes', action='store_true')
    main(parser.parse_args())
