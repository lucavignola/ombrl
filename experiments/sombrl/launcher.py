import argparse
import ast
import itertools
import os
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PROJECT_NAME = 'SOMBRL_Fig3_State_MBPO'
ENTITY = 'lvignola-eth-z-rich'
EXP_FILE = REPO_ROOT / 'experiments' / 'maxinfombsac' / 'experiment.py'

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
    'process_noise_std': [0],
    'internal_noise_std': [0],
    'internal_noise_samples': [1],
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
    'use_dynamics_entropy': [0],
} | COMMON

MBPO_GREEDY = {
    'exp_hash': ['greedy'],
    'sample_model': [0],
    'dyn_ent_lr': [0.0],
    'temp_lr': [0.0],
    'init_temperature_dyn_entropy': [1e-8],
    'init_temperature': [1e-8],
    'deterministic_policy': [1],
    'deterministic_train_actions': [1],
    'use_action_entropy': [0],
    'use_dynamics_entropy': [0],
} | COMMON

MOUNTAIN_CAR = {
    'env_name': ['MountainCarContinuous-v0'],
    'max_steps': [40_000],
    'eval_interval': [1_000],
    'action_repeat': [1],
    'num_neurons': [256],
}

CARTPOLE = {
    'env_name': ['cartpole-swingup_sparse'],
    'max_steps': [400_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'action_cost': [0.0],
    'num_neurons': [256],
}

HOPPER = {
    'env_name': ['hopper-hop'],
    'max_steps': [1_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'num_neurons': [256],
}

QUADRUPED = {
    'env_name': ['quadruped-run'],
    'max_steps': [3_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'num_neurons': [512],
}

HUMANOID = {
    'env_name': ['humanoid-stand', 'humanoid-walk'],
    'max_steps': [3_000_000],
    'eval_interval': [10_000],
    'action_repeat': [2],
    'num_neurons': [512],
}

TASKS = [MOUNTAIN_CAR, CARTPOLE, HOPPER, QUADRUPED, HUMANOID]


def dict_permutations(d):
    keys = d.keys()
    return [dict(zip(keys, values)) for values in itertools.product(*d.values())]


def generate_base_command(flags=None, interpreter=None):
    if interpreter is None:
        interpreter = sys.executable
    cmd = f'{interpreter} -u {EXP_FILE}'
    if flags is not None:
        for flag, setting in flags.items():
            if isinstance(setting, bool):
                if setting:
                    cmd += f' --{flag}'
            else:
                cmd += f' --{flag}={setting}'
    return cmd


def generate_run_commands(command_list, num_cpus=1, num_gpus=0, dry=False,
                          mem=2 * 1028, duration='3:59:00', mode='local',
                          prompt=True, gpu_type=None):
    if mode == 'euler':
        base = (
            f'sbatch --time={duration} --mem-per-cpu={mem} '
            f'--cpus-per-task {num_cpus} --account=ls_krausea '
        )
        if num_gpus > 0:
            if gpu_type is None:
                base += f'-G {num_gpus} --gres=gpumem:10240m '
            else:
                base += f'--gpus={gpu_type}:{num_gpus} '

        cluster_cmds = [base + f'--wrap={shlex.quote(cmd)}' for cmd in command_list]
        if dry:
            for cmd in cluster_cmds:
                print(cmd)
            return

        answer = 'yes'
        if prompt:
            answer = input(f'about to launch {len(command_list)} jobs with {num_cpus} cores each. proceed? [yes/no]')
        if answer == 'yes':
            for cmd in cluster_cmds:
                os.system(cmd)
        return

    if mode == 'local':
        answer = 'yes'
        if prompt:
            answer = input(f'about to run {len(command_list)} jobs in a loop. proceed? [yes/no]')
        if answer == 'yes':
            for cmd in command_list:
                if dry:
                    print(cmd)
                else:
                    os.system(cmd)
        return

    raise NotImplementedError(f'Unsupported mode: {mode}')


def build_flags(project_name=PROJECT_NAME, entity_name=ENTITY, input_knowledge=False,
                cache_input_effects=True):
    flags = []
    for task in TASKS:
        for alg in [MBPO_OPTIMISTIC, MBPO_MEAN, MBPO_GREEDY]:
            task_flags = task | alg
            task_flags['project_name'] = [project_name]
            task_flags['entity_name'] = [entity_name]
            task_flags['input_knowledge'] = [int(input_knowledge)]
            task_flags['cache_input_effects'] = [int(cache_input_effects)]
            if input_knowledge:
                task_flags['exp_hash'] = [f"{task_flags['exp_hash'][0]}_input_knowledge"]
            flags.extend(dict_permutations(task_flags))
    return flags


def validate_experiment_flags(flags):
    parser_args = set()
    with open(EXP_FILE, 'r') as f:
        tree = ast.parse(f.read(), filename=str(EXP_FILE))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != 'add_argument':
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        arg_name = node.args[0].value
        if isinstance(arg_name, str) and arg_name.startswith('--'):
            parser_args.add(arg_name[2:].replace('-', '_'))

    invalid = sorted(set().union(*(flag.keys() for flag in flags)) - parser_args)
    if invalid:
        raise ValueError(
            f'Launcher generated flags not accepted by {EXP_FILE}: {invalid}'
        )


def main(args):
    command_list = []
    logs_dir = args.logs_dir
    if args.mode == 'euler' and logs_dir is None:
        logs_dir = f'/cluster/scratch/lvignola/{PROJECT_NAME}/'
    elif logs_dir is None:
        logs_dir = os.path.abspath('./logs/sombrl_fig3_state/')

    setup_prefix = ''
    if args.mode == 'euler' and args.euler_setup:
        setup_prefix = f'. {os.path.abspath(args.euler_setup)} && '

    all_flags = build_flags(
        args.project_name,
        args.entity_name,
        args.input_knowledge,
        args.cache_input_effects,
    )
    validate_experiment_flags(all_flags)

    for flags in all_flags:
        flags['logs_dir'] = logs_dir
        interpreter = 'python' if args.mode == 'euler' else None
        cmd = setup_prefix + generate_base_command(flags=flags, interpreter=interpreter)
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
    parser.add_argument('--input_knowledge', action='store_true')
    parser.add_argument('--cache_input_effects', type=int, default=1)
    main(parser.parse_args())
