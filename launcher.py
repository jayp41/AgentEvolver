# from best_logger import print_dict
import subprocess
import argparse
import shutil
import time
import sys
import os
from dotenv import load_dotenv
from beyondagent.utils.daemon import LaunchCommandWhenAbsent

load_dotenv()
BACK_TARGETS = os.environ.get('BACK_TARGETS', './config,./beyondagent').split(',')

def parse_args():
    parser = argparse.ArgumentParser(description='The launcher of agentevolver.')
    parser.add_argument(
        '--target',
        type=str,
        default='beyondagent.main_ppo',
        required=False,
        help='Target script to run (default: beyondagent.main_ppo)'
    )
    parser.add_argument('--conf',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--db',
        type=str,
        default="",
        required=False,
        help='Path to configuration file'
    )
    parser.add_argument('--with-appworld',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch appworld'
    )
    parser.add_argument('--with-webshop',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch webshop'
    )
    parser.add_argument('--with-bfcl',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch bfcl'
    )
    parser.add_argument('--with-logview',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch logview'
    )
    parser.add_argument('--with-crafters',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='Launch Crafters Env Simulation'
    )
    parser.add_argument('--reboot',
        action='store_true',  # Changed from store_true to action='store_true'
        default=False,
        help='reboot flag'
    )

    return parser.parse_args()


def pty_launch(service_name: str):
    service_path = os.environ.get(f'{service_name.upper()}_PATH')
    service_script = os.environ.get(f'{service_name.upper()}_SCRIPT')
    companion = LaunchCommandWhenAbsent(
        full_argument_list=[service_script],
        dir=service_path,
        tag="appworld_env_service",
        use_pty=True
    )
    companion.launch(
        launch_wait_time=1800,
        success_std_string="Starting server on",
    )
def main():
    args = parse_args()

    if args.conf:
        yaml_path = args.conf
        assert yaml_path.endswith('.yaml'), "Configuration file must be a YAML file"
        exp_base = os.path.dirname(args.conf)

        if os.path.exists(exp_base):

            ## 0. read yaml (get trainer.experiment_name)
            import yaml
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            exp_name = config.get('trainer').get('experiment_name')
            if exp_name is None or exp_name == 'read_yaml_name':
                if exp_name is not None: exp_name = exp_name.replace('|', '-')
                exp_name = os.path.basename(yaml_path).replace('.yaml', '')
            else:
                exp_name = exp_name.replace('|', '-')

            print('----------------------------------------')
            backup_dir = os.path.join('launcher_record', exp_name, 'backup')
            yaml_backup_dst = os.path.join('launcher_record', exp_name, 'yaml_backup.yaml')
            exe_yaml_path = yaml_backup_dst
            exe_exp_base = os.path.dirname(yaml_backup_dst)
            print('Experiment Name:', exp_name)
            print('Experiment Backup Dir:', backup_dir)
            print('Experiment Yaml Dir:', yaml_backup_dst)
            print('----------------------------------------')
            time.sleep(2)

            ## 1. check exp_base/backup exist
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            else:
                total_seconds = 10
                for i in range(total_seconds):
                    print(f"\rWarning: backup directory already exists, we will automatically ignore this after {total_seconds - i} seconds...", end="", flush=True)
                    time.sleep(1)

            ## 2. copy files to backup
            for backup_target in BACK_TARGETS:
                print(f"Copying {backup_target} to {os.path.join(backup_dir, os.path.basename(backup_target))}")
                shutil.copytree(backup_target, os.path.join(backup_dir, os.path.basename(backup_target)), dirs_exist_ok=True)

            ## 3. copy yaml to backup
            yaml_backup_src = yaml_path
            shutil.copyfile(yaml_backup_src, yaml_backup_dst)

            ## 4. edit new yaml
            yaml_path = yaml_backup_dst
            # now, replace the trainer.experiment_name
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            config['trainer']['experiment_name'] = exp_name
            with open(yaml_path, 'w') as file:
                yaml.dump(config, file)

        else:
            raise FileNotFoundError(f"Configuration file not found: {exp_base}")

        env = os.environ.copy()
        if args.db:
            env["RAY_DEBUG_POST_MORTEM"] = "1"
            env["DEBUG_TAGS"] = args.db
            env["RAY_record_task_actor_creation_sites"] =  "true"
            print("Debug mode is ON")
        else:
            print("Debug mode is OFF")

    if args.with_appworld:
        # test done
        pty_launch("appworld")

    if args.with_crafters:
        # test done
        pty_launch("crafters")

    if args.with_webshop:
        # not tesed
        pty_launch("webshop")

    if args.with_bfcl:
        pty_launch("bfcl")

    if args.with_logview:

        companion = LaunchCommandWhenAbsent(
            full_argument_list=[
                sys.executable,
                '-m',
                'web_display.start_web',
            ],
            dir='./',
            tag="logview"
        )
        companion.launch(launch_wait_time=1800,success_std_string="Uvicorn running on", env_dict={})

    if args.conf:
        # let's begin the training process
        cmd = [
            sys.executable,
            '-m',
            args.target,
            '--config-path',
            os.path.abspath(exe_exp_base),
            '--config-name',
            os.path.basename(exe_yaml_path),
        ]

        if args.with_logview:
            env.update({
                'BEST_LOGGER_WEB_SERVICE_URL': os.environ.get('BEST_LOGGER_WEB_SERVICE_URL', 'http://127.0.0.1:8181/')
            })

        try:
            print(f"Running command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=os.path.abspath('./'), env=env)
        except subprocess.CalledProcessError as e:
            print(f"Error running subprocess: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()