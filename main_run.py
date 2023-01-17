import argparse
import os
from datetime import datetime

import copy
import os
from typing import List

from main_continual import dict_to_list
from main_continual import str_to_dict

parser = argparse.ArgumentParser()
parser.add_argument("--script", type=str, required=True)
parser.add_argument("--mode", type=str, default="normal")
parser.add_argument("--experiment_dir", type=str, default=None)
parser.add_argument("--base_experiment_dir", type=str, default="./experiments")
parser.add_argument("--gpu", type=str, default="v100-16g")
parser.add_argument("--num_gpus", type=int, default=2)
parser.add_argument("--hours", type=int, default=20)
parser.add_argument("--requeue", type=int, default=0)

args, _ = parser.parse_known_args()

def check_and_replace_required_args(element: str):
    if "$" in element:
        element = element.replace("$DATA_DIR", os.environ.get("DATA_DIR", "./data"))
        # element = element.replace("$EXP_DIR", os.environ.get("EXP_DIR", None))
    return element

# load file
if os.path.exists(args.script):
    with open(args.script) as f:
        command = [
            check_and_replace_required_args(line.strip().strip("\\").strip()) 
                for line in f.readlines()
        ]
else:
    print(f"{args.script} does not exist.")
    exit()

assert (
    "--checkpoint_dir" not in command
), "Please remove the --checkpoint_dir argument, it will be added automatically"

# collect args
command_args = str_to_dict(" ".join(command).split(" ")[2:])

# create experiment directory
if args.experiment_dir is None:
    args.experiment_dir = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.experiment_dir += f"-{command_args['--name']}"
full_experiment_dir = os.path.join(args.base_experiment_dir, args.experiment_dir)
os.makedirs(full_experiment_dir, exist_ok=True)
print(f"Experiment directory: {full_experiment_dir}")

# add experiment directory to the command
command.extend(["--checkpoint_dir", full_experiment_dir])

###### Start main continual experiment ######

def command_to_list_args(args: List[str]):
    """
    Split a list of string then check if the firt element starts with -- and group the second element in a list of strings
    """
    args_list = []
    for a in args:
        if a.startswith("--"):
            v = a.split(" ")
            if len(v) >= 2:
                args_list.extend([v[0], " ".join(v[1:])])
            else:
                args_list.append(a)
        else:
            args_list.append(a)
    return args_list

args = command_to_list_args(command)[1:]
args = str_to_dict(args)

# parse args from the script
semi_args = {k: v for k, v in args.items() if "semi" in k}

# delete things that shouldn't be used for task_idx 0
for k in semi_args.keys():
    args.pop(k, None)

# check if this experiment is being resumed: ssl
# look for the file last_checkpoint.txt
last_checkpoint_file = os.path.join(args["--checkpoint_dir"], "last_checkpoint.txt")
if os.path.exists(last_checkpoint_file):
    with open(last_checkpoint_file) as f:
        ckpt_path, args_path = [line.rstrip() for line in f.readlines()]
        args["--resume_from_checkpoint"] = ckpt_path

task_args = copy.deepcopy(args)

task_args = dict_to_list(task_args)

print(command)

from main_pretrain import main

def command_to_list_args_v2(args: List[str]):
    """
    Split a list of string then check if the firt element starts with -- and group the second element in a list of strings
    """
    args_list = []
    for a in args:
        if a.startswith("--"):
            v = a.split(" ")
            if len(v) >= 2:
                args_list.extend([i for i in v])
            else:
                args_list.append(a)
        else:
            args_list.append(a)
    return args_list

main(command_to_list_args_v2(command)[1:])