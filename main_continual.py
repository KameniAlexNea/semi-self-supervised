import copy
import itertools
import subprocess
import sys


def str_to_dict(command):
    d = {}
    for part, part_next in itertools.zip_longest(command[:-1], command[1:]):
        if part[:2] == "--":
            if part_next[:2] != "--":
                d[part] = part_next
            else:
                d[part] = part
        elif part[:2] != "--" and part_next[:2] != "--":
            part_prev = list(d.keys())[-1]
            if not isinstance(d[part_prev], list):
                d[part_prev] = [d[part_prev]]
            if not part_next[:2] == "--":
                d[part_prev].append(part_next)
    return d


def dict_to_list(command):
    s = []
    for k, v in command.items():
        s.append(k)
        if k != v and v[:2] != "--":
            s.append(v)
    return s


def run_bash_command(args):
    for i, a in enumerate(args):
        if isinstance(a, list):
            args[i] = " ".join(a)
    command = ("python3 main_pretrain.py", *args)
    command = " ".join(command)
    p = subprocess.Popen(command, shell=True)
    p.wait()


if __name__ == "__main__":
    args = sys.argv[1:]
    args = str_to_dict(args)

    # parse args from the script
    semi_args = {k: v for k, v in args.items() if "semi" in k}

    # delete things that shouldn't be used for task_idx 0
    for k in semi_args.keys():
        args.pop(k, None)

    # main task loop
    print(f"\n#### Starting Task Semi Self Supervised ####")

    task_args = copy.deepcopy(args)

    task_args.update(semi_args)

    task_args = dict_to_list(task_args)

    run_bash_command(task_args)
