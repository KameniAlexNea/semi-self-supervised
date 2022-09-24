from argparse import ArgumentParser


def semi_ssl_args(parser: ArgumentParser):
    # distillation args
    parser.add_argument("--semissl", type=str, default=None)
