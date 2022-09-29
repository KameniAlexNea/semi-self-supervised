from argparse import ArgumentParser
from semissl.semi import SEMISUPERVISED

def semi_ssl_args(parser: ArgumentParser):
    # distillation args
    parser.add_argument("--semissl", choices=SEMISUPERVISED, default=None)
