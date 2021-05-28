import argparse
from advpipe import utils
from os import path
import yaml
from advpipe.attacks import Attack



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fff', default='1', help='dummy argument to fool ipython')
    parser.add_argument( '-c', '--config', type=str, default=utils.rel_to_abs_path("attack_config/simple_iterative_attack.yaml"), help='AdvPipe attack YAML config file')
    args = parser.parse_args()

    exp_config = utils.load_yaml(args.config)
    attack = Attack.from_config(exp_config)

    attack.run()


