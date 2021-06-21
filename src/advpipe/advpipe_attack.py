import argparse
from advpipe import utils
from advpipe.config_datamodel import AdvPipeConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fff', default='1', help='dummy argument to fool ipython')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default=utils.rel_to_abs_path("attack_config/simple_iterative_attack.yaml"),
                        help='AdvPipe attack YAML config file')
    args = parser.parse_args()

    advpipe_config = AdvPipeConfig(utils.load_yaml(args.config))
    attack = advpipe_config.getAttackInstance()

    attack.run()
