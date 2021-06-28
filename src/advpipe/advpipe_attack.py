import argparse
from advpipe import utils
from advpipe.config_datamodel import AdvPipeConfig

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fff', default='1', help='dummy argument to fool ipython')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default=utils.convert_to_absolute_path("attack_config/square_attack_transfer_resnet50_to_resnet18.yaml"),
                        help='AdvPipe attack YAML config file')
    args = parser.parse_args()

    advpipe_config = AdvPipeConfig(args.config)
    attack = advpipe_config.getAttackInstance()

    attack.run()
