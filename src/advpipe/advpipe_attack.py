from __future__ import annotations
import argparse
from advpipe import utils
from advpipe.config_datamodel import AdvPipeConfig
from advpipe.log import logger
import time
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Sequence, Tuple, Dict, Iterator
    from argparse import Namespace


def split_to_macro_and_template(config_content: str) -> Tuple[str, str]:
    conf_lines = config_content.strip().split("\n")
    macro_lines = []
    template_lines = []
    line_i = 0
    while True:
        line = conf_lines[line_i]
        if line.startswith('#@  '):
            macro_lines.append(line[4:])
            line_i += 1
        elif line.startswith('#@'):
            raise Exception(
                f"Config macro must be prefixed with '#@' followed by two white-spaces!\nError line: {line}")
        else:
            break

    template_lines = conf_lines[line_i:]

    macro = "\n".join(macro_lines)
    template = "\n".join(template_lines)

    return macro, template


def params_to_str(parameters: Dict[str, str]) -> str:
    result = []
    for key, val in parameters.items():
        result.append(f"{key}={val}")

    return "_".join(result).replace(" ", "-")


def is_valid_yaml_str(yaml_str: str) -> bool:
    try:
        _ = utils.load_yaml_from_str(yaml_str)
    except:
        return False
    return True


def get_default_config_parameters() -> Iterator[Dict[str, str]]:
    return iter([{}])


def preprocess_config(config_fn: str) -> Sequence[str]:
    """Compile one non-YAML config file into several valid YAML configs,
    where variable attributes are substituted by corresponding values

        args: 
            config_fn: str - path to config template file
        returns:
            Sequence[str] - exported YAML config paths
     
    AdvPipe config file can contain optional python code-block at the beggining.
    Each python code line must be escaped by '#@  ' (two white-spaces at the end).
    These for characters are removed before passing the code to the python interpreter
    If it contains python code block, it must define a function:
        def get_config_parameters() -> Iterator[Dict[str, str]]

    This function will return individual instances of variable parameters. 

    For example, if we want to run several attacks, but each with different epsilon,
    we could define a macro:
    #@  def get_config_parameters() -> Iterator[Dict[str, str]]:
    #@      for epsilon in [0.01, 0.02, 0.05, 0.1, 0.2]:
    #@          yield {"epsilon": epsilon}

    Values returned by this function are substituted to '{key}' sections in the YAML config.
    So somewhere further down in the config, we could use this epsilon parameter instance like this:

    attack_regime:
        results_dir: results/my_attack/eps_{epsilon}
        attack_algorithm:
            ...

        epsilon: {epsilon}
        ...


    These compiled YAML configs are exported to temp directory and passed one by one to AdvPipe to be executed
    """

    with open(config_fn, "r") as f:
        config_content = f.read()

    macro, template = split_to_macro_and_template(config_content)

    if macro != '':
        # run macro
        # exports get_config_parameters() to local namespace and overwrites the default function
        # very bad practice, I know
        exec(macro)
        param_function = locals()['get_config_parameters']
    else:
        param_function = get_default_config_parameters

    temp_dir_path = utils.convert_to_absolute_path(".exported_configs")
    utils.mkdir_p(temp_dir_path)
    config_fns = []
    for parameters in param_function():
        yaml_config_str = template.format_map(parameters)

        exported_fn = f"{temp_dir_path}/{os.path.basename(config_fn).split('.')[0]}_{params_to_str(parameters)}.yaml"
        with open(exported_fn, "w") as f:
            f.write(yaml_config_str)
        config_fns.append(exported_fn)
    return config_fns


def run_attacks(config_files: Sequence[str]) -> None:
    for i, config_fn in enumerate(config_files):
        logger.info(f"Running experiment {i + 1}/{len(config_files)} from config: {config_fn}")

        start = time.time()

        advpipe_config = AdvPipeConfig(config_fn)
        logger.info(f"Results directory: {advpipe_config.attack_regime_config.results_dir}")

        attack = advpipe_config.getAttackInstance()
        attack.run()

        logger.info(f"Finished. Took {time.time() - start:.2f}s\n")


def set_log_level(args: Namespace) -> None:
    if args.loglevel is not None:
        logger.set_level(args.loglevel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fff', default='1', help='dummy argument to fool ipython')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default=utils.convert_to_absolute_path("attack_config/test.yaml"),
                        help='AdvPipe attack YAML config file')

    parser.add_argument('--loglevel',
                        default=None,
                        choices=["debug", "info", "error"],
                        help="Overwrite default log level")

    args = parser.parse_args()

    set_log_level(args)

    config_fns = preprocess_config(args.config)
    run_attacks(config_fns)
