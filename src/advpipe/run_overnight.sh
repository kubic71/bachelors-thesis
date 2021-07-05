#!/bin/bash
python advpipe_attack.py --config=attack_config/imagenet_val_baseline_resnet50.yaml
python advpipe_attack.py --config=attack_config/square_attack_resnet50.yaml

