#!/bin/bash

# Evaluate baseline performance
python advpipe_attack.py --config=attack_config/imagenet_val_baseline_resnet18.yaml
python advpipe_attack.py --config=attack_config/imagenet_val_baseline_resnet50.yaml
python advpipe_attack.py --config=attack_config/imagenet_val_baseline_gvision.yaml


# DamageNet
python advpipe_attack.py --config=attack_config/damagenet_transfer_resnet18.yaml
python advpipe_attack.py --config=attack_config/damagenet_transfer_resnet50.yaml
python advpipe_attack.py --config=attack_config/damagenet_transfer_gvision.yaml


# Square attack
python advpipe_attack.py --config=attack_config/square_attack_resnet18.yaml
python advpipe_attack.py --config=attack_config/square_attack_resnet50.yaml
# no advtrain vs. advtrain
python advpipe_attack.py --config=attack_config/square_attack_efficientnet_b0.yaml 
python advpipe_attack.py --config=attack_config/square_attack_efficientnet_b0_advtrained.yaml 


# Square transfer
python advpipe_attack.py --config=attack_config/square_attack_resnet18_to_resnet50.yaml
python advpipe_attack.py --config=attack_config/square_attack_resnet50_to_resnet18.yaml

python advpipe_attack.py --config=attack_config/square_attack_resnet50_to_gvision.yaml


