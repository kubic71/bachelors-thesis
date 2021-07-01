#!/bin/bash

# Square attack l2
python plot_queries_needed.py -f ../results/square_attack_resnet18/n_queries_needed.txt ../results/square_attack_resnet50/n_queries_needed.txt  -o plots/iterative_attacks/square_attack_l2_eps_20.png --title "Square attack (l2, eps=20): number of queries needed to craft successful adversarial example"


# Square attack transfer
python plot_baseline_vs_transfer.py --baseline_results_dir ../results/imagenet_val_baseline_resnet50 --transfer_results_dir ../results/square_attack_transfer_resnet18_to_resnet50 --title "Transfer square attack resnet18 -> resnet50" -o plots/transfer_attacks/square_attack_transfer_resnet18_to_resnet50.png
python plot_baseline_vs_transfer.py --baseline_results_dir ../results/imagenet_val_baseline_resnet18 --transfer_results_dir ../results/square_attack_transfer_resnet50_to_resnet18 --title "Transfer square attack resnet50 -> resnet18" -o plots/transfer_attacks/square_attack_transfer_resnet50_to_resnet18.png



