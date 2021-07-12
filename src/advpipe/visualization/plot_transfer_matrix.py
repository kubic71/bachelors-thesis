from utils import gather_results
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
from typing import Any


def plot_matrix(data: pd.DataFrame, plot_fn: str, title: str = "") -> Any:
    data_rows = []

    for surrogate in data["surrogate"].unique():
        for target in data["target"].unique():
            exp_df = data.query(f"surrogate == '{surrogate}' and target == '{target}'")
            foolrate = len(exp_df.query("target_loss < 0")) / len(exp_df)

            # print(surrogate, "to", target, "\t", foolrate)
            data_rows.append({"surrogate":surrogate, "target":target, "foolrate": foolrate})

    
    pd_matrix = pd.DataFrame(data_rows).pivot(index="surrogate", columns="target", values="foolrate")

    plt.figure(figsize=(6, 6))
    sns.set_theme(style="darkgrid")
    ax= sns.heatmap(pd_matrix, fmt='.1%', annot=True, xticklabels=True, yticklabels=True, square=True, cbar=False, annot_kws={"fontsize":8})
    ax.set(xlabel="Target")
    ax.set(ylabel="Surrogate")
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, rotation_mode='anchor', ha='right')
    # dist_plot.ax.set_title("APGD")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(plot_fn)
    return ax


def generate_all_heatmaps() -> None:
    # format: (title, plot_filename, source_results_dir)

    plot_details = []

    # APGD early stopping 
    for early_stop in [2, 4, 8, 16, 32, 64]:
        plot_details.append(
        (f"APGD - margin_map, early_stop_at={early_stop}, l2_norm=10", f"plots/transfer_attacks/transfer_heatmap/apgd_l2_margin_early_stop_{early_stop}_eps_10.png", f"../results/apgd_early_stopping_l2_10/stop_at_{early_stop}"))

    
    ### Square autoattack ###
    for norm, eps in [('l2', "10"), ('linf', "0.05")]:
        for n_restarts in [1, 2, 4]:
            plot_details.append((f"SquareAttack, n_images=500, l{norm}_norm={eps}, n_restarts={n_restarts}", f"plots/transfer_attacks/transfer_heatmap/square_{norm}_eps_{eps}_n_restarts_{n_restarts}.png", f"../results/square_auto_attack/norm_{norm}/n_restarts_{n_restarts}/eps_{eps}"))

        
    plot_details += [
        #### APGD L2 ####

        # output-mapping margin
        ("APGD, margin output mapping, n_images=500, n_iters=25, l2_norm=3", "plots/transfer_attacks/transfer_heatmap/apgd_l2_margin_n_iters_25_eps_3.png", "../results/net_crossproduct_apgd_surrogate_outputmap_margin/eps_3"),
        ("APGD, margin output mapping, n_images=500, n_iters=25, l2_norm=10", "plots/transfer_attacks/transfer_heatmap/apgd_l2_margin_n_iters_25_eps_10.png", "../results/net_crossproduct_apgd_surrogate_outputmap_margin/eps_10"),
        ("APGD, margin output mapping, n_images=500, n_iters=25, l2_norm=20", "plots/transfer_attacks/transfer_heatmap/apgd_l2_margin_n_iters_25_eps_20.png", "../results/net_crossproduct_apgd_surrogate_outputmap_margin/eps_20"),


        # output-mapping logits
        ("APGD, logits output mapping, n_images=500, n_iters=25, l2_norm=3", "plots/transfer_attacks/transfer_heatmap/apgd_l2_logits_n_iters_25_eps_3.png", "../results/net_crossproduct_apgd_surrogate_outputmap_logits/eps_3"),
        ("APGD, logits output mapping, n_images=500, n_iters=25, l2_norm=10", "plots/transfer_attacks/transfer_heatmap/apgd_l2_logits_n_iters_25_eps_10.png", "../results/net_crossproduct_apgd_surrogate_outputmap_logits/eps_10"),
        ("APGD, logits output mapping, n_images=500, n_iters=25, l2_norm=20", "plots/transfer_attacks/transfer_heatmap/apgd_l2_logits_n_iters_25_eps_20.png", "../results/net_crossproduct_apgd_surrogate_outputmap_logits/eps_20"),

        #### FGSM ####
        # output-mapping logits
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.01", "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.010.png", "../results/net_crossproduct_fgsm/output_map_logits/eps_0.01"),
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.025", "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.025.png", "../results/net_crossproduct_fgsm/output_map_logits/eps_0.025"),
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.05", "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.050.png", "../results/net_crossproduct_fgsm/output_map_logits/eps_0.05"),
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.1", "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.100.png", "../results/net_crossproduct_fgsm/output_map_logits/eps_0.1"),
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.2", "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.200.png", "../results/net_crossproduct_fgsm/output_map_logits/eps_0.2"),

        # output-mapping margin
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.01", "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.010.png", "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.01"),
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.025", "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.025.png", "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.025"),
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.05", "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.050.png", "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.05"),
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.1", "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.100.png", "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.1"),
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.2", "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.200.png", "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.2"),
     ]




    for title, plot_fn, res_dir in plot_details:
        print(f"Title: {title}, plot filename: {plot_fn}, source_dir: {res_dir}")
        df = gather_results(res_dir)
        _ = plot_matrix(df, plot_fn, title)



    # heatmap.show()

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--dir", required=True, type=str)

# args = parser.parse_args()

generate_all_heatmaps()
# plt.show()
# plot.figure.savefig('/tmp/test.png', transparent=True, bbox_inches='tight')




