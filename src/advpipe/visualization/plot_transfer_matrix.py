from utils import gather_results, mkdir_p
import matplotlib.pyplot as plt
from PIL import Image
import seaborn as sns
import pandas as pd
import os
from functools import lru_cache
from typing import Any, Callable, Dict


def plot_matrix(data: pd.DataFrame, plot_fn: str, title: str = "", figsize=(6, 6), square=True) -> Any:
    data_rows = []

    for surrogate in data["surrogate"].unique():
        for target in data["target"].unique():
            exp_df = data.query(f"surrogate == '{surrogate}' and target == '{target}'")
            foolrate = len(exp_df.query("target_loss < 0")) / len(exp_df)

            # print(surrogate, "to", target, "\t", foolrate)
            data_rows.append({"surrogate": surrogate, "target": target, "foolrate": foolrate})

    pd_matrix = pd.DataFrame(data_rows).pivot(index="surrogate", columns="target", values="foolrate")

    plt.figure(figsize=figsize)
    sns.set_theme(style="darkgrid")
    ax = sns.heatmap(pd_matrix,
                     fmt='.1%',
                     annot=True,
                     xticklabels=True,
                     yticklabels=True,
                     square=square,
                     cbar=False,
                     annot_kws={"fontsize": 8})
    ax.set(xlabel="Target")
    ax.set(ylabel="Surrogate")

    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8, rotation_mode='anchor', ha='right')
    # dist_plot.ax.set_title("APGD")
    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    plot_dir_name = os.path.dirname(plot_fn)
    mkdir_p(plot_dir_name)

    plt.savefig(plot_fn)
    return ax


@lru_cache(maxsize=None)
def augmentation_surrogate_name_remap(surrogate_name: str) -> str:
    # Names of augmented surrogated are not nicely alphabetically sorted, because augmentation params are not zero-padded
    # we cannot afford to run the experiment again
    # resnet18.noise-4 -> resnet18.noise-04
    # resnet18.blur-2 -> resnet18.blur-02

    m_name, augment_params = surrogate_name.split(".")

    new_name = m_name + "."
    parts = []
    for p in augment_params.split(","):
        pname, val = p.split("-")
        parts.append(pname + "-" + f"{int(val):02d}")

    new_name += ",".join(parts)
    return new_name


def generate_noise_eot_iter_exp() -> None:
    title = "Gaussian noise agumentation - std=35, n_iter=30\nNumber of gradient samples vs. transferability"
    plot_fn = f"plots/transfer_attacks/transfer_heatmap/noise_eot_iter_exp_resnet18.png"
    dfs = []
    for eot_iter in [1 ,3, 10, 30, 100]:
        res_dir = f"../results/noise_eot_iter_exp/resnet18/eot_iter_{eot_iter}_n_iters_30_std_35"

        def add_n_iter_to_surrogate_name(df: pd.DataFrame, _: Dict) -> pd.DataFrame:
            df["surrogate"] = df["surrogate"].map(lambda s_name: s_name + f",gradient_samples={eot_iter:03d}")
            return df

        dfs.append(gather_results(res_dir, dataframe_transform=add_n_iter_to_surrogate_name))

    plot_matrix(pd.concat(dfs), plot_fn, title, figsize=(6, 6), square=True)

def generate_augment_affine_exp() -> None:
    # Affine-transform augmentation experiment

    title = "Affine-transform augmentations - resnet18\nn_iters=30,grad_samples=10"
    plot_fn = "plots/transfer_attacks/transfer_heatmap/augment_affine_n_iters_30_eot_iters_10_eps_10.png"
    res_dir = "../results/augment_affine"

    df = gather_results(res_dir)
    plot_matrix(df, plot_fn, title, figsize=(9, 6), square=False)

def generate_augment_heatmaps() -> None:
    def fix_augmentation_dataframe(df: pd.DataFrame, _: Dict) -> pd.DataFrame:
        df["surrogate"] = df["surrogate"].map(augmentation_surrogate_name_remap)
        return df

    plot_details = []



    # Gaussian-noise Augmentation experiment
    plot_details += [
        (f"Gaussian-noise augmentations (old-experiment) - APGD - resnet18\n - margin_map, n_iters=25, l2_norm=10",
         f"plots/transfer_attacks/transfer_heatmap/_augment_noise_old_apgd_l2_margin_eps_10.png",
         f"../results/apgd_augment"),
        (f"Gaussian-noise augmentations - APGD - resnet18\n margin_map, n_iters=25, eot_iters=10, l2_norm=10",
         f"plots/transfer_attacks/transfer_heatmap/augment_noise_apgd_l2_margin_n_iters_25_eot_iters_10_eps_10.png",
         f"../results/apgd_augment_new/surrogate_resnet18/gaussian-noise_eot_iter_10_n_iters_25"),
        (f"Gaussian-noise augmentations (noisy gradient) - APGD - resnet18\n margin_map, n_iters=250, eot_iters=1, l2_norm=10",
         f"plots/transfer_attacks/transfer_heatmap/augment_noise_apgd_l2_margin_n_iters_250_eot_iters_1_eps_10.png",
         f"../results/apgd_augment_new/surrogate_resnet18/gaussian-noise_eot_iter_1_n_iters_250"),
    ]

    # Box-blur augmentation experiment
    plot_details += [
        (f"Box-blur augmentations - APGD - resnet18\n margin_map, n_iters=50, l2_norm=10",
         f"plots/transfer_attacks/transfer_heatmap/augment_blur_apgd_l2_margin_n_iters_50_eps_10.png",
         f"../results/apgd_augment_new/surrogate_resnet18/box-blur_eot_iter_1_n_iters_50"),
    ]

    for title, plot_fn, res_dir in plot_details:
        gather_and_plot(title, plot_fn, res_dir, dataframe_transform=fix_augmentation_dataframe)


def generate_all_heatmaps() -> None:
    # format: (title, plot_filename, source_results_dir)
    generate_augment_affine_exp()

    plot_details = []


    plot_details.append((f"Gaussian-noise augmentations - correct sampling - resnet18\n margin_map, n_iters=100, l2_norm=10",
         f"plots/transfer_attacks/transfer_heatmap/augment_noise_correct_sampling_apgd_l2_margin_n_iters_100_eps_10.png",
         f"../results/noise_correct_sampling"))

    for n in [32, 64]:
        plot_details.append(
        (f"DEBUG ensemble elastic transform test - APGD \n margin_map, n_iters={n}, eot_iters=15, l2_norm=10",
         f"plots/transfer_attacks/transfer_heatmap/ensemble_elastic_test_apgd_l2_margin_n_iters_{n}_eot_iters_15_eps_10.png",
         f"../results/ensemble_elastic_test/elastic_3.5_eot_iter_15_n_iters_{n}"))

    plot_details.append(
        ("Elastic transform augmentation - APGD - resnet18\n margin_map, n_iters=25, eot_iters=10, l2_norm=10",
         "plots/transfer_attacks/transfer_heatmap/augment_elastic_apgd_l2_margin_n_iters_25_eot_iters_10_eps_10.png",
         "../results/apgd_augment_elastic"))

    # APGD n-iters L2
    for n_iter in [1, 2, 4, 8, 16, 32, 64]:
        plot_details.append((f"n-iters={n_iter}, APGD - margin_map, l2_norm=10",
                             f"plots/transfer_attacks/transfer_heatmap/apgd_margin_l2_eps_10_n_iters_{n_iter:04d}.png",
                             f"../results/apgd_n_iters_exp_l2/n_iters_{n_iter}"))

    # APGD early stopping
    for early_stop in [2, 4, 8, 16, 32, 64]:
        plot_details += [
            (f"early_stop_at={early_stop}, APGD - margin_map, l2_norm=10",
             f"plots/transfer_attacks/transfer_heatmap/apgd_l2_margin_early_stop_{early_stop}_eps_10.png",
             f"../results/apgd_early_stopping_l2_10/stop_at_{early_stop}"),
            (f"early_stop_at={early_stop}, APGD - margin_map, linf_norm=0.04",
             f"plots/transfer_attacks/transfer_heatmap/apgd_linf_margin_early_stop_{early_stop}_eps_0.04.png",
             f"../results/apgd_early_stopping_linf_0.04/stop_at_{early_stop}")
        ]

    ### Square autoattack ###
    for norm, eps in [('l2', "10"), ('linf', "0.05")]:
        for n_restarts in [1, 2, 4]:
            plot_details.append(
                (f"SquareAttack, n_images=500, {norm}_norm={eps}, n_restarts={n_restarts}",
                 f"plots/transfer_attacks/transfer_heatmap/square_{norm}_eps_{eps}_n_restarts_{n_restarts}.png",
                 f"../results/square_auto_attack/norm_{norm}/n_restarts_{n_restarts}/eps_{eps}"))

    plot_details += [
    #### APGD L2 ####

    # output-mapping margin
        ("APGD, margin output mapping, n_images=500, n_iters=25, l2_norm=3",
         "plots/transfer_attacks/transfer_heatmap/apgd_l2_margin_n_iters_25_eps_3.png",
         "../results/net_crossproduct_apgd_surrogate_outputmap_margin/eps_3"),
        ("APGD, margin output mapping, n_images=500, n_iters=25, l2_norm=10",
         "plots/transfer_attacks/transfer_heatmap/apgd_l2_margin_n_iters_25_eps_10.png",
         "../results/net_crossproduct_apgd_surrogate_outputmap_margin/eps_10"),
        ("APGD, margin output mapping, n_images=500, n_iters=25, l2_norm=20",
         "plots/transfer_attacks/transfer_heatmap/apgd_l2_margin_n_iters_25_eps_20.png",
         "../results/net_crossproduct_apgd_surrogate_outputmap_margin/eps_20"),

    # output-mapping logits
        ("APGD, logits output mapping, n_images=500, n_iters=25, l2_norm=3",
         "plots/transfer_attacks/transfer_heatmap/apgd_l2_logits_n_iters_25_eps_3.png",
         "../results/net_crossproduct_apgd_surrogate_outputmap_logits/eps_3"),
        ("APGD, logits output mapping, n_images=500, n_iters=25, l2_norm=10",
         "plots/transfer_attacks/transfer_heatmap/apgd_l2_logits_n_iters_25_eps_10.png",
         "../results/net_crossproduct_apgd_surrogate_outputmap_logits/eps_10"),
        ("APGD, logits output mapping, n_images=500, n_iters=25, l2_norm=20",
         "plots/transfer_attacks/transfer_heatmap/apgd_l2_logits_n_iters_25_eps_20.png",
         "../results/net_crossproduct_apgd_surrogate_outputmap_logits/eps_20"),

    #### FGSM ####
    # output-mapping logits
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.01",
         "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.010.png",
         "../results/net_crossproduct_fgsm/output_map_logits/eps_0.01"),
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.025",
         "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.025.png",
         "../results/net_crossproduct_fgsm/output_map_logits/eps_0.025"),
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.05",
         "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.050.png",
         "../results/net_crossproduct_fgsm/output_map_logits/eps_0.05"),
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.1",
         "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.100.png",
         "../results/net_crossproduct_fgsm/output_map_logits/eps_0.1"),
        ("FGSM, logits output mapping, n_images=500, linf_norm=0.2",
         "plots/transfer_attacks/transfer_heatmap/fgsm_logits_eps_0.200.png",
         "../results/net_crossproduct_fgsm/output_map_logits/eps_0.2"),

    # output-mapping margin
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.01",
         "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.010.png",
         "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.01"),
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.025",
         "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.025.png",
         "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.025"),
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.05",
         "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.050.png",
         "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.05"),
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.1",
         "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.100.png",
         "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.1"),
        ("FGSM, margin output mapping, n_images=500, linf_norm=0.2",
         "plots/transfer_attacks/transfer_heatmap/fgsm_margin_eps_0.200.png",
         "../results/net_crossproduct_fgsm/output_map_organism_margin/eps_0.2"),
    ]

    for title, plot_fn, res_dir in plot_details:
        gather_and_plot(title, plot_fn, res_dir)

    generate_noise_eot_iter_exp()
    generate_augment_heatmaps()

def gather_and_plot(
        title: str,
        plot_fn: str,
        res_dir: str,
        experiment_filter: Callable[[Dict], bool] = lambda config: True,
        dataframe_transform: Callable[[pd.DataFrame, Dict], pd.DataFrame] = lambda data, config: data) -> None:

    print(f"Title: {title}, plot filename: {plot_fn}, source_dir: {res_dir}")
    df = gather_results(res_dir, experiment_filter, dataframe_transform)
    _ = plot_matrix(df, plot_fn, title)
    # heatmap.show()


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--dir", required=True, type=str)

# args = parser.parse_args()

generate_all_heatmaps()
# plt.show()
# plot.figure.savefig('/tmp/test.png', transparent=True, bbox_inches='tight')
