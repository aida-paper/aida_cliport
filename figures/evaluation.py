import os
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize
import numpy as np
from matplotlib import pyplot as plt

from aida_cliport.utils import utils


if __name__ == "__main__":
    cmap = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    utils.set_plot_style()
    results = {}
    interactive_demos = 300
    tasks = [
        "packing-seen-google-objects-seq",
        "packing-seen-google-objects-group",
        "packing-seen-shapes",
        "put-block-in-bowl-seen-colors",
    ]
    fig, ax = plt.subplots(2, 4, figsize=(6.50127, 0.6 * 6.50127))
    for aida in [False, True]:
        ax_idx = -1
        for seen in [True, False]:
            for task in tasks:
                ax_idx += 1
                eval_task = task
                if "seen" in task:
                    if not seen:
                        eval_task = eval_task.replace("seen", "unseen")
                relabeling_demos = aida
                validation_demos = aida
                pier = aida
                setting = f"hd={relabeling_demos}_pr={pier}_vd={validation_demos}_n={interactive_demos}"
                results = np.zeros((6, 10))
                for iteration in range(10):
                    hydra.core.global_hydra.GlobalHydra.instance().clear()
                    os.chdir(os.environ["AIDA_ROOT"])
                    with initialize(config_path="../src/aida_cliport/cfg"):
                        vcfg = compose(
                            config_name="eval",
                            overrides=[
                                f"iteration={iteration}",
                                f"interactive_demos={interactive_demos}",
                                f"model_task={task}",
                                f"eval_task={eval_task}",
                                f"relabeling_demos={relabeling_demos}",
                                f"validation_demos={validation_demos}",
                                f"eval.pier={pier}",
                            ],
                        )
                    OmegaConf.set_struct(vcfg, False)
                    eval_results = utils.get_eval_results(vcfg)
                    if eval_results is not None:
                        for i, interactive_demos in enumerate([50, 100, 150, 200, 250, 300]):
                            ckpt = f"interactive={interactive_demos}.ckpt"
                            if ckpt in eval_results:
                                results[i, iteration] = eval_results[ckpt]["mean_reward"] * 100
                            else:
                                print(f"No results for checkpoint {ckpt}, continuing...")
                                continue
                for i, interactive_demos in enumerate([50, 100, 150, 200, 250, 300]):
                    color = cmap[0] if relabeling_demos else cmap[1]
                    offset = 10 if relabeling_demos else -10
                    label = None
                    if ax_idx == 0 and seen and i == 0:
                        label = "AIDA" if aida else "Active DAgger"
                    ax[ax_idx // 4, ax_idx % 4].bar(
                        interactive_demos + offset,
                        np.mean(results[i]),
                        yerr=np.std(results[i]),
                        label=label,
                        width=20,
                        color=color,
                    )

                    title_split = eval_task.split("-")
                    title = title_split[0]
                    for i, split in enumerate(title_split[1:]):
                        if (i + 1) % 3 == 0:
                            title += f"\n{split}"
                        else:
                            title += f"-{split}"
                    ax[ax_idx // 4, ax_idx % 4].set_title(title)

                    if ax_idx % 4 == 0:
                        ax[ax_idx // 4, ax_idx % 4].set_ylabel("Reward")
                    if ax_idx // 2 >= 2:
                        ax[ax_idx // 4, ax_idx % 4].set_xlabel("Demonstrations")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}", r"\textbf{E}", r"\textbf{F}", r"\textbf{G}", r"\textbf{H}"]
    for axis, label in zip(ax.flatten(), labels):
        axis.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", transform=axis.transAxes, va="top", ha="right")
        axis.set_xticks([100, 200, 300])
    plt.savefig("figures/evaluation.pdf")
