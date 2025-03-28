import os
import hydra
from omegaconf import OmegaConf
from hydra.experimental import compose, initialize
import numpy as np
from matplotlib import pyplot as plt

import aida_cliport
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
    fig, ax = plt.subplots(1, 4, figsize=(6.50127, 0.35 * 6.50127))
    for aida in [False, True]:
        ax_idx = -1
        for task in tasks:
            setting_results = dict(annotated=[], validated=[], relabeled=[])
            ax_idx += 1
            relabeling_demos = aida
            validation_demos = aida
            pier = aida
            setting = f"hd={relabeling_demos}_pr={pier}_vd={validation_demos}_n={interactive_demos}"
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
                            f"eval_task={task}",
                            f"relabeling_demos={relabeling_demos}",
                            f"validation_demos={validation_demos}",
                            f"eval.pier={pier}",
                        ],
                    )
                OmegaConf.set_struct(vcfg, False)
                train_results = utils.get_train_results(vcfg)
                if train_results is not None:
                    r = train_results["r"]
                    demos = train_results["demos"]
                    episodes = train_results["episodes"]
                    demo_count = 0
                    max_i = 0
                    max_e = 0
                    n_relabeling = 0
                    n_annotation = 0
                    n_validation = 0
                    relabeled = []
                    annotated = []
                    validated = []
                    stop = False
                    while demo_count < interactive_demos:
                        episode = episodes[max_e]
                        demo = False
                        for step in range(len(episode[3])):
                            if demos[max_i]:
                                demo = True
                                if validation_demos and r[max_i] == aida_cliport.KNOWN_SUCCESS:
                                    n_validation += 1
                                elif not validation_demos or r[max_i] == aida_cliport.KNOWN_FAILURE:
                                    n_annotation += 1
                            if max_i < len(r) - 1:
                                if r[max_i + 1] == aida_cliport.UNKNOWN_RELABELING:
                                    demo = True
                                    n_relabeling += 1
                                    max_i += 1
                            max_i += 1
                        if demo:
                            demo_count += 1
                            if demo_count % 50 == 0:
                                annotated.append(n_annotation)
                                validated.append(n_validation)
                                relabeled.append(n_relabeling)
                        max_e += 1
                        if demo_count < interactive_demos and max_e >= len(episodes):
                            stop = True
                            break
                    if stop:
                        print(f"Not enough demos collected, only {demo_count}/{interactive_demos}")
                        print(f"setting={setting}, task={task}, iteration={iteration}")
                    setting_results["annotated"].append(annotated)
                    setting_results["validated"].append(validated)
                    setting_results["relabeled"].append(relabeled)

            if len(setting_results["annotated"]) == 10:
                offset = 10 if aida else -10
                hatch = "//" if aida else None

                ax[ax_idx].set_xlabel("Demonstrations")
                if ax_idx == 0:
                    ax[ax_idx].set_ylabel("Demonstration Tuples")

                title_split = task.split("-")
                if "seen" in title_split:
                    title_split.remove("seen")
                if "colors" in title_split:
                    title_split.remove("colors")
                title = title_split[0]
                for i, split in enumerate(title_split[1:]):
                    if i % 2 == 0:
                        title += f"-{split}"
                    else:
                        title += f"\n{split}"
                ax[ax_idx].set_title(title)

                # Annotation demos
                o_color = "r"
                mean_annotated = np.mean(setting_results["annotated"], axis=0)
                label = "AIDA: Annotation" if aida else "Active DAgger: Annotation"
                ax[ax_idx].bar(
                    50 * np.arange(1, 1 + len(mean_annotated)) + offset,
                    mean_annotated,
                    color=o_color,
                    width=20,
                    hatch=hatch,
                    label=label,
                )

                # Relabeled demos
                if relabeling_demos:
                    mean_relabeled = np.mean(setting_results["relabeled"], axis=0)
                    label = "AIDA: Relabeled"
                    ax[ax_idx].bar(
                        50 * np.arange(1, 1 + len(mean_annotated)) + offset,
                        mean_relabeled,
                        color="k",
                        bottom=mean_annotated,
                        width=20,
                        hatch=hatch,
                        label=label,
                    )

                # Validation demos
                if validation_demos:
                    mean_validated = np.mean(setting_results["validated"], axis=0)
                    label = "AIDA: Validation"
                    ax[ax_idx].bar(
                        50 * np.arange(1, 1 + len(mean_annotated)) + offset,
                        mean_validated,
                        color="g",
                        bottom=mean_annotated + mean_relabeled,
                        width=20,
                        hatch=hatch,
                        label=label,
                    )
            else:
                print(f"setting={setting}, task={task} missing results")

    handles, labels = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.17, 1, 1])
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0))
    labels = [r"\textbf{A}", r"\textbf{B}", r"\textbf{C}", r"\textbf{D}"]
    for axis, label in zip(ax.flatten(), labels):
        axis.text(-0.1, -0.1, label, fontsize=8, fontweight="bold", transform=axis.transAxes, va="top", ha="right")
    plt.savefig("figures/demo_types.pdf")
