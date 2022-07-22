from files.utils.constants import *
from files.utils.utility_functions import *
from files.functions.results_extractors import *

def make_scores_rocs_plots(root_path, base_path, results=None, towrite=False):
    results = extract_results_models(
        base_path, towrite=towrite) if results is None else results
    if towrite:
        base_path = joinpath(base_path, "summary_photos")
        os.makedirs(base_path, exist_ok=True)
    for model, dict1 in results.items():
        if model != SUBS:
            for ds_name, dict2 in dict1.items():
                ds, gt = load_dataset_by_name(ds_name, file_path=joinpath(
                    root_path, f"datasets/2d/{'circles' if 'circle' in ds_name else 'lines'}/with_outliers"))
                for threshold, dict3 in dict2.items():
                    try:
                        threshold = float(threshold)
                    except:
                        continue
                    also_localized = "roc_auc_localized" in dict3.keys()

                    # Generate the figure and axes
                    fig, axes = plt.subplots(
                        2 if also_localized else 1, 2, figsize=(16, 9), dpi=200)
                    if also_localized:
                        (ax1, ax2), (ax3, ax4) = axes
                    else:
                        ax1, ax2 = axes

                    fig.suptitle(
                        f"Model {model} on ds {ds_name} with thr {threshold:.4f}")

                    ax1.set_aspect("equal")
                    ax2.set_aspect("equal")

                    # Plot uniform sampling results
                    uniform_scores_path = root_path + dict3["scores_path"]
                    uniform_scores = np.array(
                        read_arr_json(uniform_scores_path))

                    im1 = ax1.scatter(
                        ds[:, 0], ds[:, 1], c=uniform_scores, cmap="jet", s=10, alpha=0.9)
                    ax1.set_title("Uniform sampling")

                    make_roc(gt=gt, scores=uniform_scores, orig_ax=ax2,
                             title="Uniform ROC", show=False)

                    # Plot localized sampling results
                    if also_localized:
                        ax3.set_aspect("equal")
                        ax4.set_aspect("equal")
                        localized_scores_path = uniform_scores_path.replace(
                            UNIFORM, LOCALIZED)
                        localized_scores = np.array(
                            read_arr_json(localized_scores_path))
                        im2 = ax3.scatter(
                            ds[:, 0], ds[:, 1], c=localized_scores, cmap="jet", s=10, alpha=0.9)
                        ax3.set_title("Localized sampling")
                        make_roc(gt=gt, scores=localized_scores,
                                 orig_ax=ax4, title="Localized ROC", show=False)

                    # Create the colorbar
                    if also_localized:
                        fig.colorbar(im1, ax=ax1, shrink=1,
                                     fraction=0.1, pad=0.02)
                        fig.colorbar(im2, ax=ax3, shrink=1,
                                     fraction=0.1, pad=0.02)
                    else:
                        fig.colorbar(im1, ax=ax1, shrink=0.9,
                                     fraction=0.1, pad=0.02)

                    fig.tight_layout()
                    if towrite:
                        fig.savefig(joinpath(
                            base_path, f"dataset_{ds_name}_model_{model}_ith_{str(threshold).replace('.', '_')}.svg"), bbox_inches="tight")
                    else:
                        plt.show()
                    plt.close(fig)


def make_rocs_barplot(base_path, results=None, towrite=False):
    results = extract_results_datasets(
        base_path, towrite=towrite) if results is None else results
    if not results:
        print("Can't create results dict")
        return
    if towrite:
        base_path = joinpath(base_path, "summary_barplots")
        os.makedirs(base_path, exist_ok=True)

    for ds_name, dict1 in results.items():
        models = [m for m in dict1.keys() if m != SUBS]
        thresholds = [th for th in dict1[models[0]].keys() if "." in th]

        thresholds.sort()
        fig, axes = plt.subplots(len(thresholds), 1, figsize=(
            10, 30), dpi=300, constrained_layout=True)

        for i, th in enumerate(thresholds):
            if len(thresholds) == 1:
                ax = axes
            else:
                ax = axes[i]
            aucs = [float(dict1[model][th]["roc_auc_uniform"])
                    for model in models]
            ax.barh([i for i in range(len(models))], aucs,
                    align='center', color=['#B93A32', '#6ea331', '#217fbe'])
            for i, v in enumerate(aucs):
                v_str = f"{v:.3f}"
                color = "black" if v <= 1 else "white"
                v = v+0.01 if v <= 1 else v-0.07
                ax.text(v, i-0.05, v_str, color=color)
            # , labels=thresholds)
            ax.set_yticks([i for i in range(len(models))])
            ax.set_yticklabels(models)

            mi, ma = ax.get_ylim()
            ax.set_ylim(mi-0.3, ma+0.3)
            ax.set_xlim([0, 1.065])
            ax.set_ylabel("Models")
            ax.set_xlabel("AUC")
            ax.set_title(f"Threshold {float(th):.3f}")

        fig.suptitle(f"Dataset {ds_name}")
        print(f"Ds name: {ds_name}")
        # fig.tight_layout()
        if towrite:
            fig.savefig(joinpath(
                base_path, f"dataset_{ds_name}.svg"), bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
