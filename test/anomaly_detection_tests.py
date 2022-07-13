from files.utils.constants import *
from files.utils.utility_functions import *
from files.pif.pif import *
from files.models.neural_models import AEModel
from files.models.self_organizing_maps import SelfOrganizingMaps

from multiprocessing.pool import Pool
import time
import collections
import traceback


def extract_results_paths(base_path, results=None):
    results = np.array([]) if results is None else results
    dirs = os.listdir(base_path)
    if "results.json" in dirs:
        ret = np.array([joinpath(base_path, "results.json")])
        return ret
    else:
        for d in dirs:
            d = joinpath(base_path, d)
            if os.path.isdir(d):
                for r in extract_results_paths(d, results):
                    if r not in results.tolist():
                        results = np.append(results, r)
        return results


def extract_results_models(base_path, towrite=False):
    results_paths = extract_results_paths(base_path)
    final_results = {}
    for path in results_paths:
        params = read_arr_json(path, return_dict=True)
        mn = params["model_name"]
        if mn not in final_results.keys():
            final_results[mn] = {}
        ds_name = params["dataset_name"]
        if ds_name not in final_results[mn].keys():
            final_results[mn][ds_name] = {}
            # final_results[fc][ds_name]["npoints"] = params["npoints"]
        in_th_num = str(params["in_th_to_test"]).replace(".", "_")
        # in_th = f"in_th_{in_th_num}"
        in_th = params["in_th_to_test"]
        in_th = str(in_th)
        if in_th not in final_results[mn][ds_name].keys():
            final_results[mn][ds_name][in_th] = {}

        if LOCALIZED in path:
            final_results[mn][ds_name][in_th]["roc_auc_localized"] = params["roc_auc"]
        elif UNIFORM in path:
            final_results[mn][ds_name][in_th]["roc_auc_uniform"] = params["roc_auc"]
        final_results[mn][ds_name][in_th]["scores_path"] = params["scores_path"]
        final_results[mn][ds_name][in_th]["exec_time_seconds"] = params["exec_time_seconds"]

    for mn, dict1 in final_results.items():
        for ds_name, dict2 in dict1.items():
            thresholds = list(dict2.keys())
            try:
                roc_aucs = [dict2[v]["roc_auc_localized"]
                            for v in thresholds]
                final_results[mn][ds_name]["roc_auc_localized_median"] = np.median(
                    roc_aucs)
                final_results[mn][ds_name]["roc_auc_localized_mean"] = np.mean(
                    roc_aucs)
            except KeyError:
                pass
            try:
                roc_aucs = [dict2[v]["roc_auc_uniform"]
                            for v in thresholds]
                final_results[mn][ds_name]["roc_auc_uniform_median"] = np.median(
                    roc_aucs)
                final_results[mn][ds_name]["roc_auc_uniform_mean"] = np.mean(
                    roc_aucs)
            except KeyError:
                pass
            exec_times = [dict2[v]["exec_time_seconds"]
                          for v in thresholds]
            final_results[mn][ds_name]["exec_time_seconds_mean"] = np.mean(
                exec_times)

    os.makedirs(joinpath(base_path), exist_ok=True)
    if towrite:
        write_dict_json(final_results, joinpath(
            base_path, "summary_results.json"))
    return final_results


def extract_results_datasets(base_path, towrite=False):
    results_paths = extract_results_paths(base_path)
    final_results = {}
    for path in results_paths:
        params = read_arr_json(path, return_dict=True)
        mn = params["model_name"]
        ds_name = params["dataset_name"]
        if ds_name not in final_results.keys():
            final_results[ds_name] = {}
        if mn not in final_results[ds_name].keys():
            final_results[ds_name][mn] = {}
            # final_results[fc][ds_name]["npoints"] = params["npoints"]
        in_th_num = str(params["in_th_to_test"]).replace(".", "_")
        # in_th = f"in_th_{in_th_num}"
        in_th = params["in_th_to_test"]
        in_th = str(in_th)
        if in_th not in final_results[ds_name][mn].keys():
            final_results[ds_name][mn][in_th] = {}

        if LOCALIZED in path:
            final_results[ds_name][mn][in_th]["roc_auc_localized"] = params["roc_auc"]
        elif UNIFORM in path:
            final_results[ds_name][mn][in_th]["roc_auc_uniform"] = params["roc_auc"]
        final_results[ds_name][mn][in_th]["scores_path"] = params["scores_path"]
        final_results[ds_name][mn][in_th]["exec_time_seconds"] = params["exec_time_seconds"]

    # for mn, dict1 in final_results.items():
    #     for ds_name, dict2 in dict1.items():
    #         thresholds = list(dict2.keys())
    #         try:
    #             roc_aucs = [dict2[v]["roc_auc_localized"]
    #                         for v in thresholds]
    #             final_results[ds_name][mn]["roc_auc_localized_median"] = np.median(
    #                 roc_aucs)
    #             final_results[ds_name][mn]["roc_auc_localized_mean"] = np.mean(
    #                 roc_aucs)
    #         except KeyError:
    #             pass
    #         try:
    #             roc_aucs = [dict2[v]["roc_auc_uniform"]
    #                         for v in thresholds]
    #             final_results[ds_name][mn]["roc_auc_uniform_median"] = np.median(
    #                 roc_aucs)
    #             final_results[ds_name][mn]["roc_auc_uniform_mean"] = np.mean(
    #                 roc_aucs)
    #         except KeyError:
    #             pass
    #         exec_times = [dict2[v]["exec_time_seconds"]
    #                       for v in thresholds]
    #         final_results[ds_name][mn]["exec_time_seconds_mean"] = np.mean(
    #             exec_times)

    os.makedirs(joinpath(base_path), exist_ok=True)
    if towrite:
        write_dict_json(final_results, joinpath(
            base_path, "summary_results.json"))
    return final_results


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


def test(inp):
    (dataset_name, threshold), model_name, sampling = inp
    dataset, gt = load_dataset_by_name(dataset_name)

    in_th_str = f"{float(threshold):.3f}".replace('.', '_')
    results_path = joinpath(
        params["base_path"], dataset_name, model_name, f"in_th_{in_th_str}", sampling)

    os.makedirs(results_path, exist_ok=True)

    start_time = time.monotonic()
    weights, predict = False, False

    pif = PreferenceIsolationForest(
        data=dataset,
        model_name=model_name,
        in_th=float(threshold),
        sampling=sampling,
        verbose=0
    )
    scores = pif.anomaly_detection(
        num_models=params["num_models"], mss=params["mss"])
    write_arr_json(path=joinpath(results_path, "scores.json"), arr=scores)

    if params["cool_visualization"]:
        x0_min, x0_max = ds[:, 0].min() - .1, ds[:, 0].max() + .1
        x1_min, x1_max = ds[:, 1].min() - .1, ds[:, 1].max() + .1

        h = .035
        xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                               np.arange(x1_min, x1_max, h))
        data = np.c_[xx0.ravel(), xx1.ravel()]

        preference_matrix, _ = build_preference_matrix(
            data=data, models_ithrs=pif.models_ithrs, verbose=1)
        pif.voronoi.fit(preference_matrix)
        cool_scores = pif.voronoi.score_samples(preference_matrix)
        write_arr_json(path=joinpath(
            results_path, "cool_scores.json"), arr=cool_scores)

    exec_time = time.monotonic() - start_time

    auc, tpr, fpr, thr, fig = make_roc(
        gt, scores, show=False, title=f"{dataset_name} Model: {model_name}, Thr: {in_th_str}, Smpl: {sampling}")
    fig.tight_layout()
    fig.savefig(fname=joinpath(results_path, "roc.svg"), bbox_inches="tight")
    plt.close(fig)

    results = params.copy()
    results["dataset_name"] = dataset_name
    results["scores_path"] = joinpath(results_path, "scores.json")

    # results["roc_tpr"] = str(list(tpr))
    # results["roc_fpr"] = str(list(fpr))
    # results["roc_thr"] = str(list(thr))
    results["roc_auc"] = auc

    results["exec_time_seconds"] = exec_time
    results["in_th_to_test"] = pif.in_ths
    results["model_name"] = model_name
    write_dict_json(results, joinpath(results_path, "results.json"))
    print(f"Done {results_path}")

    return pif.models_ithrs[0][0]


def main(params):

    datasets_std = np.array([
        list(itertools.product(*[
            [k], v
        ]))
        for k, v in params["datasets"].items()
    ])
    datasets_std = datasets_std.reshape(int(len(datasets_std.flatten())/2), 2)

    all_combinations = np.array(
        list(
            itertools.product(
                *[datasets_std,
                  params["models_to_use"],
                  params["samplings"]
                  ])
        ),
        dtype=object
    )

    os.makedirs(params["base_path"], exist_ok=True)
    write_dict_json(params, joinpath(params["base_path"], "parameters.json"))

    models = []
    with Pool() as p:
        models = p.map(test, all_combinations)

    model = set(models)
    for model in models:
        if isinstance(model, AEModel):
            params["AEModel_structure"] = {
                "n_inputs": model.n_inputs, "n_first_hidden": model.n_first_hidden, "n_outputs": model.n_outputs}
        if isinstance(model, SelfOrganizingMaps):
            params["SOM_structure"] = {"n_rows": model.n_rows, "n_cols": model.n_cols,
                                       "sigma": model.sigma, "n_dimensions": model.n_dimensions}

    write_dict_json(params, joinpath(params["base_path"], "parameters.json"))

    make_scores_rocs_plots(
        params["root_path"], params["base_path"], towrite=True)

    make_rocs_barplot(params["base_path"], towrite=True)

    time.sleep(100)


def main2():
    dataset, gt = create_dataset_line(300, m_s=[0, 0, 0, 0], centers=[(
        5, 5), (2.5, 2.5), (0, 0), (-2.5, -2.5)], outliers_fraction=0.2)
    # dataset, gt = create_dataset_parabola(300, a_s=[1], centers=[(0, 0)], outliers_fraction=0.)
    dataset = normalize_points(dataset)
    # framework = SOM_Sac(data=dataset, in_th=1, verbose=1, v=NEW_INLIERS)
    # prefs, cluss = framework.fit(mss=30,
    #                             k_max=50,
    #                             delete=False)
    framework = NeuralRansac(dataset,
                             AEModel,
                             it_limit=20,
                             mss=20,
                             # threshold=0.5,
                             epochs=50,
                             delete=False,
                             penalizations=False)
    prefs, cluss = framework.fit(verbose=0)
    old_prefs = framework.preference_matrix
    cluss = transform_clusters_notation(cluss, dataset)
    fig = plot_clusters(cluss, dataset, show=False)
    # plot_consensus(framework.models, dataset, show=True, weights=False)


# k = np.array([0.5, 1, 2, 6, 10, 14, 16],)
k = np.array([1, 3, 5, 10, 25, 50, 75, 100],)

if __name__ == "__main__":
    import warnings
    from datetime import datetime

    warnings.filterwarnings("error")
    start_time = time.time()
    params = {
        "mss": 5,
        "num_models": 1000,
        "training_epochs": 500,
        "base_path": joinpath("test", "results", "july", f"test_{time.strftime('%d-%m-%H-%M-%S')}"),
        "root_path": joinpath(os.path.dirname(os.path.abspath(__file__)), "../"),
        "cool_visualization": False,
        "delete": False,
        "models_to_use": [SOM, AE, LINE],  # SUBS],
        "samplings": [UNIFORM],  # LOCALIZED],
        "datasets": {
            "stair3":   0.003*k,
            "stair4":   0.004*k,
            "star5":    0.009*k,

            # "star11":   0.005*k,
            # "circle3":  0.006*k,
            # "circle4":  0.004*k,

            "circle5":  0.003*k,

            "circles_parable3": 0.2*k/10,
            "lines_rects4": 0.2*k/10
        }
    }
    params2 = {
        "mss": 2,
        "num_models": 1000,
        "training_epochs": 500,
        "base_path": joinpath("test", "results", "july", f"test_{time.strftime('%d-%m-%H-%M-%S')}"),
        "root_path": joinpath(os.path.dirname(os.path.abspath(__file__)), "../"),
        "cool_visualization": False,
        "delete": False,
        "models_to_use": [SOM, AE, LINE],  # SUBS],
        "samplings": [UNIFORM],  # LOCALIZED],
        "datasets": {
            "stair3":   0.003*k,
            "stair4":   0.004*k,
            "star5":    0.009*k,

            # "star11":   0.005*k,
            # "circle3":  0.006*k,
            # "circle4":  0.004*k,

            "circle5":  0.003*k,

            "circles_parable3": 0.2*k/10,
            "lines_rects4": 0.2*k/10
        }
    }
    try:
        main(params)
        # main(params2)
#       orig_dir = joinpath(params["root_path"], "test", "results")
#         for dir in os.listdir(orig_dir):
#             print(dir)
#             make_rocs_barplot(joinpath(orig_dir, dir), towrite=True)
        print("Wii short.mp3")
        os.system("mpg123 -q ~/Downloads/Wii\ short.mp3")
    except:
        print(traceback.format_exc())
        print("fail-trombone-03.mp3")
        os.system("mpg123 -q ~/Downloads/fail-trombone-03.mp3")

    end_time = time.time()
    print(f"Started at: {datetime.fromtimestamp(start_time)}")
    print(f"End at: {datetime.fromtimestamp(end_time)}")
    print(f"Elapsed time: {timedelta(seconds=end_time - start_time)}")
    # os.system("systemctl suspend")
