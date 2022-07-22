from files.utils.constants import *
from files.utils.utility_functions import *
from files.pif.pif import *
from files.classes.neural_models import AEModel
from files.classes.self_organizing_maps import SelfOrganizingMaps
from files.functions.results_extractors import *
from files.functions.results_rocs_maker import *

from multiprocessing.pool import Pool
import time
import collections
import traceback

k = np.array([0.5, 1, 2, 6, 10, 14, 16],)
# k = np.array([1, 3, 5, 10, 25, 50, 75, 100],)
params = {
    "mss": 2,
    "num_models": 2000,
    "training_epochs": 500,
    "base_path": joinpath("results", "pif_benchmark", f"test_{time.strftime('%d-%m-%H-%M-%S')}"),
    "root_path": joinpath(os.path.dirname(os.path.abspath(__file__)), "..", "..", "../"),
    "cool_visualization": False,
    "delete": False,
    "make_rocs_plots": True,
    "make_scores&rocs_plots": True,
    "models_to_use": [LINE],# SOM, AE],  # SUBS],

    "AE_structure": {
        "n_inputs": 2,
        "n_hidden": 0,              # if 0, the structure of the AE will be (2,1,2)
        "n_outputs": 2,
        "activation": torch.tanh # lambda x: x,  # identity activation. Other activations are like torch.tanh or torch.sigmoid
    },

    "SOM_structure": {
        "n_rows": 5,
        "n_cols": 5
    },

    "samplings": [UNIFORM],  # LOCALIZED],
    "datasets": {
        "stair3":   0.003*k,
        "stair4":   0.004*k,
        "star5":    0.009*k,

        "star11":   0.005*k,
        "circle3":  0.006*k,
        "circle4":  0.004*k,

        "circle5":  0.003*k,

        "circles_parable3": 0.2*k/10,
        "lines_rects4": 0.2*k/10
    }
}


def test(inp):
    (dataset_name, threshold), model_name, sampling = inp
    dataset, gt = load_dataset_by_name(dataset_name)

    in_th_str = f"{float(threshold):.3f}".replace('.', '_')
    results_path = joinpath(
        params["base_path"], dataset_name, model_name, f"in_th_{in_th_str}")

    os.makedirs(results_path, exist_ok=True)

    start_time = time.monotonic()
    weights, predict = False, False

    pif = PreferenceIsolationForest(
        data=dataset,
        model_name=model_name,
        in_th=float(threshold),
        sampling=sampling,
        verbose=0,
    )
    scores = pif.anomaly_detection(
        num_models=params["num_models"], mss=params["mss"], params=params)
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

    with Pool() as p:
        p.map(test, all_combinations)

    write_dict_json(params, joinpath(params["base_path"], "parameters.json"))

    if params["make_scores&rocs_plots"]:
        make_scores_rocs_plots(
            params["root_path"], params["base_path"], towrite=True)

    if params["make_rocs_plots"]:
        make_rocs_barplot(params["base_path"], towrite=True)

    time.sleep(100)



if __name__ == "__main__":
    import warnings
    from datetime import datetime

    warnings.filterwarnings("error")
    start_time = time.time()

    try:
        main(params)
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
    os.system("systemctl suspend")
