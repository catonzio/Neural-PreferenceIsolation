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


def read_parameters(path="parameters.json"):
    params = read_dict_json(path)
    params["base_path"] = joinpath(
        params["root_path"], "results", "pif_benchmark", f"test_{time.strftime('%d-%m-%H-%M-%S')}")
    params["datasets"] = {a: v*np.array(params["std_multiples"])
                          for a, v in params["datasets_std"].items()}
    act = params["AE_structure"]["activation"].lower()
    params["AE_structure"]["activation"] = \
        torch.tanh if act == "tanh" else \
        torch.sigmoid if act == "sigmoid" else \
        lambda x: x
    params["models_to_use"] = list(
        map(lambda x: x.lower(), params["models_to_use"]))

    return params


params = read_parameters()


def test(inp):
    (dataset_name, threshold), model_name = inp
    dataset, gt = load_dataset_by_name(
        name=dataset_name, base_path=params["root_path"])
    if params["datasets_normalization"]:
        dataset = normalize_points(dataset)

    in_th_str = f"{float(threshold):.3f}".replace('.', '_')
    results_path = joinpath(
        params["base_path"], dataset_name, model_name, f"in_th_{in_th_str}")

    os.makedirs(results_path, exist_ok=True)

    start_time = time.monotonic()
    weights, predict = False, False

    pif = PreferenceIsolationForest(
        data=dataset,
        model_name=model_name,
        verbose=0,
        ivor_parameters=params["ivor_parameters"]
    )
    scores = pif.anomaly_detection(in_th=float(threshold), params=params)
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
        gt, scores, show=False, title=f"{dataset_name} Model: {model_name}, Thr: {in_th_str}")
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
                  ])
        ),
        dtype=object
    )

    os.makedirs(params["base_path"], exist_ok=True)
    write_dict_json(params, joinpath(params["base_path"], "parameters.json"))

    with Pool(params["n_jobs"]) as p:
        p.map(test, all_combinations)

    write_dict_json(params, joinpath(params["base_path"], "parameters.json"))
    print("Saved parameters.json at ", joinpath(
        params["base_path"], "parameters.json"))
    if params["make_scores&rocs_plots"]:
        make_scores_rocs_plots(
            params["root_path"], params["base_path"], towrite=True)
        print("Made scores rocs plots")

    if params["make_rocs_plots"]:
        make_rocs_barplot(params["base_path"], towrite=True)
        print("Made rocs barplot")

    time.sleep(100)


if __name__ == "__main__":
    import warnings
    from datetime import datetime

    warnings.filterwarnings("error")
    start_time = time.time()

    try:
        main(params)
        print("Wii short.mp3")
        os.system(f"mpg123 -q {params['root_path']}/datasets/Wii\ short.mp3")
    except:
        print(traceback.format_exc())
        print("fail-trombone-03.mp3")
        os.system(f"mpg123 -q {params['root_path']}/datasets/fail-trombone-03.mp3")

    end_time = time.time()
    print(f"Started at: {datetime.fromtimestamp(start_time)}")
    print(f"End at: {datetime.fromtimestamp(end_time)}")
    print(f"Elapsed time: {timedelta(seconds=end_time - start_time)}")
    # os.system("systemctl suspend")
