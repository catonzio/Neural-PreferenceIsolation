from files.utils.constants import *
from files.utils.utility_functions import *


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

        final_results[mn][ds_name][in_th]["roc_auc"] = params["roc_auc"]
        final_results[mn][ds_name][in_th]["scores_path"] = params["scores_path"]
        final_results[mn][ds_name][in_th]["exec_time_seconds"] = params["exec_time_seconds"]

    for mn, dict1 in final_results.items():
        for ds_name, dict2 in dict1.items():
            thresholds = list(dict2.keys())
            roc_aucs = [dict2[v]["roc_auc"]
                        for v in thresholds]
            final_results[mn][ds_name]["roc_auc_median"] = np.median(roc_aucs)
            final_results[mn][ds_name]["roc_auc_mean"] = np.mean(roc_aucs)
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

        final_results[ds_name][mn][in_th]["roc_auc"] = params["roc_auc"]
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
