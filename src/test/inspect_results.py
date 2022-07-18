import sys
sys.path[0] += "/../.."

from src.files.helper import *
from datetime import datetime
from multiprocessing.pool import Pool
from functools import partial

import traceback

def plot_datasets(inp):
    clusters, datasets_paths = inp
    datasets = []
    for ds in datasets_paths[:-1]:
        datasets.append(read_np_array(ds))
    with open(datasets_paths[-1], 'r') as f:
        datasets_titles = f.read().split("\n")

    n_rows = n_cols = int(np.sqrt(len(datasets)))
    datasets = np.array(datasets, dtype=object).reshape((n_rows, n_cols))
    datasets_titles = np.array(datasets_titles, dtype=object).reshape((n_rows, n_cols))

    for clusters_threshold in clusters:
        entire_fig, e_axs = plt.subplots(n_rows, n_cols, dpi=600)
        entire_fig.tight_layout()
        print(clusters_threshold[0])

        if len(clusters_threshold) > 0:
            try:
                clusters_threshold = np.array(clusters_threshold).reshape((n_rows, n_cols))
            except Exception as ex:
                print(clusters_threshold)
                print(traceback.format_exc())
                print(ex)
            for i, (clusters_row, ds_row, dst_row) in enumerate(zip(clusters_threshold, datasets, datasets_titles)):
                for j, (ds, title, clusters_ds) in enumerate(zip(ds_row, dst_row, clusters_row)):
                    fig, ax = plt.subplots(1, 1, dpi=300)
                    ax.scatter(ds[:,0], ds[:,1], s=15, alpha=0.4)
                    ax.set_title(title + f". DS #{i}{j}. {clusters_ds.split('/')[-3]}")

                    e_axs[i][j].scatter(ds[:,0], ds[:,1], s=15, alpha=0.4)
                    e_axs[i][j].set_title(title + f". DS #({i},{j}). {clusters_ds.split('/')[-3]}")
                    
                    clusters_arr = read_np_array(clusters_ds)
                    for k, cluster in enumerate(clusters_arr):
                        arr = ds[list(map(int, filter(lambda a: a < len(ds), cluster)))]
                        ax.scatter(arr[:,0], arr[:,1], label=f"Cluster{k}", s=5, alpha=0.7)
                        e_axs[i][j].scatter(arr[:,0], arr[:,1], label=f"Cluster{k}", s=5, alpha=0.7)
                    ax.legend()
                    e_axs[i][j].legend()
                    clusters_ds_plot_name = clusters_ds[:clusters_ds.index(".csv")] + ".png"
                    fig.savefig(clusters_ds_plot_name, bbox_inches='tight')
                    plt.close(fig)
            entire_fig.savefig(joinpath(clusters_ds_plot_name[:clusters_ds_plot_name.rfind("/")], "total_clusters.png"), bbox_inches='tight')
            plt.close(entire_fig)

def plot_results(inp):
    total_results, mo_num_points_path = inp
    models_outliers_num_points = read_np_array(mo_num_points_path)
    for results in total_results:
        for i, result in enumerate(results):
            try:
                cl_results = read_np_array(result)
                if len(cl_results) < 1:
                    continue
            except Exception as ex:
                print(result)
                print(traceback.format_exc())
                print(ex)
            # for data_i, cl_results in enumerate(clusters_results):
            x = range(len(cl_results)*2)
            fig, axs = plt.subplots(1, 2, dpi=300, figsize=(10,5))
            
            tick_label_ = ["# Inliers", "# Outliers"]
            tick_label = []
            for k, _ in enumerate(cl_results): tick_label.extend(tick_label_)
            
            tick_label_perc_ = ["% Inliers", "% Outliers"]
            tick_label_perc = []
            for k, _ in enumerate(cl_results): tick_label_perc.extend(tick_label_perc_)

            # plt.figure()#dpi=300)
            fig.suptitle(f"DS #{i}. #Models points: {models_outliers_num_points[i, 0]}. #Outliers points: {models_outliers_num_points[i, 1]}")

            j = 0
            labels = []
            for k, (a, res) in enumerate(zip(x, cl_results.flatten())):
                j += 1 if k%2==0 else 0
                p = axs[0].bar([a], [res])
                labels.append(f"{j}")
                axs[0].bar_label(p, fontsize=8)
            axs[0].set_xticks(x)#, rotation=90)
            axs[0].set_xticklabels(labels, fontsize=7, rotation=45)
            axs[0].set_ylabel("Number of points")
            axs[0].set_xlabel("#Inliers / #Outliers per cluster")
            # axs[0].legend()

            y1 = []
            if cl_results.shape[1] > 0:
                try:
                    y1 = cl_results[:,0] / models_outliers_num_points[i, 0]
                    y2 = cl_results[:,1] / models_outliers_num_points[i, 1] if models_outliers_num_points[i, 1] != 0 else [0 for _ in cl_results]
                    y1 = np.dstack((y1, y2))[0].flatten()
                except Exception as ex:
                    print(traceback.format_exc())
                    print(ex)
                
                for a, res in zip(x, y1):
                    p = axs[1].bar([a], [res])
                    axs[1].bar_label(p, labels=[f"{res:.2f}"] if res!=0 else None, fontsize=8)
                axs[1].set_xticks(x)
                axs[1].set_xticklabels(labels, fontsize=7, rotation=45)
                axs[1].set_ylabel("Percentage of points")
                axs[1].set_xlabel("%Inliers / %Outliers per cluster")

                plt.savefig(result[:result.index(".csv")] + ".png", bbox_inches='tight')
                plt.close(fig)


def plot_test_folder(test_folder_path):
    print("Started at: ", datetime.now().strftime("%H:%M:%S"))

    datasets_names = os.listdir(test_folder_path)
    datasets_names_path = [joinpath(test_folder_path, dataset) for dataset in datasets_names]
    
    clusters_final = []
    results_final = []

    for dataset_path in datasets_names_path:
        assert os.path.exists(dataset_path)

        clusters_to_plot = []
        results_to_plot = []
        datasets_paths = []

        datasets_paths = [joinpath(dataset_path, name) for name in os.listdir(dataset_path) if name.endswith(".csv")]
        datasets_paths.sort()


        dimensions_names = list(filter(lambda x: os.path.isdir(joinpath(dataset_path, x)), os.listdir(dataset_path)))
        dimensions_names_path = [joinpath(dataset_path, dimension_name) for dimension_name in dimensions_names]

        for dimension_name_path in dimensions_names_path:
            assert os.path.exists(dimension_name_path)

            models_names = os.listdir(dimension_name_path)
            models_names_path = [joinpath(dimension_name_path, model) for model in models_names]

            for model_name_path in models_names_path:
                assert os.path.exists(model_name_path)
                # print(model_name_path)
                thresholds = os.listdir(model_name_path)
                thresholds_path = [joinpath(model_name_path, thr) for thr in thresholds]

                for threshold_path in thresholds_path:
                    assert os.path.exists(threshold_path)

                    threshold_datasets = joinpath(threshold_path, "datasets")
                    threshold_results = joinpath(threshold_path, "results")

                    clusters_to_plot.append([joinpath(threshold_datasets, cl) for cl in os.listdir(threshold_datasets) if cl.endswith(".csv")])
                    results_to_plot.append([joinpath(threshold_results, cl) for cl in os.listdir(threshold_results) if cl.endswith(".csv")])
                    for row in clusters_to_plot: row.sort()
                    for row in results_to_plot: row.sort()

                    for clusters, ress in zip(clusters_to_plot, results_to_plot):
                        for cluster, res in zip(clusters, ress):
                            assert os.path.exists(cluster)
                            assert os.path.exists(res)
        clusters_final.append([clusters_to_plot, datasets_paths[:-1]])
        results_final.append([results_to_plot, datasets_paths[-1]])

    with Pool() as p:
        p.map(plot_datasets, clusters_final)
        p.map(plot_results, results_final)
        # plot_datasets(clusters_to_plot, datasets_paths[:-1])
        # plot_results(results_to_plot, datasets_paths[-1])
        # with Pool() as p:
        #     p.map(partial(plot_datasets, clusters_to_plot) , datasets_paths[:-1])
        # with Pool() as p:
        #     p.map(partial(plot_results, results_to_plot), datasets_paths[-1])
        

    # print(datasets_paths)
    # print(clusters_to_plot)
    # print(results_to_plot)
                
# plot_test_folder(joinpath(ROOT_DIR, "images", "report_finale", "lines", "test_02-05-23-05-49"))
plot_test_folder(joinpath(ROOT_DIR, "images", "som_experiments", "lines", "test_13-05-19-12-29"))
# plot_test_folder(joinpath(ROOT_DIR, "images", "report_finale", "sin", "test_02-05-16-59-49"))
