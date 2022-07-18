from files.utils.utility_functions import *

'''
nameDatasets = ["circle3", "circle4", "circle5", "star5", "star11", ...
                "stair3", "stair4"];
epsis = [0.06, 0.035, 0.025, 0.05, 0.035, 0.025, 0.025];
'''

# def read_np_array(path):
#     with open(path, 'r') as f:
#         lines = [l for l in f.read().split("\n") if len(l) > 0]
#     X = np.array([float(l) for l in lines[0].split(",")])
#     y = np.array([float(l) for l in lines[1].split(",")])
#     return np.dstack((X, y))[0]

def dataset_with_outliers(path, outliers_fraction=0):
    arr = read_np_array(path)
    num_outliers = int(len(arr)*outliers_fraction)

    max_x, max_y = np.max(arr, axis=0)
    min_x, min_y = np.min(arr, axis=0)

    outliers_x = np.random.uniform(min_x, max_x,
                                 size=(num_outliers, 1))
    outliers_y = np.random.uniform(min_y, max_y,
                                 size=(num_outliers, 1))
    outliers = np.hstack((outliers_x, outliers_y))

    gt = np.zeros(shape=(len(arr)+num_outliers), dtype=int)
    gt[:len(arr)] = 1

    return np.vstack((arr, outliers)), gt

def create_datasets_with_outliers():
    base_path = joinpath("..", "datasets", "2d")
    lines_path = joinpath(base_path, "lines")
    circles_path = joinpath(base_path, "circles")

    lines_names = ["stair3", "stair4", "star5", "star11"]
    circles_names = ["circle3", "circle4", "circle5"]

    for name in lines_names:
        arr, gt = dataset_with_outliers(joinpath(lines_path, "no_outliers", f"{name}.csv"), outliers_fraction=1)
        write_np_array(arr, joinpath(lines_path, "with_outliers"), name=f"{name}.csv")
        with open(joinpath(lines_path, "with_outliers", f"{name}_gt.csv"), 'w') as f:
            f.write("\n".join(gt.astype(str)))

    for name in circles_names:
        arr, gt = dataset_with_outliers(joinpath(circles_path, "no_outliers", f"{name}.csv"), outliers_fraction=1)
        write_np_array(arr, joinpath(circles_path, "with_outliers"), name=f"{name}.csv")
        with open(joinpath(circles_path, "with_outliers", f"{name}_gt.csv"), 'w') as f:
            f.write("\n".join(gt.astype(str)))

def main():
    name = "star11"
    ds = read_np_array(joinpath("datasets", "2d", "lines", "with_outliers", f"{name}.csv"))
    gt = read_np_array(joinpath("datasets", "2d", "lines", "with_outliers", f"{name}_gt.csv")).astype(int)
    gt = gt.reshape(gt.shape[0])
    plot_clusters(gt, ds, show=True)

if __name__ == "__main__":
    main()