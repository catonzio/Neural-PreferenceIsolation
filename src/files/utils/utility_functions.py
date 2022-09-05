from files.utils.constants import *


#    Distances


def euclidean_distance(arr1, arr2, single=False):
    # if isinstance(arr1, torch.Tensor): arr1 = arr1.detach().numpy()# torch.FloatTensor(arr1)
    # if isinstance(arr2, torch.Tensor): arr2 = arr2.detach().numpy()# torch.FloatTensor(arr2)
    if np.any(arr1 == np.inf) or np.any(arr2 == np.inf):
        return np.inf
    if single:
        return np.linalg.norm(np.subtract(arr1, arr2), axis=-1)
    res = (((arr1 - arr2)**2).sum())**(1/2)
    return res


def manhattan_distance(arr1, arr2):
    return np.abs(arr2[0] - arr1[0]) + np.abs(arr2[1] - arr1[1])


def jaccard_distance(arr1, arr2):
    # arr1 = arr1.detach().numpy() if isinstance(arr1, torch.Tensor) else np.array(arr1)
    # arr2 = arr2.detach().numpy() if isinstance(arr2, torch.Tensor) else np.array(arr2)
    arr1, arr2 = arr1.astype(int), arr2.astype(int)

    num = sum(arr1 & arr2)
    denom = sum(arr1 | arr2)
    denom = max(denom, 1e-8)
    return 1 - num / denom


def tanimoto_distance(arr1, arr2):

    # arr1 = arr1.detach().numpy() if isinstance(arr1, torch.Tensor) else np.array(arr1)
    # arr2 = arr2.detach().numpy() if isinstance(arr2, torch.Tensor) else np.array(arr2)

    pq = np.inner(arr1, arr2)

    if pq == 0:
        return 1

    p_square = np.inner(arr1, arr1)
    q_square = np.inner(arr2, arr2)

    t_distance = 1 - pq / (p_square + q_square - pq)
    return t_distance


#    I/O functions


def write_np_array(x, path, name="default.txt", mode="w", dimensions=2):
    with open(joinpath(path, name), mode) as f:
        if dimensions == 1:
            to_write = ",".join(map(str, x))
            f.write(to_write)
        elif dimensions == 2:
            to_write = [",".join(map(str, row)) for row in x]
            to_write = "\n".join(to_write)
            f.write(to_write)


def read_np_array(path, res_type=float):
    with open(path, "r") as f:
        y = f.read().split("\n")
    y = [a.split(",") for a in y]
    y = np.array([[res_type(b) for b in a if b != '']
                 for a in y], dtype=object)
    return y


def write_arr_json(path, arr=None, arrs=None, titles=None, word='arr'):
    assert (arr is None) ^ (arrs is None)

    if arr is not None:
        data = {word: arr.tolist()}
    else:
        titles = [f'{word}{i}' for i, _ in enumerate(
            arrs)] if titles is None else titles
        assert len(titles) == len(arrs)

        data = {title: arr.tolist() if not isinstance(
            arr, list) else arr for title, arr in zip(titles, arrs)}

    write_dict_json(data, path)
    # data = json.dumps(data, indent=4, sort_keys=True, default=str)
    # with open(path, "w") as f:
    #     json.dump(data, f)
    # print(data)
    return data


def read_arr_json(path, return_dict=False):
    with open(path, 'r') as f:
        data = json.load(f)
    # data = json.load(data)
    if return_dict:
        return data

    if len(data.keys()) == 1:
        return np.array(list(data.values())[0])
    else:
        return [np.array(v) for v in data.values()]


def write_dict_json(dic, path):
    with open(path, 'w') as f:
        # json.dump(json.dumps(dic, indent=4, sort_keys=True, default=str), f)
        json.dump(dic, f, indent=4, sort_keys=True, default=str)


def read_dict_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def load_dataset_by_name(name, base_path="", with_outliers=True):
    if "plane" in name.lower() or "sphere" in name.lower() or "paraboloid" in name.lower():
        file_path = joinpath(base_path, "datasets", "3d")
    else:
        file_path = joinpath(base_path, "datasets", "2d",
                         "circles" if "circle" in name.lower() else "lines",
                         "with_outliers" if with_outliers else "no_outliers")
    if with_outliers:
        ds = read_np_array(joinpath(file_path, f"{name}.csv"))
        real_gt = read_np_array(joinpath(file_path, f"{name}_gt.csv"))
    else:
        with open(joinpath(file_path, f"{name}.csv"), 'r') as f:
            content = f.read().split("\n")
            if "" in content:
                content.remove("")
            X = np.array(content[0].split(","), dtype=float)
            y = np.array(content[1].split(","), dtype=float)
            ds = np.dstack((X, y))[0]
        with open(joinpath(file_path, f"{name}_gt.csv"), 'r') as f:
            real_gt = np.array([int(float(l))
                               for l in f.read().split("\n") if l != ''])
    ds = ds.astype(float)
    gt = real_gt.astype(int).reshape(len(ds))
    gt[gt > 1] = 1
    return ds, gt


#    Visualization functions


def plot(inp, title="", s=20, alpha=0.7, c=None, cmap="jet", label="", show=False, new_fig=False, dpi=150, ax=None, equal=False):
    inp = inp.detach().numpy() if isinstance(inp, torch.Tensor) else inp
    if ax is not None:
        ax.scatter(inp[:, 0], inp[:, 1], s=s, alpha=alpha, c=c, label=label)
        if title:
            ax.set_title(title)
    else:
        if new_fig:
            fig = plt.figure(dpi=dpi)
            if title:
                fig.suptitle(title)
        if equal:
            plt.axis('equal')
        plt.scatter(inp[:, 0], inp[:, 1], s=s, alpha=alpha, c=c, cmap=cmap, label=label)
    if label:   # != ""
        if ax is None:
            plt.legend()
        else:
            ax.legend()
    if show:
        plt.show()


def plot_clusters(clusters, data, dpi=100, c='#cc5500', n_models=np.inf, ax=None, show=False):
    if ax is None:
        fig, new_ax = plt.subplots(1, 1, dpi=dpi)
    for i in range(min(max(clusters)+1, n_models)):
        arr = data[np.where(clusters == i)]
        if i == 0:
            plot(arr, label="Outliers", c='gray',
                 alpha=0.3, dpi=dpi, ax=new_ax)
        else:
            plot(arr, label=f"{i}", c=c, ax=new_ax)
    if show:
        plt.show()


def idx_2d_to1d(pos, width):
    x, y = pos
    return x*width + y


def idx_1d_to2d(index, width):
    return int(index / width), index % width


#   Sampling


def get_localized_prob(pts, pt, ni):
    d_squared = np.sum(np.square(np.subtract(pts, pt)), axis=1).astype(float)

    sigma = ni * np.median(np.sqrt(d_squared))
    sigma_squared = sigma ** 2

    prob = np.exp(- (1 / sigma_squared) * d_squared)

    return prob


def localized_sampling(src_pts, k, ni=1 / 3):
    num_of_pts = src_pts.shape[0]
    g = np.random.Generator(np.random.PCG64())

    mss0 = g.choice(num_of_pts, 1)
    prob = get_localized_prob(src_pts, src_pts[mss0], ni)

    prob[mss0] = 0
    prob = (prob / np.sum(prob))

    mss1 = g.choice(num_of_pts, k-1, replace=False, p=prob)
    mss = mss0.tolist() + mss1.tolist()
    return np.array(mss)


#   Miscellaneous


def mad(arr):
    return np.median(np.absolute(arr - np.median(arr)))


def torch_where(cond, x1, x2=None):
    fact1 = (cond * x1)
    return fact1 + (1-cond)*x2 if x2 is not None else fact1


def tensor_from_np(arrs, device=None):
    device = torch.device('cpu') if device is None else device
    res = [torch.FloatTensor(arr).to(device) if arr is not None else None if not isinstance(
        arr, torch.Tensor) else arr for arr in arrs]
    res = [arr.unsqueeze(1) if arr is not None and len(
        arr.shape) == 1 else arr for arr in res]
    if len(res) == 1:
        return res[0]
    else:
        return res


def compute_neighbouring(dataset, threshold=0.1):
    distances = torch.tensor([[euclidean_distance(p1, p2)
                             for p2 in dataset] for p1 in dataset])
    distances -= threshold
    distances[torch.arange(distances.shape[0])[:, None]
              >= torch.arange(distances.shape[1])] = float('inf')
    distances = torch_where(distances <= 0, distances)
    density_values = torch.zeros((len(distances)))
    for i, row1 in enumerate(distances):
        for j, row2 in enumerate(distances):
            if row1[j] != 0 and not torch.isnan(row1[j]):
                density_values[i] += 1
                density_values[j] += 1
    mi, ma = density_values.min(), density_values.max()
    density_values = 1 - ((density_values - mi) / (ma - mi))
    return density_values


def arr_contains(arr, p):
    p = p.detach().numpy() if isinstance(p, torch.Tensor) else p
    arr = arr.detach().numpy() if isinstance(arr, torch.Tensor) else arr
    if len(arr) > 0 and len(p) > 0:
        res = (arr[:] == p)
        if isinstance(res, bool):
            return res
        return np.any(res.all(1))
    else:
        return False


def normalize_points(ds):
    return (ds - np.mean(ds)) / np.std(ds)


#   Errors


def compute_clustering_errors(cluster_mask_est, cluster_mask_gt):
    """
    :param cluster_mask_est: estimated mask
    :param cluster_mask_gt: ground truth mask
    :return: number of error between the ground-truth and estimated cluster masks
    """

    # Get the list of different labels for the ground truth mask and the estimated one.
    label_list_gt = list(set(cluster_mask_gt))
    label_list_est = list(set(cluster_mask_est))

    # Initialize the cost matrix
    cost_matrix = np.empty(shape=(len(label_list_gt), len(label_list_est)))

    # Fill the cost matrix for Munkres (Hungarian algorithm)
    for i, gt_lbl in enumerate(label_list_gt):
        # set as 1 only points with the current gt label
        tmp_gt = np.where(cluster_mask_gt == gt_lbl, 1, 0)
        for j, pred_lbl in enumerate(label_list_est):
            # set as 1 only points with the current pred label
            tmp_est = np.where(cluster_mask_est == pred_lbl, 1, 0)
            cost_matrix[i, j] = np.count_nonzero(
                np.where(tmp_gt + tmp_est == 1, 1, 0))  # rationale below

    # Rationale: we need to fill a cost matrix, thus we need to count the number of errors,
    # namely False Positives (FP) and False Negatives (FN).
    # Both FP and FN are such that gt_lbl != pred_lbl and gt_lbl + pred_lbl == 1
    # [(either gt_lbl == 0 and pred_lbl == 1) or  (either gt_lbl == 1 and pred_lbl == 0)].
    # Note, we don't care about True Positives (TP), where gt_lbl == pred_lbl == 1,
    # and True Negatives (TN), where gt_lbl == pred_lbl == 0, since they are not errors.

    # Run Munkres algorithm
    mnkrs = Munkres()
    # Remark: the cost matrix must have num_of_rows <= num_of_cols (see documentation)
    if len(label_list_gt) <= len(label_list_est):
        match_pairs = mnkrs.compute(np.array(cost_matrix))
        lbl_pred2gt_dict = dict([(j, i) for (i, j) in match_pairs])
    else:  # otherwise, we compute invert rows and columns
        cost_matrix = np.transpose(cost_matrix)
        match_pairs = mnkrs.compute(np.array(cost_matrix))
        lbl_pred2gt_dict = dict(match_pairs)

    # Relabel cluster_mask_est according to the correct label mapping found by Munkres
    clusters_mask_relabeled = []
    for cl_idx in cluster_mask_est:
        if cl_idx in lbl_pred2gt_dict:
            clusters_mask_relabeled.append(lbl_pred2gt_dict[cl_idx])
        else:
            clusters_mask_relabeled.append(-1)
    clusters_mask_relabeled = np.array(clusters_mask_relabeled)

    # Compute the number of errors as the difference of the gt and relabelled arrays
    errors = np.count_nonzero(cluster_mask_gt - clusters_mask_relabeled)

    return errors


def make_roc(gt, scores, show=True, title="", orig_ax=None, to_plot=True):
    # Pos score is 0 because the higher the score, the more it is an outlier
    fpr, tpr, thr = roc_curve(gt, scores, pos_label=0)
    roc_auc: float = auc(fpr, tpr)
    if not to_plot: return roc_auc, fpr, tpr, thr, None

    lw = 2
    if orig_ax is None:
        fig = plt.figure(dpi=100)
        ax = fig.add_subplot(111, aspect='equal')
    else:
        ax = orig_ax
        ax.set_aspect("equal")
    ax.plot(fpr, tpr, color='darkorange', lw=lw,
            label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw,
            linestyle='--', label="Baseline")

    thr[0] = 1
    ax.plot(fpr, thr, color='mediumseagreen',
            lw=lw, linestyle='--', label="Threshold")

    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.yaxis.set_label_position("right")
    # ax.yaxis.tick_right()
    ax.set_title(
        'Receiver Operating Characteristic curve' if title == "" else title)
    ax.legend(loc="lower right")
    # plt.tight_layout()

    if show:
        plt.show()

    return roc_auc, fpr, tpr, thr, fig if orig_ax is None else None


if __name__ == "__main__":
    ds = np.random.uniform(size=(100, 2))
    scores = np.random.uniform(size=(100))
    gt = np.random.randint(0, 2, size=(100))
    fig1 = plt.figure()
    ax = fig1.add_subplot(111)
    *_, fig = make_roc(gt, scores, show=False)  # , ax=ax)
    print(fig is None)
    plt.close(fig)
    # print(localized_sampling(ds, 3))
    # cluss = np.zeros(len(ds))
    # cluss = cluss.astype(int)
    # cluss[4:] = 1
    # print(cluss)
    # plot_clusters(cluss, ds)
    # plt.show()
