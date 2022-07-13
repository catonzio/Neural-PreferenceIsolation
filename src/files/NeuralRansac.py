from files.utils.constants import OLD_INLIERS, NEW_INLIERS, NN, SOM, np, time
from files.utils.utility_functions import *
from files.self_organizing_maps.self_organizing_maps import SelfOrganizingMaps
from files.neural_networks.base_class import AEModel
from files.mss_model import MSSModel
from files.pif.voronoi_iforest import *
from datetime import timedelta

'''
    DEPRECATED CLASS. USE PreferenceIsolationForest class
'''

def clustering(pref_m, verbose=0):
    if verbose > 0:
        print('-'*50+"\nClustering of preference matrix")

    num_of_pts = pref_m.shape[0]
    pts = range(num_of_pts)
    clusters = [[i] for i in pts]
    new_idx = pref_m.shape[0]

    pos = {i: i for i in pts}

    x0 = list(itertools.combinations(range(num_of_pts), 2))
    x0 = [(cl_i, cl_j, tanimoto_distance(pref_m[cl_i], pref_m[cl_j]))
          for cl_i, cl_j in x0]

    while pref_m.shape[0] > 1:
        x0.sort(key=lambda y: y[-1])
        cl_0, cl_1, min_distance = x0[0]
        if min_distance >= 1:
            break

        # element-wise min
        new_pf = np.minimum(pref_m[pos[cl_0]], pref_m[pos[cl_1]])
        new_cluster = clusters[pos[cl_0]] + clusters[pos[cl_1]]

        pref_m = np.delete(pref_m, (pos[cl_0], pos[cl_1]), axis=0)
        pref_m = np.vstack((pref_m, new_pf))
        clusters = [c for idx_c, c in enumerate(clusters) if idx_c not in (
            pos[cl_0], pos[cl_1])]  # delete C_i and C_j
        clusters = clusters + [new_cluster]
        new_cluster.sort()

        pos0 = pos[cl_0]
        pos1 = pos[cl_1]
        del pos[cl_0]
        del pos[cl_1]

        for k in pos:
            if pos[k] >= pos0:
                pos[k] -= 1
            if pos[k] >= pos1:
                pos[k] -= 1

        pos[new_idx] = pref_m.shape[0] - 1

        pts = [p for p in pts if p not in (cl_0, cl_1)]
        x0 = [(cl_i, cl_j, d) for cl_i, cl_j, d in x0
              if cl_i not in (cl_0, cl_1) and cl_j not in (cl_0, cl_1)]

        new_comb = [(p, new_idx) for p in pts]
        pts.append(new_idx)
        new_idx += 1
        x1 = [(cl_i, cl_j, tanimoto_distance(pref_m[pos[cl_i]], pref_m[pos[cl_j]]))
              for cl_i, cl_j in new_comb]
        x0 += x1

    return pref_m, clusters


def build_preference_matrix(data, models, verbose=0, images=0):
    if verbose > 0:
        print('-'*50+"\nBuilding preference matrix")
    data = data.copy()

    residuals = np.array([[euclidean_distance(pr, p) for pr, p in zip(
        data, m.predict(data))] for m in models.keys()])

    prefs = np.array([[np.exp(-residual[i]/ithr)
                       # if residual[i] < ithr*3 else 0
                       if arr_contains(consensus, p) else 0
                       for i, p in enumerate(data)
                       ]
                      for residual, (consensus, _, ithr) in zip(residuals, models.values())]).T

    preference_matrix = prefs
    idxs_to_eliminate = np.array(
        [i for i, row in enumerate(prefs) if sum(row) == 0])
    if len(idxs_to_eliminate) == 0:
        return prefs, data
    preference_matrix = np.delete(preference_matrix, idxs_to_eliminate, axis=0)
    data = np.delete(data, idxs_to_eliminate, axis=0)

    if images > 1:
        plt.figure(dpi=200)
        plt.imshow(preference_matrix, cmap="Blues")
        plt.colorbar()
        plt.axis("off")
    return preference_matrix, data


class NeuralRansac:

    def __init__(self, data, model_name, in_th=None, verbose=1, images=0, v=NEW_INLIERS):
        self.original_data = data.copy()
        self.new_data = data.copy()
        self.model_name = model_name
        self.n_dimensions = data.shape[-1]
        if in_th is None:
            self.in_ths = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1]
            if self.model_name == SOM and v==NEW_INLIERS:
                self.in_ths = [0.2, 0.5, 0.75, 1, 1.4826, 1.4826, 1.4826, 2, 2.5]
        else:
            self.in_ths = in_th
        self.verbose = verbose
        self.images = images
        self.v = v
        self.models = None
        self.preference_matrix = None

    def ransac(self, epochs=50, mss=5, k_max=30, delete=False):
        if self.verbose > 0:
            print('-'*50+"\nBuilding RanSac models")
        models = {}
        k = 0
        it = 0
        best_consensus_len = 0

        chars_to_print = 30
        start_time = time.monotonic()

        # fissare le topologie!
        def divide_mss(mss, n): return int(mss/n)
        def sqrt_mss(mss): return int(np.sqrt(mss))
        possible_rows_cols = [(1, mss), (mss, 1), (1, divide_mss(mss, 2)), (divide_mss(mss, 2), 1),
                                      (sqrt_mss(mss), sqrt_mss(mss)), (sqrt_mss(divide_mss(mss, 2)), sqrt_mss(divide_mss(mss, 2))), (1, 20), (20, 1), (7, 7), (3, 3)]
        np.random.shuffle(possible_rows_cols)

        possible_hiddens = [0, 0, 0, 4, 4, 8, 16, 32]
        np.random.shuffle(possible_hiddens)

        sampling_data = self.new_data.copy()
        while it < k_max and len(sampling_data) >= mss:
            it += 1
            try:
                sampled_ds_idxs = localized_sampling(sampling_data, mss)
                sampled_ds = sampling_data[sampled_ds_idxs]
            except Exception as ex:
                print(traceback.format_exc())
                print(ex)
                break

            if self.model_name == SOM:                
                nr, nc = possible_rows_cols[np.random.choice(
                    range(len(possible_rows_cols)))]

                model = SelfOrganizingMaps(
                    nr, nc, data=sampled_ds, n_dimensions=self.n_dimensions, init_type=GRID)

            elif self.model_name == NN:
                model = AEModel(n_first_hidden=possible_hiddens[np.random.choice(
                    range(len(possible_hiddens)))], n_inputs=self.n_dimensions, n_outputs=self.n_dimensions)

            elif self.model_name == MSS:
                model = MSSModel(mss=sampled_ds)

            model.fit(data=sampled_ds, epochs=epochs)

            if isinstance(self.in_ths, list):
                ithr = self.in_ths[np.random.randint(0, len(self.in_ths))]
            else:
                ithr = self.in_ths
            consensus = model.get_inliers(sampling_data, in_th=ithr, v=self.v)
            # print(f"Model {it} in_th: {ithr}. Len of consensus: {len(consensus)}")

            if len(consensus) > 0 and not arr_contains(consensus, np.array([np.inf, np.inf])):
                model.fit(data=consensus, epochs=epochs)
                consensus = model.get_inliers(
                    consensus, in_th=ithr, v=self.v)

            if delete:
                # delete from sampling data all the points of the consensus set
                sampling_data = sampling_data[np.all(
                    np.any((sampling_data-consensus[:, None]), axis=2), axis=0)]

            if len(consensus) > best_consensus_len:
                best_consensus_len = len(consensus)
            models[model] = (consensus, sampled_ds, ithr)

            if self.verbose > 0:
                perc = int(it / k_max * chars_to_print)
                dt = timedelta(seconds=time.monotonic() - start_time)
                sys.stdout.write(f"\rIteration {it}/{k_max}: [" + "="*perc + ">" + "."*(
                    chars_to_print-perc) + f"] ({int(it/k_max*100)}%) ETA: {dt} Len best cons: {best_consensus_len}")
                sys.stdout.flush()
        if self.verbose > 0:
            dt = timedelta(seconds=time.monotonic()-start_time)
            print(f"\rIteration {k_max}/{k_max}: [" + "="*chars_to_print +
                  f"] (100%) ETA: {dt} Len best cons: {best_consensus_len}")

        # sort models depending on consensus size
        # models = dict(sorted(models.items(), key=lambda item: len(item[1]), reverse=True))

        self.models = models
        return models

    def t_linkage(self, epochs=50, mss=5, k_max=30, n_models=-1, sort_clusters=True,
                  delete=False):
        self.ransac(epochs, mss, k_max, delete=delete)
        self.preference_matrix, self.new_data = build_preference_matrix(
            data=self.new_data, models=self.models, verbose=self.verbose, images=self.images)
        new_preferences, clusters = clustering(
            self.preference_matrix, verbose=self.verbose)

        if self.verbose > 0:
            print('-'*50+"\nDone")
        if sort_clusters:
            clusters.sort(key=lambda c: -len(c))

        self.new_data_to_original = {i: j for j, p2 in enumerate(
            self.original_data) for i, p in enumerate(self.new_data) if np.all(p == p2)}

        new_clusters = np.zeros(len(self.original_data))
        for i, cl in enumerate(clusters[:n_models]):
            for j, p in enumerate(cl):
                clusters[i][j] = self.new_data_to_original[p]
            new_clusters[cl] = i+1
        new_clusters = new_clusters.astype(int)

        if self.images > 0:
            plot_clusters(clusters, self.original_data)

        return new_preferences, new_clusters

    def anomaly_detection(self, epochs=50, mss=5, k_max=30, delete=False):
        self.ransac(epochs, mss, k_max, delete=delete)
        self.preference_matrix, self.new_data = build_preference_matrix(
            data=self.new_data, models=self.models, verbose=self.verbose, images=self.images)

        diff_len = len(self.original_data) - len(self.new_data)
        self.preference_matrix = np.append(self.preference_matrix, np.zeros(shape=(diff_len, self.preference_matrix.shape[1])), axis=0)

        if self.verbose > 0:
            print('-'*50+"\nBuilding Voronoi Forest")
        ivor_parameters: dict = {'num_trees': 100, 'max_samples': 256, 'branching_factor': 2, 'metric': 'tanimoto',
                                 'n_jobs': -1}
        voronoi = VoronoiIForest(**ivor_parameters)
        voronoi.fit(self.preference_matrix)
        scores = voronoi.score_samples(self.preference_matrix)
        if self.verbose > 0:
            print('-'*50+"\nDone")
        return scores


if __name__ == "__main__":
    from files.utils.dataset_creator import *
    data, gt = create_dataset_line(
        100, m_s=[-1, 4], centers=[(-2, 1), (1, -1)], outliers_fraction=0.3)
    data = normalize_points(data)

    som_ransac = NeuralRansac(data, SOM, in_th=1, verbose=1, images=2)
    new_prefs, clusters = som_ransac.t_linkage(mss=20)
