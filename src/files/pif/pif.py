from files.utils.constants import *
from files.utils.utility_functions import *

from files.classes.self_organizing_maps import SelfOrganizingMaps
from files.classes.neural_models import AEModel, NeuralNetwork
from files.classes.base_models import *
from files.classes.base_models import PlaneEstimator
from files.classes.mss_model import MSSModel

from files.pif.voronoi_iforest import *
from datetime import timedelta
from numbers import Number


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


def build_preference_matrix(data, models, threshold, verbose=0, images=0):
    if verbose > 0:
        print('-'*50+"\nBuilding preference matrix")

    residuals = np.array([m.get_residuals(data) for m in models]).astype(float)

    preference_matrix = np.array([
        (r < 3*threshold).astype(float) *  # 1 if r[i] < 3*t, 0 otherwise
        # gaussian preference value
        np.exp(-r**2/threshold) for r in residuals
    ]).T

    if images > 1:
        plt.figure(dpi=200)
        plt.imshow(preference_matrix, cmap="Blues")
        plt.colorbar()
        plt.axis("off")
    return preference_matrix# , data


ivor_parameters: dict = {'num_trees': 100, 'max_samples': 256, 'branching_factor': 2, 'metric': 'tanimoto',
                         'n_jobs': -1}


class PreferenceIsolationForest:

    def __init__(self, data, model_name, ivor_parameters=ivor_parameters, verbose=1, images=0, sampling=UNIFORM):
        self.original_data = data.copy()
        self.new_data = data.copy()
        self.model_name = model_name
        self.n_dimensions = data.shape[-1]
        self.verbose = verbose
        self.images = images
        self.sampling = sampling
        self.voronoi = VoronoiIForest(**ivor_parameters)
        self.models = None
        self.preference_matrix = None

    def build_models_OLD(self, num_models=30, mss=2, delete=False):
        if self.verbose > 0:
            print('-'*50+"\nBuilding RanSac models")
        models_ithrs = []
        k = 0
        it = 0

        chars_to_print = 30
        start_time = time.monotonic()

        sampling_data = self.new_data.copy()
        mss = 2 if self.model_name == LINE else 3 if self.model_name == CIRCLE else mss

        while it < num_models and len(sampling_data) >= mss:
            it += 1
            try:

                # localized_sampling(sampling_data, mss)
                sampled_ds_idxs = np.random.randint(
                    0, len(sampling_data), size=(mss)) if self.sampling == UNIFORM else localized_sampling(sampling_data, mss)
                sampled_ds = sampling_data[sampled_ds_idxs]
            except Exception as ex:
                print(traceback.format_exc())
                print(ex)
                break

            if self.model_name == LINE:
                model = LineEstimator()
            elif self.model_name == CIRCLE:
                model = CircleEstimator()
            elif self.model_name == MIXED:
                if np.random.uniform() < 0.5:
                    model = LineEstimator()
                else:
                    model = CircleEstimator()
            elif self.model_name == SOM:
                # nr, nc = [1, mss] if np.random.rand() < 0.5 else [mss, 1]
                nr, nc = 5, 5
                model = SelfOrganizingMaps(
                    nr, nc, data=sampled_ds, n_dimensions=self.n_dimensions, init_type=GRID)
            elif self.model_name == AE:
                model = AEModel(n_inputs=self.n_dimensions,
                                n_outputs=self.n_dimensions)
            elif self.model_name == SUBS:
                model = MSSModel(mss=sampled_ds)

            model.fit(data=sampled_ds, epochs=self.epochs)

            if isinstance(self.in_ths, list) or isinstance(self.in_ths, np.ndarray):
                ithr = self.in_ths[np.random.randint(0, len(self.in_ths))]
            else:
                ithr = self.in_ths

            if delete:
                # delete from sampling data all the points of the consensus set
                sampling_data = sampling_data[np.all(
                    np.any((sampling_data-consensus[:, None]), axis=2), axis=0)]

            models_ithrs.append([model, ithr])

            if self.verbose > 0:
                perc = int(it / num_models * chars_to_print)
                dt = timedelta(seconds=time.monotonic() - start_time)
                sys.stdout.write(f"\rIteration {it}/{num_models}: [" + "="*perc + ">" + "."*(
                    chars_to_print-perc) + f"] ({int(it/num_models*100)}%) ETA: {dt}")
                sys.stdout.flush()
        if self.verbose > 0:
            dt = timedelta(seconds=time.monotonic()-start_time)
            print(f"\rIteration {num_models}/{num_models}: [" + "="*chars_to_print +
                  f"] (100%) ETA: {dt}")

        self.models_ithrs = np.array(models_ithrs)
        return self.models_ithrs

    def build_models(self, **params):
        if bool(params):
            params = params["params"]
        if self.verbose > 0:
            print('-'*50+"\nBuilding RanSac models")

        chars_to_print = 30
        start_time = time.monotonic()

        sampling_data = self.new_data.copy()
        mss = 2 if self.model_name == LINE \
            else 3 if (self.model_name == CIRCLE or self.model_name == PLANE) \
            else params["mss"]
        num_models = params["num_models"]

        sampled_ds_idxs = np.random.randint(
            0, len(sampling_data), size=(num_models, mss)) if self.sampling == UNIFORM else np.array([localized_sampling(sampling_data, mss) for _ in range(num_models)])

        sampled_ds_s = np.array([sampling_data[idxs]
                                for idxs in sampled_ds_idxs])
        if self.model_name == AE:
            dev = torch.device('cpu')
            sampled_ds_s = tensor_from_np(sampled_ds_s, device=dev)
        models = np.array([
            LineEstimator() if self.model_name == LINE else
            CircleEstimator() if self.model_name == CIRCLE else
            PlaneEstimator() if self.model_name == PLANE else

            SelfOrganizingMaps(n_rows=params["SOM_structure"]["n_rows"],
                               n_cols=params["SOM_structure"]["n_cols"],
                               data=sampled_ds_s[i], n_dimensions=self.n_dimensions, init_type=GRID) if self.model_name == SOM else

            NeuralNetwork(neurons=params["AE_structure"]["neurons"],
                          activation=params["AE_structure"]["activation"]) if self.model_name == AE else

            MSSModel(mss=sampled_ds_s[i])
            for i in range(params["num_models"])])

        def build_model(inp):
            it, (model, sampled_ds) = inp
            model.fit(data=sampled_ds, epochs=params["training_epochs"])
            if self.verbose > 0:
                perc = int(it / num_models * chars_to_print)
                dt = timedelta(seconds=time.monotonic() - start_time)
                sys.stdout.write(f"\rIteration {it}/{num_models}: [" + "="*perc + ">" + "."*(
                    chars_to_print-perc) + f"] ({int(it/num_models*100)}%) ETA: {dt}")
                sys.stdout.flush()

        list(map(build_model, enumerate(zip(models, sampled_ds_s))))
        if self.verbose > 0:
            dt = timedelta(seconds=time.monotonic()-start_time)
            print(f"\rIteration {num_models}/{num_models}: [" + "="*chars_to_print +
                  f"] (100%) ETA: {dt}")

        self.models = np.array(models)
        return self.models

    def t_linkage(self, num_models=30, mss=2, n_models=-1, sort_clusters=True,
                  delete=False):
        if self.verbose > 0:
            print("T-Linkage")
        if self.models_ithrs is None:
            self.build_models(num_models, delete=delete)
        else:
            if self.verbose > 0:
                print('-'*50+"\nNot building models Pool because already generated.")
        if self.preference_matrix is None:
            self.preference_matrix, self.new_data = build_preference_matrix(
                data=self.new_data, models_ithrs=self.models_ithrs, verbose=self.verbose, images=self.images)
        else:
            if self.verbose > 0:
                print(
                    '-'*50+"\nNot building Preference Matrix because already generated.")
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

    def anomaly_detection(self, in_th=1, **params):
        if bool(params):
            params = params["params"]
        if self.verbose > 0:
            print("Anomaly Detection")
        if self.models is None:
            self.build_models(params=params)
        else:
            if self.verbose > 0:
                print('-'*50+"\nNot building models Pool because already generated.")

        if self.preference_matrix is None:
            self.preference_matrix, self.new_data = build_preference_matrix(
                data=self.new_data, models=self.models, threshold=in_th, verbose=self.verbose, images=self.images)
        else:
            if self.verbose > 0:
                print(
                    '-'*50+"\nNot building Preference Matrix because already generated.")

        # diff_len = len(self.original_data) - len(self.new_data)
        # self.preference_matrix = np.append(self.preference_matrix, np.zeros(
        #     shape=(diff_len, self.preference_matrix.shape[1])), axis=0)

        if self.verbose > 0:
            print('-'*50+"\nBuilding Voronoi Forest")
        self.voronoi.fit(self.preference_matrix)
        scores = self.voronoi.score_samples(self.preference_matrix)
        if self.verbose > 0:
            print('-'*50+"\nDone")
        return scores


if __name__ == "__main__":
    ds, gt = load_dataset_by_name("circle5")
    # plot_clusters(gt, ds, show=True)

    st = time.time()
    pif = PreferenceIsolationForest(ds, SUBS, in_th=0.008, epochs=100)
    scores = pif.anomaly_detection(num_models=1000)
    print(f"Elapsed time: {time.time() - st}")
    fig, (ax1, ax2) = plt.subplots(1, 2, dpi=150)
    ax1.scatter(ds[:, 0], ds[:, 1], c=scores, cmap="jet")
    make_roc(gt, scores, ax=ax2)
    plt.show()
