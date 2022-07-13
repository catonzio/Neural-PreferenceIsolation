import numpy as np


def rect(x, m, q): return m*x + q
def parabola(x, a, b, c): return a*x**2 + b*x + c
def circle(x, a, b, c): return -x**2 - a*x - c - b


def rot_matrx(theta): return np.array(
    [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def generate_funct_dataset(num_points, func, noise=0, range_=(-2, 2), return_type=np.ndarray,
                           outliers_fraction=0, center=(0, 0)):
    x = np.random.uniform(range_[0], range_[1], int(
        num_points*(1-outliers_fraction)))
    y = np.array([func(i) for i in x])

    x += center[0]
    y += center[1]

    x = np.append(x, np.random.uniform(
        range_[0]*3, range_[1]*3, num_points-len(x)))
    y = np.append(y, np.random.uniform(
        range_[0]*3, range_[1]*3, num_points-len(y)))

    y_noise = y + np.random.normal(0, noise, len(y))

    ds = np.dstack((x, y_noise))[0]

    if return_type is torch.Tensor:
        return torch.FloatTensor(ds)
    else:
        return ds


def create_dataset_line(num_points, noise_var=0.1, m_s=None, q_s=None, ranges_x=None,
                        centers=None, rot_angle=None, outliers_fraction=0, shuffle=False, outliers_displacement=1.5):
    m_s = [1] if m_s is None else m_s
    q_s = [0 for _ in m_s] if q_s is None else q_s
    ranges_x = [(-2, 2) for _ in m_s] if ranges_x is None else ranges_x
    centers = [(0, 0) for _ in m_s] if centers is None else centers
    rot_angle = [0 for _ in m_s] if rot_angle is None else rot_angle
    ds = None
    gt = []
    for i, (m, q, angle, range_x, center) in enumerate(zip(m_s, q_s, rot_angle, ranges_x, centers)):
        x = np.linspace(range_x[0], range_x[1], int(
            num_points*(1-outliers_fraction)/len(m_s)))
        y = np.array([rect(i, m, q) for i in x])
        gt = gt + [i+1 for _ in x]

        x += center[0]
        y += center[1]

        if i == 0:
            ds = np.dstack((x, y))[0]
        else:
            ds = np.vstack((ds, np.dstack((x, y))[0]))

    # mean_centers = np.mean(centers, axis=0)
    maxx = np.max(ds)
    minn = np.min(ds)
    best = max(np.abs(maxx), np.abs(minn))

    outliers = np.random.uniform(-best*outliers_displacement, best*outliers_displacement,
                                 size=(int(num_points*outliers_fraction), 2))  # + mean_centers[0]
    # outliers_y = np.random.uniform(-best*outliers_displacement, best*outliers_displacement, size=(int(num_points*outliers_fraction))) # + mean_centers[1]
    # outliers = np.dstack((outliers_x, outliers_y))[0]
    gt = gt + [0 for _ in outliers]

    ds = np.vstack((ds, outliers))
    ds[:, 1] += np.random.normal(0, noise_var, len(ds))

    if shuffle:
        np.random.shuffle(ds)

    return ds, np.array(gt)


def create_dataset_parabola(num_points, noise_var=0.1, a_s=None, b_s=None, c_s=None, ranges_x=None,
                            centers=None, rot_angle=None, outliers_fraction=0, shuffle=False, outliers_displacement=1.5):
    a_s = [1] if a_s is None else a_s
    b_s = [0 for _ in a_s] if b_s is None else b_s
    c_s = [0 for _ in a_s] if c_s is None else c_s
    ranges_x = [(-2, 2) for _ in a_s] if ranges_x is None else ranges_x
    centers = [(0, 0) for _ in a_s] if centers is None else centers
    rot_angle = [0 for _ in a_s] if rot_angle is None else rot_angle
    ds = None
    gt = []
    for i, (a, b, c, angle, range_x, center) in enumerate(zip(a_s, b_s, c_s, rot_angle, ranges_x, centers)):
        x = np.random.uniform(range_x[0], range_x[1], int(
            num_points*(1-outliers_fraction)/len(a_s)))
        y = np.array([parabola(i, a, b, c) for i in x])
        gt = gt + [i+1 for _ in x]

        if i == 0:
            ds = np.dstack((x, y))[0]
            ds = rotate_dataset(ds, angle)
            ds[:, 0] += center[0]
            ds[:, 1] += center[1]
        else:
            ds_tmp = np.dstack((x, y))[0]
            ds_tmp = rotate_dataset(ds_tmp, angle)
            ds_tmp[:, 0] += center[0]
            ds_tmp[:, 1] += center[1]
            ds = np.vstack((ds, ds_tmp))

    # mean_centers = np.mean(centers, axis=0)
    maxx = np.max(ds)
    minn = np.min(ds)
    best = max(np.abs(maxx), np.abs(minn))

    outliers = np.random.uniform(-best*outliers_displacement, best*outliers_displacement,
                                 size=(int(num_points-len(ds)), 2))  # + mean_centers[0]
    # outliers_y = np.random.uniform(-best*outliers_displacement, best*outliers_displacement, size=(int(num_points-len(ds)))) # + mean_centers[1]
    # outliers = np.dstack((outliers_x, outliers_y))[0]
    gt = gt + [0 for _ in outliers]
    gt = np.array(gt)

    ds = np.vstack((ds, outliers))
    ds[:, 1] += np.random.normal(0, noise_var, len(ds))

    if shuffle:
        p = np.random.permutation(num_points)
        ds, gt = ds[p], gt[p]
        # np.random.shuffle(ds)

    return ds, gt


def create_dataset_circle(num_points, radiuses=None, centers=None, noise_var=0.1, range_x=(-2, 2),
                          outliers_fraction=0.0, outliers_displacement=1.5, shuffle=True):
    radiuses = [1] if radiuses is None else radiuses
    centers = [(0, 0) for _ in radiuses]

    data = []
    for r, center in zip(radiuses, centers):
        c_x, c_y = center  # np.random.uniform(range_x)
        for theta in np.random.uniform(0, 60, int(num_points*(1-outliers_fraction)/len(radiuses))):
            data.append(
                np.array([c_x + np.cos(theta) * r, c_y + np.sin(theta) * r]))

    ds = np.array(data)

    maxx = np.max(ds)
    minn = np.min(ds)
    best = max(np.abs(maxx), np.abs(minn))

    outliers_x = np.random.uniform(-best*outliers_displacement, best *
                                   outliers_displacement, size=(int(num_points-len(ds))))  # + mean_centers[0]
    outliers_y = np.random.uniform(-best*outliers_displacement, best *
                                   outliers_displacement, size=(int(num_points-len(ds))))  # + mean_centers[1]
    outliers = np.dstack((outliers_x, outliers_y))[0]

    ds = np.vstack((ds, outliers))
    ds[:, 1] += np.random.normal(0, noise_var, len(ds))

    if shuffle:
        np.random.shuffle(ds)

    return ds


def generate_outliers(num, min_, max_, dims=2):
    res = np.array([np.random.uniform(min_[i]*3, max_[i]*3, num)
                   for i in range(dims)])
    res = res.reshape(num, dims)
    return res


def generate_test_datasets(data, num_ds, noise_range=(0, 0.5), outliers_range=(0, 0.6)):
    noises = np.linspace(noise_range[0], noise_range[1], num_ds)
    outliers_factors = np.linspace(
        outliers_range[0], outliers_range[1], num_ds)

    combinations = np.array(np.meshgrid(
        noises, outliers_factors)).T.reshape(-1, 2)

    datasets = []
    datasets_titles = []
    datasets_gt = []    # ground truth, the points belonging to the models
    for i, (noise, factor) in enumerate(combinations):
        datasets_titles.append(f"#{i}. n: {noise:.3f} o: {factor:.3f}")
        print(
            f"Generating {i+1}-th dataset. Noise: {noise:.3f}. Outliers factor: {factor:.3f}")
        ds = data.copy()
        # such that it will be the `factor` percentage of the total ds length
        num_outliers = int(len(data)*factor)  # /(1-factor))
        outliers = generate_outliers(num_outliers, min_=[min(ds[:, 0]), min(ds[:, 1])],
                                     max_=[max(ds[:, 0]), max(ds[:, 1])], dims=data.shape[1])
        ds = np.vstack((ds, outliers))
        data_noise = np.random.normal(0, noise, len(ds))
        ds[:, 1] += data_noise

        # ds = normalize_points(ds)
        datasets_gt.append(ds[:data.shape[0]])

        # np.random.shuffle(ds)
        datasets.append(ds)

    return np.array(datasets, dtype=object), np.array(datasets_titles), np.array(datasets_gt)


def rotate_dataset(ds, theta):
    return ds @ rot_matrx(theta)


def normalize_points(ds):
    return (ds - np.mean(ds)) / np.std(ds)


def normalize_points_old(x, rescale=False):
    """
        Input:
            x   --> list of points
        Output:
            xn  --> normalized list of points
            s   --> s such that xn = x * s
    """
    is_tensor = False
    if isinstance(x, torch.Tensor):
        is_tensor = True
        x = x.detach().numpy()

    x_bar = np.mean(x, 0)
    xc = x - x_bar
    rho = np.sqrt(np.sum(xc**2, axis=1))
    rho_bar = np.mean(rho)
    s = np.sqrt(2) / rho_bar
    xn = s*xc

    if is_tensor:
        xn = torch.FloatTensor(xn)
    reverse_factor = (1 + x_bar*s)/s

    if rescale:
        return xn, reverse_factor
    else:
        return xn


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ds1, gt1 = create_dataset_line(
        500, m_s=[1, -1], centers=[(2, 2), (-2, 2)], outliers_fraction=0.3)
    ds1, gt1 = create_dataset_parabola(
        500, a_s=[1, -1], centers=[(2, 2), (-2, 2)], outliers_fraction=0.3)
    ds1 = create_dataset_circle(
        500, radiuses=[1, -1], centers=[(2, 2), (-2, 2)], outliers_fraction=0.3)
    plt.scatter(ds1[:, 0], ds1[:, 1])
    plt.show()
