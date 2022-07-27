from files.utils.constants import *
from files.utils.utility_functions import *
from files.utils.dataset_creator import *
import plotly
import plotly.graph_objs as go
from numpy import cos, sin


def plot_3d(x, y, z):
    trace = go.Scatter3d(x=x, y=y, z=z, mode='markers', marker={
        'size': 5,
        'opacity': 0.8,
    },
    )

    # Configure the layout.
    layout = go.Layout(
        width=500,
        height=500,
        paper_bgcolor="LightSteelBlue"
        # margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
    )

    data = [trace]

    plot_figure = go.Figure(data=data, layout=layout)

    plotly.offline.iplot(plot_figure)

def plot3d(ds, gt=None):
    gt = gt if gt is not None else np.ones(shape=(len(ds))).astype(int)
    fig = go.Figure()
    add_trace(fig, ds[gt==1], label="Inliers")
    add_trace(fig, ds[gt==0], label="Outliers")
    fig.update_layout(
        autosize=False,
        # width=1000,
        # height=1000,
        paper_bgcolor="LightSteelBlue",
        showlegend=True
    )
    fig.show()

def add_trace(fig, data, label=None):
    fig.add_trace(go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker={
            'size': 3,
            'opacity': 0.8,
            # 'color': 'red',
        },
        name=label
    ))


def R(theta, u):
    return [[cos(theta) + u[0]**2 * (1-cos(theta)),
             u[0] * u[1] * (1-cos(theta)) - u[2] * sin(theta),
             u[0] * u[2] * (1 - cos(theta)) + u[1] * sin(theta)],
            [u[0] * u[1] * (1-cos(theta)) + u[2] * sin(theta),
             cos(theta) + u[1]**2 * (1-cos(theta)),
             u[1] * u[2] * (1 - cos(theta)) - u[0] * sin(theta)],
            [u[0] * u[2] * (1-cos(theta)) - u[1] * sin(theta),
             u[1] * u[2] * (1-cos(theta)) + u[0] * sin(theta),
             cos(theta) + u[2]**2 * (1-cos(theta))]]


def create_plane(num_points=1000, normal=None, point=None, of=0.5, noise_std=0.1):
    normal = np.array([1, 1, 2]) if normal is None else normal
    point = np.array([0, 0, 0]) if point is None else point
    half_points = int(num_points*of)

    d = -point.dot(normal)

    points = np.random.uniform(-2, 2, (num_points-half_points, 2))
    x, y = points[:, 0], points[:, 1]

    z = (-normal[0] * x - normal[1] * y - d) * 1. / normal[2]
    # z2 = (-normal2[0] * x - normal2[1] * y - d2) * 1. / normal2[2]
    model1 = np.dstack((x, y, z))[0]
    model1 += np.random.normal(0, noise_std, (len(model1), 3))
    # model2 = np.dstack((x, y, z2))[0]
    outliers = np.random.uniform(-2, 2, size=(half_points, 3))

    res = np.vstack((model1, outliers))
    gt = np.zeros(shape=(len(res,)))
    gt[:len(model1)] = 1
    return res, gt.astype(int)


def create_paraboloide(num_points=1000, of=0.5, noise_std=0.1):
    half_points = int(num_points*of)
    ds = np.empty((0, 3))
    for i, theta in enumerate(np.random.uniform(0, 360, int((num_points-half_points)/5))):
        matrx = R(theta, [0, 1, 0])
        parabola, gt = create_dataset_parabola(5, a_s=[1], noise_var=0)
        parabola = np.hstack(
            (parabola, np.array([0 for _ in parabola]).reshape(-1, 1)))
        parabola += np.random.normal(0, noise_std, (len(parabola), 3))
        parabola[:, 1] -= 2
        rotated_par = np.matmul(parabola, matrx)
        ds = np.vstack((ds, rotated_par))

    outliers = np.random.uniform(-2, 2, (half_points, 3))
    res = np.vstack((ds, outliers))
    gt = np.zeros(shape=(len(res,)))
    gt[:len(ds)] = 1
    return res, gt.astype(int)


def create_sphere(num_points=1000, of=0.5, noise_std=0.1):
    ds = np.empty((0, 3))
    half_points = int(num_points*of)

    for i, theta in enumerate(np.random.uniform(0, 360, int((num_points-half_points)/5))):
        matrx = R(theta, [0, 1, 0])
        parabola = create_dataset_circle(5, radiuses=[2], noise_var=0)
        parabola = np.hstack(
            (parabola, np.array([0 for _ in parabola]).reshape(-1, 1)))
        parabola += np.random.normal(0, noise_std, (len(parabola), 3))
        # parabola[:,1] -= 2
        rotated_par = np.matmul(parabola, matrx)
        ds = np.vstack((ds, rotated_par))

    outliers = np.random.uniform(-2, 2, (half_points, 3))
    res = np.vstack((ds, outliers))
    gt = np.zeros(shape=(len(res,)))
    gt[:len(ds)] = 1
    return res, gt.astype(int)


if __name__ == "__main__":
    ds, gt = create_sphere(num_points=1000, of=0.5)
    print(ds.shape)
    plot3d(ds, gt)
