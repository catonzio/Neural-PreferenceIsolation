from scipy import optimize
from files.utils.constants import *
from files.utils.utility_functions import *


class BaseEstimator:

    def __init__(self, class_type):
        self.class_type = class_type

    # epochs is for consistency
    def fit(self, data, epochs=0):
        pass

    def get_residuals(self, data):
        pass

    def get_inliers(self, data, in_th=1, v=None):
        pass


class LineEstimator(BaseEstimator):

    def __init__(self):
        super(LineEstimator, self).__init__(class_type=LINE)
        self.slope = np.inf
        self.intercept = np.inf

    # epochs is for consistency
    def fit(self, data, epochs=0):
        assert len(data) == 2
        p1, p2 = data
        # if two points have same x-coord
        if np.abs(p1[0] - p2[0]) < 1e-3:
            self.slope = np.inf
            self.intercept = p1[0]
        # if two points have same y-coord
        elif np.abs(p1[1] - p2[1]) < 1e-3:
            self.slope = 0
            self.intercept = p1[1]
        else:
            self.slope = (p1[1] - p2[1]) / (p1[0] - p2[0])
            self.intercept = -self.slope*p1[0] + p1[1]
        return self

    def get_residuals(self, data):
        # residuals: distance of each point from the rect
        if self.slope != np.inf:
            return np.ones(shape=(len(data)))*(np.abs(data[:, 1] - (self.slope*data[:, 0]+self.intercept)) / (np.sqrt(1 + self.slope**2)))
        elif self.slope == 0:
            return np.ones(shape=(len(data)))*(data[:, 1]-self.intercept)
        else:
            return np.ones(shape=(len(data)))*(data[:, 0]-self.intercept)

    def get_inliers(self, data, in_th=1, v=None):
        residuals = self.get_residuals(data)
        return data[residuals**2 < in_th**2]


def estimate_circle(data):
    x, y = data[:, 0], data[:, 1]

    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc = c
        df2b_dc = np.empty((len(c), x.size))

        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

        return df2b_dc

    center, ier = optimize.leastsq(f_2b, (0, 0), Dfun=Df_2b, col_deriv=True)

    return center, calc_R(*center).mean()    # center (x, y), radius r


class CircleEstimator(BaseEstimator):

    def __init__(self):
        super(CircleEstimator, self).__init__(class_type=CIRCLE)
        self.center = (np.inf, np.inf)
        self.radius = np.inf

    # Fonte: https://math.stackexchange.com/questions/213658/get-the-equation-of-a-circle-when-given-3-points
    # epochs is for consistency
    def fit(self, data, epochs=0):
        assert len(data) == 3
        self.center, self.radius = estimate_circle(data)
        return self

    def get_residuals(self, data):
        # residuals: distance of each point from the rect
        return np.array([
            np.abs(euclidean_distance(p, self.center) - self.radius)
            for p in data
        ])

    def get_inliers(self, data, in_th=1, v=None):
        # residuals: distance of each point from the rect
        residuals = self.get_residuals(data)
        return data[residuals**2 < in_th**2]
