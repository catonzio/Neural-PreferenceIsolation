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

# Fonte: https://www.inf.ed.ac.uk/teaching/courses/cg/lectures/implicit.pdf


class LineEstimator(BaseEstimator):

    def __init__(self):
        super(LineEstimator, self).__init__(class_type=LINE)
        self.a, self.b, self.c = np.inf, np.inf, np.inf

    # epochs is for consistency
    def fit(self, data, epochs=0):
        assert len(data) == 2
        (x1, y1), (x2, y2) = data
        self.a = y2 - y1
        self.b = x1 - x2
        self.c = x2*y1 - y2*x1
        return self

    def get_residuals(self, data):
        return (self.a*data[:, 0] + self.b*data[:, 1] + self.c) / np.sqrt(self.a**2 + self.b**2)

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


class PlaneEstimator(BaseEstimator):

    def __init__(self):
        super(PlaneEstimator, self).__init__(class_type=PLANE)
        self.a, self.b, self.c, self.d = self.alpha = np.array([np.inf, np.inf, np.inf, np.inf])

    # epochs is for consistency
    def fit(self, data, epochs=0):
        assert len(data) == 3
        p1, p2, p3 = data
        n1 = p2 - p1
        n2 = p3 - p1
        a, b, c = cross_ = np.cross(n1, n2)
        d = -np.dot(cross_, p1)
        # p1, p2, p3 = data
        # x1, x2 = p1
        # y1, y2 = p2
        # z1, z2 = p3
        # a1 = x2 - x1
        # b1 = y2 - y1
        # c1 = z2 - z1
        # a2 = x3 - x1
        # b2 = y3 - y1
        # c2 = z3 - z1
        # a = b1 * c2 - b2 * c1
        # b = a2 * c1 - a1 * c2
        # c = a1 * b2 - b1 * a2
        # d = (- a * x1 - b * y1 - c * z1)
        self.a, self.b, self.c, self.d = self.alpha = np.array([a, b, c, d])
        return self.alpha

    def get_residuals(self, data):
        return (np.abs(np.dot(data, self.alpha[:-1])) + self.alpha[-1]) / np.sqrt(np.sum(self.alpha[:-1]**2))

    def get_inliers(self, data, in_th=1, v=None):
        residuals = self.get_residuals(data)
        return data[residuals**2 < in_th**2]
