from files.utils.constants import NEW_INLIERS, OLD_INLIERS, np
from files.utils.utility_functions import euclidean_distance, mad


class MSSModel:

    def __init__(self, mss):
        self.mss = mss

    def get_residuals(self, data):
        return np.array([self.compute_inlier_factor(point=p) for p in data])

    def get_nearest_points(self, data=None):
        data = self.data if data is None else data
        result = np.array([
            self.mss[
                # compute distances between weights and point
                np.argmin([euclidean_distance(p, point) for point in self.mss])
            ]
            for p in data
        ])
        return result

    def compute_inlier_factor(self, point, n=3):
        residuals = np.array([euclidean_distance(point, m) for m in self.mss])
        # the n points of the mss nearest to the point
        min_ms = []
        for i in range(n):
            min_idx = np.argmin(residuals)
            min_m = residuals[min_idx]
            min_ms.append(min_m)
            residuals[min_idx] = np.inf
        return np.mean([euclidean_distance(point, m) for m in min_ms])

    def get_inliers(self, sampling_data, in_th, indexes=False, v=NEW_INLIERS):
        nearest_points = self.get_nearest_points(data=sampling_data)
        if v == OLD_INLIERS:
            return sampling_data[np.where(nearest_points < np.median(nearest_points)*in_th)]
        else:
            inlier_factors = np.array(
                [self.compute_inlier_factor(p) for p in sampling_data])
            if len(inlier_factors) > 0:
                residuals_mad = (np.median(inlier_factors) +
                                 mad(inlier_factors))*in_th
                idxs = np.where(inlier_factors < residuals_mad)
                res = nearest_points[idxs]
                if len(res) > 0:
                    idxs = np.where(nearest_points < np.median(res))
                    if indexes:
                        return sampling_data[idxs], idxs
                    else:
                        return sampling_data[idxs]
            if indexes:
                return np.array([[np.inf, np.inf]]), np.array([np.inf])
            else:
                return np.array([[np.inf, np.inf]])

    # This method is only for consistency
    def fit(self, data=None, epochs=None):
        pass
