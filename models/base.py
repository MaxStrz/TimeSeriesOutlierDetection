import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import median_abs_deviation

class PredEstBase:
    """
    Base class for Prediction vs Estimation sliding window detectors.
    Transforms univariate data array into a 2D array of windows of size 
    window_size.
    Creates prediction and estimation arrays to be compared with the data.
    
    Parameters
    ----------
    window_size : int
        Size of the sliding window
    """

    def __init__(self, window_size: int=50):
        self.window_size = window_size

    def window_maker(self, data):
        """Returns 2D array of windows of size window_size"""

        return sliding_window_view(data, self.window_size)
    
    def remove_end_windows(self, data):
        """Returns array of len(data), window at beginning and end removed"""

        data_minus_windows = np.concatenate(
            (np.full(self.window_size, np.nan), 
            data[self.window_size:-self.window_size], 
            np.full(self.window_size, np.nan))
            )

        return data_minus_windows
    
    def _fit_predict(self, predictions):
        """
        Predictions for all data points except the last window_size 
        e.g 50 data points because we don't have estimates for them.
        We have an estimate for the window-size+1th last data point 
        (e.g. 51st last) so we want a prediction for it too.
        We remove window_size + 1 because this is the last data point to get 
        a prediction but it isn't used to make a prediction.
        """

        predictions_with_windows = np.concatenate(
            (np.full(self.window_size, np.nan),
            predictions[:-(self.window_size + 1)],
            np.full(self.window_size, np.nan))
            )
        
        return predictions_with_windows
    
    def _fit_estimate(self, estimations):
        """We don't have predictions for the first window_size 
        # e.g. 50 data points, so we remove them
        # We remove window_size + 1 because this is the first data point to get
        # an estimate but it isn't used to make an estimate.
        """

        estimations_with_windows = np.concatenate(
            (np.full(self.window_size, np.nan),
            estimations[self.window_size + 1:],
            np.full(self.window_size, np.nan))
            )
        
        return estimations_with_windows