import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from models.base import PredEstBase

class MLS(PredEstBase):
    """
    Detects level shifts in univariate data using the median of each window.

    Parameters
    ----------
    window_size : int
        Size of the sliding window
    
    Attributes
    ----------
    data_with_windows : np.ndarray
        Data array with windows at beginning and end removed and replaced with
        np.nan
    
    windows : np.ndarray
        2D array of windows of size window_size
    
    meds : np.ndarray
        Median values of each window
    
    mad : float
        Median Absolute Deviation of the data
    
    predictions : np.ndarray
        Predictions are the median of the immediate previous window
    
    estimations : np.ndarray
        Estimations are the median of the immediate next window
    
    pred_error : np.ndarray
        Relative differences between data and predictions
    
    est_error : np.ndarray
        Relative differences between data and estimations
    
    decision_scores_ : np.ndarray
        Relative differences between predictions and estimations
    """

    def __init__(self, window_size: int=50):
        super().__init__(window_size)
    
    def fit(self, data: np.ndarray):
        """
        Fit the detector to the data providing level-shift outlier scores
        for each data point.
        
        A high score indicates a large diff
        """

        self.data_with_windows = super().remove_end_windows(data)
        self.windows = super().window_maker(data)
        # Array of length data - window_size + 1 of median values of each window
        self.meds = np.median(self.windows, axis=1)
        # Measure of spread used is Median Absolute Deviation which is more robust 
        # against effects of outliers than standard deviation
        # Data type is float64
        # Interquartile Range would also be an option 
        self.mad = median_abs_deviation(data, axis=0)
        self.predictions = super()._fit_predict(self.meds)
        self.estimations = super()._fit_estimate(self.meds)
        self._calculate_outlier_scores()

        return self
     
    def _calculate_outlier_scores(self):
            
        # Relative differences between data and predictions
        self.pred_error = np.absolute(
            self.data_with_windows - self.predictions
            )/self.mad

        # Relative differences between data and estimations
        self.est_error = np.absolute(
            self.data_with_windows - self.estimations
            )/self.mad

        # Relative differences between predications and estimations
        self.decision_scores_ = np.absolute(
            self.predictions - self.estimations
            )/self.mad
        
        # Score for entire univariate timeseries
        self.score = np.nanmax(self.decision_scores_)
        
def est_pred_agg(my_groupby, columns, model):
    """
    Aggregates the univariate outlier scores for all groups of a dataframe.

    Parameters
    ----------
    my_groupby : pandas groupby object
        Groupby object of a dataframe

    columns : list
        List of columns to apply the model to

    model : object
        Unfitted model that has this api: model().fit(array).score

    Returns
    -------
    pandas DataFrame
        DataFrame of outlier scores for each group

    """

    df_scores = pd.DataFrame({
        col: my_groupby.apply(lambda x: model.fit(x[col]).score)
        for col in columns
        })
    
    return df_scores
