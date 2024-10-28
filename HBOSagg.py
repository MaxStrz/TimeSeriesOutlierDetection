import numpy as np
import pandas as pd
from pyod.models.hbos import HBOS, _calculate_outlier_scores, check_parameter, invert_order, BaseDetector
from sklearn.utils.validation import check_array, check_is_fitted
from typing import Type, Generator, Any
from tqdm.notebook import tqdm

class HBOSAgg:
    """
    Histogram-Based Outlier Score Aggregator (HBOSAgg) takes the mean of the 
    outlier scores of a datapoint from multiple HBOS instances using different
    bin counts. 

    HBOSAgg implements an adapted version of pyod's HBOS class, enabling the
    use of dynamic bins. See HBOS_dynamic_bins for more information.

    Parameters
    ----------
    instances : int, optional (default=100)
        The number of HBOS instances to generate.
    
    dynamic_bins : bool, optional (default=True)
        If True, the dynamic bin method is used.
    
    Attributes
    ----------
    _arr_bin_counts : numpy.ndarray of shape (instances,)
        Array of length instances with random bin counts between 2 and 
        sqrt(len(X)).
    
    results : dict
        Dictionary containing the results of the HBOS instances. Keys are 
        'hbos_instance', 'bins', 'threshold', and 'decision_scores'.
    
    decision_scores_ : numpy.ndarray of shape (n_samples,)
        The mean score for each sample.
    
    arr_rank : numpy.ndarray of shape (n_samples,)
        The rank of the mean score for each sample.

    """
    def __init__(self,
                 instances: int=100,
                 dynamic_bins: bool=True
                 ) -> None:
        self.instances = instances
        self.dynamic_bins = dynamic_bins

    def fit(self, X: np.ndarray):
        """
        Fit detector.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        self : object
            Fitted detector.

        """
        # Generates array of length instances with random bin counts 
        # between 2 and sqrt(len(X))
        self._arr_bin_counts = np.random.randint(3, int(np.sqrt(len(X))), 
                                                 self.instances)

        self.results = {
            # Key for the HBOS-Instance
            'hbos_instance': np.arange(0, self.instances, 1),
            # Integar zwischen 2 und sqrt(len(X))
            'bins':self._arr_bin_counts,
            'threshold':np.full(self.instances, 2.2222, dtype=np.float32),
            'decision_scores':np.full((self.instances, len(X)), 2.2222, 
                                      dtype=np.float32),
                                      }
        
        # Fit the HBOS instances and store the results
        for instance in range(self.instances):
            if self.dynamic_bins:
                hbos_model = HBOS_dynamic_bins(
                    dynamic_bins=True,
                    n_bins=self._arr_bin_counts[instance]
                    )
                hbos_model.fit(X)
            else:	
                hbos_model = HBOS(n_bins=self._arr_bin_counts[instance])
                hbos_model.fit(X)
            self.results['threshold'][instance] = hbos_model.threshold_
            self.results['decision_scores'][instance] = hbos_model.decision_scores_

        # Calculate the mean score for each sample
        # Array von (100, 500) auf (500, 100) ändern
        decision_scores_per_sample = np.transpose(
            self.results['decision_scores']) 
        self.decision_scores_ = np.mean(decision_scores_per_sample, axis=1)

        # Calculate the rank of the mean score
        # E.g. For smallest mean score, 
        # returns the index of that score in arr_mean_score.
        # [3, 5, 2] -> [2, 0, 1]
        position_of_rank_in_mean_score = self.decision_scores_.argsort()
        # Where to find the first number in arr_mean_score, 
        # who's position in position_of_rank_in_mean_score is its rank
        # E.g. [2, 0, 1] -> [1, 2, 0]
        self.arr_rank = position_of_rank_in_mean_score.argsort()
    
        return self

class HBOS_dynamic_bins(HBOS):
    """
    Histogram-Based Outlier Score (HBOS) with dynamic bins.
    
    All bins in a histogram with dynamic bin contain the
    same number of samples, and therefore have different bin widths. The area 
    of each bin is the same i.e. bin width * bin height remains constant. The 
    height of the bin is therefore the measure of the density of the space
    within which a data point finds itself.

    Parameters
    ----------
    dynamic_bins : bool, optional (default=True)
        If True, the dynamic bin method is used.
    
    **kwargs : optional (default=None)
        Other keyword arguments accepted by HBOS.
    
    Attributes
    ----------

    """
    def __init__(self, dynamic_bins=True, **kwargs):
        self.dynamic_bins=dynamic_bins
        super().__init__(**kwargs)

    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y) 

        _, n_features = X.shape[0], X.shape[1]

        if self.dynamic_bins:
            # Check the number of bins
            check_parameter(self.n_bins, low=2, high=np.inf)
            self.hist_ = np.zeros([self.n_bins, n_features])
            self.bin_edges_ = np.zeros([self.n_bins + 1, n_features])

            # build the histograms for all dimensions
            for i in range(n_features):

                # Bin belonging for each sample 
                # Output: pandas.core.arrays.categorical.Categorical
                # [(0.561, 0.69], (0.282, 0.468], (0.561, 0.69], (excluded lower 
                # bound, included upper bound], ..., (0.561, 0.69]]
                # Length: len(X)
                # Categories (self.n_bins, interval[float64, right]): 
                # [(0.0419, 0.282] < (0.282, 0.468] < (0.468, 0.561] < 
                # (0.561, 0.69] < (0.69, 0.964]]

                # Returns array of all values in dimension i
                x_i = X[:, i]

                # Generate tiny noise
                noise = np.random.normal(0, 0.001, size=x_i.size)

                # Add noise to the data to avoid identical values
                x_i_noise = x_i + noise

                # pd.qcut: Quantile-based discretization function
                # q is the number of quantiles i.e. bins
                # x is the data to discretize
                bin_belonging = pd.qcut(x=x_i_noise, q=self.n_bins, precision=6)

                # Count the number of samples in each bin
                # pandas deals with identical values by assigning them to the 
                # same bin but we added noise to avoid identical values
                # Output: pandas.core.series.Series
                # (0.0419, 0.282]    10
                # (0.282, 0.468]     10
                # (0.468, 0.561]     10
                # (0.561, 0.69]      10
                # (0.69, 0.964]      10
                # Name: count, dtype: int64
                dynamic_counts = bin_belonging.value_counts()

                # Get the bin intervals
                # Output: pandas.core.indexes.interval.IntervalIndex
                # IntervalIndex([(0.0419, 0.282], (0.282, 0.468], (0.468, 0.561]
                # , (0.561, 0.69], (0.69, 0.964]],
                #               dtype='interval[float64, right]')
                dynamic_bin_intervals = bin_belonging.categories

                # Get the bin edges using list comprehension
                # Output: list of bin edges
                # [0.0419, 0.282, 0.468, 0.561, 0.69, 0.964]
                dynamic_bin_edges = [
                    interval.left # left edge of the interval
                    for interval # for each interval in the IntervalIndex
                    # final bin edge is the right edge of the last interval
                    in dynamic_bin_intervals]+[dynamic_bin_intervals[-1].right]

                # density=True: the sum of the area of all bins is 1 i.e.
                # result of np.histogram is the value of the pdf at each bin

                # hist_[:, i] assigns the returned array of pdf values for each
                # observation for dimension i
                # bin_edges_[:, i] assigns the bin edges for dimension i which
                # we already have and we use as input for np.histogram

                self.hist_[:, i], self.bin_edges_[:, i] = np.histogram(
                    x_i_noise, # data of dimension i with added noise
                    bins=dynamic_bin_edges, # bin edges already calculated
                    density=True) # return values of pdf

                # the sum of (width * height) should equal to 1
                # np.diff(self.bin_edges_[:, i]) calculates the width bins
                assert (np.isclose(1, np.sum(
                    self.hist_[:, i] * np.diff(self.bin_edges_[:, i])
                    ),
                    atol=0.1))

            outlier_scores = _calculate_outlier_scores(X, 
                                                       self.bin_edges_,
                                                       self.hist_,
                                                       self.n_bins,
                                                       self.alpha, 
                                                       self.tol)
        
        else:
            print("Using original fit method.")
            super().fit(X, y=None)

        # invert decision_scores_. Outliers comes with higher outlier scores
        self.decision_scores_ = invert_order(np.sum(outlier_scores, axis=1))
        self._process_decision_scores()
        return self