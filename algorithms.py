import copy
from collections import defaultdict
import itertools
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view # for Univariate_2sm to create sliding windows
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.preprocessing import RobustScaler
from statsmodels.tsa.ar_model import AutoReg
from typing import Type, Generator, Any
from tqdm.notebook import tqdm

class UniAR(Univariate2SidedEstimator): # Detector
    def __init__(self,
                 window_size: int=50
                ) -> None:
        self.window_size = window_size
    
    def fit(self, X: np.ndarray):
        super().__init__(X, self.window_size)
        self._fit_predict()
        self._fit_estimate()
        self._calculate_outlier_scores()
    #    return self

    def _fit_predict(self):
        fitted_model = AutoReg(self.data, lags=self.window_size, trend='c').fit()
        self.predictions = np.concatenate((np.empty(self.window_size), fitted_model.fittedvalues))

    def _fit_estimate(self):
        fitted_flpd_model = AutoReg(np.flip(self.data), lags=self.window_size, trend='c').fit()
        self.estimations = np.concatenate((np.flip(fitted_flpd_model.fittedvalues), np.empty(self.window_size)))
    
    #def predict(self, X):
    #    return self.predictions

    def _calculate_outlier_scores(self):
        self.rel_abs_diff_data_pred = np.absolute(self.data - self.predictions)/self.mad
        self.rel_abs_diff_pred_est = self.point_outlier_scores = self.decision_scores_ = np.absolute(self.predictions - self.estimations)/self.mad
        self.max_pred_est_error = self.max_outlier_score = np.nanmax(self.rel_abs_diff_pred_est)

class AREstimator(BaseEstimator): # Estimator for GridSearchCV

    """
    How to build an estimator: https://scikit-learn.org/stable/developers/develop.html
    """
    def __init__(self, 
                 window_size: int=50, 
                 hold_back: int=50,
                 trend = 'c') -> None: # All keyword arguments of scikit-learn estimators have default values
        self.window_size = window_size
        self.hold_back = hold_back
        self.trend = trend
    
    # scikit-learn estimators accept data as argument in the fit method
    def fit(self, X: np.ndarray, y=None):
        self.X_ = X
        # Attributes that have been estimated from the data must always have a name ending with trailing underscore
        self.fitted_model_ = AutoReg(X, lags=self.window_size, 
                                    trend=self.trend, 
                                    hold_back=self.hold_back).fit()
        return self
    
    def predict(self, X: np.ndarray): # X is an array of observations immediately prior to the next observation
        check_is_fitted(self) # Check if the estimator has been fitted
        # Assert that prediction index X equals length of endog array, an integar representing the next observation number
        # E.g. if len(training_array) = 5, last observation index is 4, so prediction index X = 5 = len(training_array)
        start = len(self.fitted_model_.model.endog)
        return self.fitted_model_.predict(start=start, end=start, dynamic=False)
    
    def predict_1(self, X: int): # X is the index of the next observation
        check_is_fitted(self) # Check if the estimator has been fitted
        # Assert that prediction index X equals length of endog array, an integar representing the next observation number
        # E.g. if len(training_array) = 5, last observation index is 4, so prediction index X = 5 = len(training_array)
        assert X == len(self.fitted_model_.model.endog)
        return self.fitted_model_.predict(start=X, end=X, dynamic=False)
    
    def score(self, X: np.ndarray, y: np.ndarray):
        check_is_fitted(self)
        prediction = self.predict(X)
        score = -mean_squared_error(y, prediction)
        #score = mean_squared_error(X[self.hold_back:], self.fitted_model_.fittedvalues)
        return score

    def get_params(self, deep=True):
        return {"window_size": self.window_size,
                'hold_back': self.hold_back,
                'trend': self.trend
                }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# class AggUni2:

#     def __init__(self, 
#                  model: Any, 
#                  gen: Generator[pd.DataFrame, None, None]) -> None:
#         self.gen = gen # Generator to iterate over all coil element dataframes
#         self.model = model # Model class instance with predefined parameters
        
#         # Dictionary to store outlier scores for each KPI 
#         # (many per coil element)
#         self.dict_kpi_decision_scores = defaultdict(dict)
        
#         # Dictionary to store outlier scores for each coil element
#         self.dict_ce_outlier_score = defaultdict(dict)

#     def get_outlier_scores(self):

#         for df in self.gen:
#             for kpi in df:
#                 fitted_model = self.model.fit(df[kpi].values)
#                 # Store decision scores for each KPI for every prediction vs 
#                 # estimation pair in each time series
#                 self.dict_kpi_decision_scores[
#                     (df.serial, df.coil_serial, df.coil_element)
#                     ][kpi] = fitted_model.decision_scores_
#                 # Store the maximum relative difference between predictions 
#                 # and estimations for each coil element
#                 self.dict_ce_outlier_score[(df.serial, 
#                                             df.coil_serial, 
#                                             df.coil_element
#                                             )][kpi] = fitted_model.max_pred_est_error
        
#         df = pd.DataFrame.from_dict(self.dict_ce_outlier_score, orient='index')

#         df.rename(
#             columns={kpi}: f'{
#                 self.model.__class__.__name__
#                 }_{kpi}' for kpi in df.columns}, inplace=True)
#         df.index.names = ['serial', 'coil_serial', 'coil_element']
        
#         return df

# class AggMulti:

#     def __init__(self, model_instance: BaseDetector, model_params: dict=None) -> None:
#         self.paths = Config() # Object with paths to data
#         self.get_data_instance = GetData() # Object to get data
#         #self.model_params = model_params
#         self.dict_ce_outlier_score = copy.deepcopy(self.get_data_instance.structure_dict) # Dictionary to store outlier scores for each coil element
#         self.point_decision_scores_ = copy.deepcopy(self.get_data_instance.structure_dict) # Dictionary to store outlier scores for each KPI (many per coil element)
#         self.my_gen_instance = dict_key_value_generator(self.get_data_instance.structure_dict) # Generator to iterate over all coil element kpis
#         self.model_instance = model_instance # Model class object

#     def get_outlier_scores(self):

#         for serial, coil_serial, coil_element in self.my_gen_instance:
#             # Return array of scaled kpis for coil element
#             df_ce_kpis_scaled = pd.read_pickle(self.paths.cleaned_scaled_pkl_path.format(serial, coil_serial, coil_element))

#             self.model_instance.fit(df_ce_kpis_scaled.values)
#             self.point_decision_scores_[serial][coil_serial][coil_element] = self.model_instance.decision_scores_
            
#             # Outlier Score for entire data array is the maximum relative difference between predictions and estimations
#             max_outlier_score = np.nanmax(self.model_instance.decision_scores_)
#             self.dict_ce_outlier_score[serial][coil_serial][coil_element]['max_outlier_score'] = max_outlier_score

#         self.make_max_df()

#     def make_max_df(self):
        
#         self.temp_dict = {(i, j, k): self.dict_ce_outlier_score[i][j][k]
#                           for i in self.dict_ce_outlier_score.keys()
#                           for j in self.dict_ce_outlier_score[i].keys()
#                           for k in self.dict_ce_outlier_score[i][j].keys()}

#         self.df_outlier_scores = pd.DataFrame.from_dict(self.temp_dict, orient='index')

class DummyParamLearner:
    def __init__(self):
        None

    def learn_params(self, X_train, y_name, fitted_model, trainin):
        None

    def create_kpi_param_dicts(self):
        None

class ParamLearnerMLPReg:
    def __init__(self):
        self.dict_of_all_coefficients = defaultdict(dict)

    def learn_params(self, X_train, y_name, fitted_model, trainin):
        s, cs, ce, p = trainin
        dict_key = (s, cs, ce, p, y_name)
        self.dict_of_all_coefficients[dict_key]['coeffs'] = fitted_model.coefs_
        self.dict_of_all_coefficients[dict_key]['intercept'] = fitted_model.intercepts_

    def create_kpi_param_dicts(self):
        None

class ParamLearnerLinReg:
    def __init__(self):
        self.dict_of_all_coefficients = defaultdict(dict)

    def learn_params(self, X_train, y_name, fitted_model, trainin):
        s, cs, ce, p = trainin
        dict_key = (s, cs, ce, p, y_name)
        for variable in X_train:
            self.dict_of_all_coefficients[dict_key][variable] = fitted_model.coef_[X_train.columns.get_loc(variable)]
        self.dict_of_all_coefficients[dict_key]['intercept'] = fitted_model.intercept_

    def create_kpi_param_dicts(self):
        new_dict = lambda dict, key2del: {k[:-1]: v for k, v in dict.items() if k[-1] == key2del}
        dict_to_df = lambda dict, key2del: pd.DataFrame.from_dict(new_dict(dict, key2del), orient='index')
        self.df_ssr_reg = dict_to_df(self.dict_of_all_coefficients, 'SSR')
        self.df_csi_reg = dict_to_df(self.dict_of_all_coefficients, 'CSI')
        self.df_csp_reg = dict_to_df(self.dict_of_all_coefficients, 'CSP')
        self.df_cnl_reg = dict_to_df(self.dict_of_all_coefficients, 'CNL')

class MVPDetect:
    """Multivariate Dependency Base Detector

    MVPDetect trains four regression models on a time series subsequence, one
    model to predict each KPI based on the other three KPIs.

    Prediction errors for each model are calculated over all other subsequences 
    in each time series. 

    Model parameters are passed to a ParamLearner class, which is specific 
    to each model type. The DummyParamLearner class is used when no parameter
    learning is required, although this is not recommended; training takes time
    and parameters are useful.

    Parameters
    ----------
    scorer : callable, default=mean_squared_error
        The scoring function used to evaluate the quality of the model's 
        predictions. 
    
    model : object, default=LinearRegression()
        The regression model to be used. Needs to have a fit() and predict()
        method and needs to return itselt when fit() is called. All scikit-learn
        regression models should work.

    param_learner : object, default=ParamLearnerLinReg()
        This class learns the parameters of the regression model. Needs to have
        the same interface as DummyParamLearner() and ParamLearnerLinReg().
    
    Attributes
    ----------
    collected_scores_ : defaultdict
        A dictionary of dictionaries of lists to collect the scores for each 
        partition-partition pair predictions. 

    aggregated_scores_ : dict
        A dictionary that stores the median of the scores for each partition.
        TODO: Remove. Class should judt collect scores and decide on aggregation
        afterwards.

    """
    def __init__(self,
                 scorer=mean_squared_error,
                 model=LinearRegression(),
                 param_learner=ParamLearnerLinReg()):
        self.scorer = scorer
        self.model = model
        self.param_learner = param_learner
        self.collected_scores_ = defaultdict(lambda: defaultdict(list))
    
    def fit_predict(self, 
                    dict_partitions, 
                    train_test_pairs, 
                    iteration_limit=None):
        """Fit and predict models on each partition
        
        Parameters
        ----------
        dict_partitions : dict
            A dictionary of dataframes, each dataframe representing a partition.
            Keys are tuples of (serial, coil_serial, coil_element, partition).
        
        train_test_pairs : list
            A list of nested tuples with structure ((trainin), (testin)).
            trainin is the index for the training partition and testin is the
            index for the testing partition.

        iteration_limit : int, default=None
            The number of iterations to run. Useful for very short test runs to 
            check that the code is working. If None, all partitions are used.

        """
        # tqdm is used to show a progress bar.
        # iterate over all partitions
        for trainin, train in tqdm(itertools.islice(dict_partitions.items(), 
                                                    0, 
                                                    iteration_limit)):
            for kpi in train:
                # Scikit-learn MLP has a neat warm_start option to use the
                # previous neural network weights as a starting point for the 
                # next training. But this requires useing the same training KPIs
                # so the process must be repeated for each KPI.

                # USE ME 4 MLP #if kpi == 'CNL' or kpi == 'SSR' or kpi == 'CSP': 
                #    continue

                # scale training data
                my_scalar = RobustScaler().fit(train)
                scaled_train = pd.DataFrame(my_scalar.transform(train), 
                                            columns=train.columns)
                
                # split training data into X and y
                X_train = scaled_train.drop(kpi, axis=1)
                y_train = scaled_train[kpi]

                # fit model
                fitted_model = self._fit(X_train, y_train)
                
                # learn parameters using some ParamLearner class
                self.param_learner.learn_params(X_train,
                                                kpi, 
                                                fitted_model, 
                                                trainin)
                
                # loop through all train test pairs
                # train test pairs always belong to the same coil element
                # no parition is paired with itself
                for index_train, testin in train_test_pairs:
                    # ignores pairs where the training index does not match 
                    # the current index set by trainin
                    # there is likely a more efficient way to do this
                    if trainin != index_train:
                        continue
                    # fetch the test dataframe 
                    test = dict_partitions[testin]
                    # scale the test data using the training scalar
                    scaled_test = pd.DataFrame(my_scalar.transform(test), 
                                               columns=test.columns)
                    # split the test data into X and y
                    X_test = scaled_test.drop(kpi, axis=1)
                    y_test = scaled_test[kpi]
                    y_prediction = self._predict(fitted_model, X_test)
                    score = self._score(y_prediction, y_test)
                    self._score_collector(trainin, kpi, score)

        # convert collected scores to a dictionary
        # defaultdicts didn't work            
        self.collected_scores_ = dict(self.collected_scores_)

        # aggregate the scores. This should be removed and done separately
        self._score_aggregator()

        # create the outlier scores dataframe using the aggregated scores
        # to set a coil-element level outlier score
        partition_size = len(list(dict_partitions.values())[0])
        self._create_ce_outlier_scores(partition_size)

        # aggregates model-specific parameters
        # this should also be done separately
        self.param_learner.create_kpi_param_dicts()

    def _fit(self, X_train, y_train):
        fitted_model = self.model.fit(X_train, y_train)
        return fitted_model
        
    def _predict(self, fitted_model, X):
        prediction = fitted_model.predict(X)
        return prediction

    def _score(self, y_prediction, y):
        score = self.scorer(y_prediction, y)
        return score

    def _score_collector(self, trainin, kpi, score):
        self.collected_scores_[trainin][kpi].append(score)

    def _score_aggregator(self):
        self.aggregated_scores_ = {
            trainin:{f'{type(self.model).__name__}_{kpi}': np.median(scores) 
                     for kpi, scores in dict_scores.items()}
                     for trainin, dict_scores 
                     in self.collected_scores_.items()}
        
    def _create_ce_outlier_scores(self, partition_size):
        self.df_ce_outlier_scores = pd.DataFrame.from_dict(
            self.aggregated_scores_, 
            orient='index').groupby(level=[0, 1, 2]).max()
        
        for column in self.df_ce_outlier_scores:
            # rename column
            self.df_ce_outlier_scores.rename(
                columns={column: f'{column}_{partition_size}'}, 
                inplace=True)
            
        return self.df_ce_outlier_scores

class AR4CVal(): 
    def __init__(self, window_size: int=2, hold_back: int=5) -> None:
        self.window_size = window_size
        self.hold_back = hold_back
    
    def fit(self, X: np.ndarray):
        self.X_ = X 
        self.fitted_model = AutoReg(X, lags=self.window_size, trend='c', hold_back=self.hold_back).fit()
        return self
    
    def predict(self, X: np.ndarray):
        
        return self.fitted_model.fittedvalues
    
    def score(self, X: np.ndarray):
        prediction = self.fit(X).predict()
        score = mean_squared_error(X[self.hold_back:], prediction)
        return score

    def get_params(self, deep=True):
        return {"window_size": self.window_size,
                'hold_back': self.hold_back
                }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self