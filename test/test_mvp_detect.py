from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from models.MVPDetect import MVPDetect, ParamLearnerLinReg
from test.test_data import my_data

def test_fit_predict():
    # Data Generation
    data = my_data()
    partitions = data.dict_partitions
    train_test_pairs = data.train_test_idx

    # Model Initialization
    detector = MVPDetect(
        scorer=mean_squared_error,
        model=LinearRegression(),
        param_learner=ParamLearnerLinReg()
    )

    # Initialise Model
    detector.fit_predict(
        dict_partitions=partitions,
        train_test_pairs=train_test_pairs,
        iteration_limit=None
    )

    # Check if collected_scores_ exists and is populated
    assert hasattr(detector, 'collected_scores_'), "collected_scores_ not found."
    len_coll_scores = len(detector.collected_scores_)
    assert len(detector.collected_scores_) > 0, "No scores collected."
    assert len_coll_scores == len(partitions), "Incorrect no. of collected scores."

    # Check of aggregated_scores_ exists and is populated
    assert hasattr(detector, 'aggregated_scores_'), "aggregated_scores_ not found."
    len_agg_scores = len(detector.aggregated_scores_)
    assert len_agg_scores > 0, "No aggregated scores."
    assert len_agg_scores == len(partitions), "Incorrect no. of agg scores."

    # Length of param_learn dictionary of linear coefficients
    # Should be number of partitions * number of variables
    coeffs = len(detector.param_learner.dict_of_all_coefficients)
    variables = len(list(partitions.values())[0].columns)
    assert coeffs == variables * len(partitions), "Incorrect number of coeffs."

    

