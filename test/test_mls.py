from models.MedianLevelShift import MLS
import numpy as np

def test_mls(my_data_fixture):
    data = my_data_fixture.one_dim_pareto

    my_2sm = MLS() # Create a MedianLevelShift instance. Window size defaults to 50
    my_2sm.fit(data) # Fit the model to the data
    assert hasattr(my_2sm, "data_with_windows"), "data_with_windows not found."
    assert hasattr(my_2sm, "windows"), "windows not found."
    
    ws = my_2sm.window_size
    warr = my_2sm.windows
    assert len(warr) == len(data) - ws + 1, "Incorrect number of windows."
    
    no_end_windows = my_2sm.data_with_windows
    assert len(no_end_windows) == len(data), "Incorrect data length."

    # The length of the first window should be the same as window size
    assert len(my_2sm.windows[0]) == ws, "Incorrect window size."

    # The length of the prediction array should be same as data
    assert len(my_2sm.predictions) == len(data), "Incorrect prediction length."

    # The length of the estimation array should be same as data
    assert len(my_2sm.estimations) == len(data), "Incorrect estimation length."

    # The length of the prediction error array should be same as data
    assert len(my_2sm.pred_error) == len(data), "Incorrect pred error length."

    # The length of the estimation error array should be same as data
    assert len(my_2sm.est_error) == len(data), "Incorrect est error length."

    # The length of the decision scores array should be same as data
    assert len(my_2sm.decision_scores_) == len(data), "Incorrect d scores len."
    



