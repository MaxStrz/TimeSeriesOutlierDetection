from collections import defaultdict
from models.MVPDetect import MVPDetect
from test.test_data import my_data
import numpy as np
import pandas as pd

data_instance = my_data()

#train_test_idx = create_train_test_idx(list(dict_partitions.keys()))

my_mvp = MVPDetect() # Create a MedianLevelShift instance. Window size defaults to 50
my_mvp.fit_predict(data_instance.dict_partitions, 
                   data_instance.train_test_idx) # Fit the model to the data

print("MVPDetect tested successfully")