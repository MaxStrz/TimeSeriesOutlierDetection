from models.MedianLevelShift import MLS
from test.test_data import my_data
import numpy as np

data_instance = my_data()

my_2sm = MLS() # Create a MedianLevelShift instance. Window size defaults to 50
my_2sm.fit(data_instance.one_dim_pareto) # Fit the model to the data
print("MedianLevelShift tested successfully")