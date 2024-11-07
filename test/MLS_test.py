from models.MedianLevelShift import MLS
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import seaborn as sns

scaler = RobustScaler()

# Generate univariate data from a Pareto distribution with shape/slope 3
# and 1000 samples
arr_pareto = np.random.pareto(3, (1000,))

my_2sm = MLS() # Create a MedianLevelShift instance. Window size defaults to 50
my_2sm.fit(arr_pareto) # Fit the model to the data
print("MedianLevelShift tested successfully")