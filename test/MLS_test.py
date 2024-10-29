from models.MedianLevelShift import MLS
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import seaborn as sns

scaler = RobustScaler()

arr_pareto = np.random.pareto(3, (1000,))

my_2sm = MLS()
my_2sm.fit(arr_pareto)
print("MedianLevelShift tested successfully")