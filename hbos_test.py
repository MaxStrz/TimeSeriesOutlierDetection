from HBOSagg import HBOSAgg
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import seaborn as sns

scaler = RobustScaler()

arr_pareto = np.random.pareto(3, (1000, 3))

my_hbos_agg = HBOSAgg(instances=5, dynamic_bins=True)
my_hbos_agg.fit(arr_pareto)

scores = my_hbos_agg.decision_scores_
print(scores)