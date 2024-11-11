from models.HBOSagg import HBOSAgg
from test.test_data import my_data
import numpy as np

data_instance = my_data()

# Create an HBOSAgg instance with 5 instances of PyODs HBOS each with different
# bin counts. See HBOSAgg-Class for an explanation of dynamic_bins.
my_hbos_agg = HBOSAgg(instances=5, dynamic_bins=True)
my_hbos_agg.fit(data_instance.four_dim_pareto)

scores = my_hbos_agg.decision_scores_
print("HBOSAgg tested successfully")