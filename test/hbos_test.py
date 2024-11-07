from models.HBOSagg import HBOSAgg
import numpy as np

# Generate univariate data from a Pareto distribution with shape/slope 3
# and 1000 samples
arr_pareto = np.random.pareto(3, (1000, 3))

# Create an HBOSAgg instance with 5 instances of PyODs HBOS each with different
# bin counts. See HBOSAgg-Class for an explanation of dynamic_bins.
my_hbos_agg = HBOSAgg(instances=5, dynamic_bins=True)
my_hbos_agg.fit(arr_pareto)

scores = my_hbos_agg.decision_scores_
print("HBOSAgg tested successfully")