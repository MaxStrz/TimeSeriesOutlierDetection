from collections import defaultdict
from models.MVPDetect import MVPDetect
from test.test_data import my_data
import numpy as np
import pandas as pd

# # Generate 4-dimensionale data from a Pareto distribution with shape/slope 3
# # and 1000 samples
# arr_pareto = np.random.pareto(3, (1000, 4))

# # Create dataframe from array
# df1 = pd.DataFrame(arr_pareto, columns=['V1', 'V2', 'V3', 'V4'])
# df1.attrs = {'name':'df1'}

# # Create partitions of the data
# def create_partitions(df, partition_size):
#     # Remainder of division of dataframe length by partition_size
#     remainder = len(df) % partition_size

#     if remainder == 0:
#         df_no_remainder = df
#     else:
#         # Remove the remainder to make dataframe divisible by partition_size
#         df_no_remainder = df.iloc[:-remainder]

#     no_of_partitions = len(df_no_remainder) / partition_size
#     # Split the dataframe into partitions
#     arr_partitions = np.array_split(df_no_remainder.values, no_of_partitions)

#     dict_partitions = defaultdict()

#     for i, arr_partition in enumerate(arr_partitions):
#         df_part = pd.DataFrame(arr_partition, columns=df.columns)
#         dict_partitions[f"{df.attrs['name']}_{i}"] = df_part
    
#     return dict_partitions

# dict_partitions = create_partitions(df1, 50)

# def create_train_test_idx(idx):
#     # Return all possible combinations of train and test partitions within the
#     # same dataframe and without using same partition for both train and test
#     train_test_idx = [(tr, te) for tr in idx for te in idx if tr != te]
#     return train_test_idx

data_instance = my_data()

#train_test_idx = create_train_test_idx(list(dict_partitions.keys()))

my_mvp = MVPDetect() # Create a MedianLevelShift instance. Window size defaults to 50
my_mvp.fit_predict(data_instance.dict_partitions, 
                   data_instance.train_test_idx) # Fit the model to the data

print("MVPDetect tested successfully")