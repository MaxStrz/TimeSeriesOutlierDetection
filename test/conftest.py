from collections import defaultdict
import numpy as np
import pandas as pd
import pytest

class my_data:
    def __init__(self):
        self.one_dim_pareto = self.create_1d_pareto()
        self.four_dim_pareto = self.create_4d_pareto(name='df1')
        self.dict_partitions = self.create_partitions(self.four_dim_pareto, 50)
        self.train_test_idx = self.create_train_test_idx(
            list(self.dict_partitions.keys())
            )

    def create_1d_pareto(self) -> np.ndarray:  
        # Generate univariate data from a Pareto distribution with shape/slope 3
        # and 1000 samples
        return np.random.pareto(3, (1000,))
    
    def create_4d_pareto(self, name) -> pd.DataFrame:
        # Generate 4-dimensionale data from a Pareto distribution with shape/slope 3
        # and 1000 samples
        arr_pareto = np.random.pareto(3, (1000, 4))

        # Create dataframe from array
        df1 = pd.DataFrame(arr_pareto, columns=['V1', 'V2', 'V3', 'V4'])
        df1.attrs = {'name':name}

        return df1

    # Create partitions of the data
    def create_partitions(self, df, partition_size):
        # Remainder of division of dataframe length by partition_size
        remainder = len(df) % partition_size

        if remainder == 0:
            df_no_remainder = df
        else:
            # Remove the remainder to make dataframe divisible by partition_size
            df_no_remainder = df.iloc[:-remainder]

        no_of_partitions = len(df_no_remainder) / partition_size
        # Split the dataframe into partitions
        arr_partitions = np.array_split(df_no_remainder.values, no_of_partitions)

        dict_partitions = defaultdict()

        for i, arr_partition in enumerate(arr_partitions):
            df_part = pd.DataFrame(arr_partition, columns=df.columns)
            dict_partitions[f"{df.attrs['name']}_{i}"] = df_part
        
        return dict_partitions

    def create_train_test_idx(self, idx):
        # Return all possible combinations of train and test partitions within the
        # same dataframe and without using same partition for both train and test
        train_test_idx = [(tr, te) for tr in idx for te in idx if tr != te]
        return train_test_idx
    
@pytest.fixture(scope='module')
def my_data_fixture():
    return my_data()