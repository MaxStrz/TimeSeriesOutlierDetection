import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import RobustScaler

class AnalysisPaths:
    def __init__(self, analysis_name: str):
        # File paths to the data
        self.data_path = os.path.join(analysis_name, "data")
        self.raw_folder_path = os.path.join(self.data_path, 'raw_csv')
        self.raw_file_paths = [
            os.path.join(self.raw_folder_path, p)
            for p in os.listdir(self.raw_folder_path)
            ]

class Config:

    # Confirm column names before generating dataframe    
    column_names = ["brand_id", "car_id", "car_component", "component_id",
                     "timestamp", "component_part_id", "sensor_1", "sensor_2", 
                     "sensor_3", "sensor_4", "non_defect_likelihood"]

    # Confirm data types before generating dataframe
    dtypes = {'brand_id':'str',
          'car_id':'int32',
          'car_component': 'str',
          'component_id':'int32',
          'timestamp':'int64',
          'component_part_id':'str',
          'sensor_1':'float64',
          'sensor_2':'float64',
          'sensor_3':'float64',
          'sensor_4':'float64',
          'non_defect_likelihood':'float64'
         }

class TransformRawData:
    sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
    idx = ['car_id', 'component_id', 'component_part_id']
    config = Config()

    # Name of the analysis must correspond to the folder name
    def __init__(self, analysis_name: str): 
        self.analysis_name = analysis_name
        self._paths = AnalysisPaths(self.analysis_name)
        self.labels = None
        self.df_all = None
        self._called_methods = set()

    def _make_df(self,
                 file_path: str,
                 dtypes=None,
                 index_col: int=0,
                 separator: str=',',
                 float_separator: str='.',
                 date_column_name: str='timestamp'
                 ) -> pd.DataFrame:
    
        df = pd.read_csv(file_path,
                        dtype=dtypes,
                        index_col=index_col,
                        sep=separator,
                        decimal=float_separator,
                        parse_dates=[date_column_name]
                        )

        return df

    def create_df_all(self):
        dataframes = []

        for file_path in self._paths.raw_file_paths:
            # Generate dataframe from file. Expect message if successful.
            df_cs = self._make_df(file_path=file_path,
                          dtypes=self.config.dtypes,
                          index_col=False,
                          separator=',',
                          float_separator='.',
                          date_column_name="timestamp"
                         )

            dataframes.append(df_cs)

        self.df_all = pd.concat(dataframes, ignore_index=True)

        self._called_methods.add("create_df_all")

        return self

    def create_multi_index(self):
        df = self.df_all
        index = self.idx
        
        df = df.set_index(index)
        df.index.names = index

        self.df_all = df

        self._called_methods.add("create_multi_index")

        return self

    def _find_outliers(self, df):
        mus = df.mean()
        stds = df.std()
        over_stds_idx = (np.abs(df - mus) > stds * 4).any(axis=1)
        return over_stds_idx
    
    def clean_data(self):
        cols_to_clean = self.sensors
        df = self.df_all

        # Only the kpis are required for removing outliers
        df = df[cols_to_clean]
        
        # boolean indexes of outliers
        over_stds_idx = self._find_outliers(df)

        # remove outliers
        # ~ ist a negation operator. It turns boolean values around.
        self.df_all = self.df_all[~over_stds_idx].reset_index(drop=True)

        self._called_methods.add("clean_data")

        return self

    def remove_columns(self):
        df = self.df_all
        cols_to_keep = self.sensors
        df = df[cols_to_keep]
        self.df_all = df

        self._called_methods.add("remove_columns")

        return self

    def robust_scale_data(self):
        
        df = self.df_all
        data = self.df_all.values

        # Create and save scaler as attribute
        scaler = RobustScaler().fit(data)

        # Scale all data
        scaled_data = scaler.transform(data)

        # Create new dataframe with scaled data
        self.df_all = pd.DataFrame(scaled_data,
                                   columns=df.columns,
                                   index=df.index)
        
        # Save scaler as attribute
        self.robust_scaler = scaler

        self._called_methods.add("robust_scale_data")

        return self

    def run(self):
        return(
            self.create_df_all()
            .create_multi_index()
            .clean_data()
            .remove_columns()
            .robust_scale_data()
        )

    def __repr__(self):
        methods = {"create_df_all", "create_multi_index", "clean_data",
                   "remove_columns", "robust_scale_data", "None"}
        pending_methods = methods - self._called_methods
        note = f"TransformRawData instance. Pending methods: {pending_methods}"
        return note