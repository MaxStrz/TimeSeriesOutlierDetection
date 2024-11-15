import os
import pandas as pd

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
        self.paths = AnalysisPaths(self.analysis_name)
        self.labels = None
        self.df_all = None

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

        for file_path in self.paths.raw_file_paths:
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

    def create_multi_index(self):
        df = self.df_all
        index = self.idx
        
        df = df.set_index(index)
        df.index.names = index

        self.df_all = df