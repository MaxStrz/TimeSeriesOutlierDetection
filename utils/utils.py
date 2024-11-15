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
    """ Turns raw csv files into single cleaned, scaled dataframe.

    The class is designed to be used in a pipeline. The methods are
    designed to be called in a specific order. The order is as follows:
    1. create_df_all
    2. create_multi_index
    3. clean_data
    4. remove_columns
    5. robust_scale_data
    
    The class has a run method that calls all the methods in the correct order.

    Parameters
    ----------
    analysis_name : str
        Name of the analysis. Must correspond to existing folder name.
        The folder must contain a 'data' folder with a 'raw_csv' folder
        containing the raw csv files.
    
    Attributes
    ----------
    _paths : AnalysisPaths
        Creates an instance of the AnalysisPaths class.
        The class creates the paths to the raw csv files based on the
        analysis_name.
    
    labels : pd.Series
        Series with boolean values. True if component part deemed outlier.

    df_all : pd.DataFrame
        Dataframe with all the data from the raw csv files. The dataframe is
        transformed, cleaned and scaled by the methods in the class.
    
    robust_scaler : sklearn.preprocessing.RobustScaler
        Scaler used to scale the data in the df_all dataframe.
    
    _called_methods : set
        Set with the names of the methods that have been called. Used to
        keep track of the order of the methods called.
    
    """
    sensors = ['sensor_1', 'sensor_2', 'sensor_3', 'sensor_4']
    idx = ['car_id', 
           'component_id', 
           'component_part_id']
    tempo_idx = 'temporal_index'
    time_column = 'timestamp'
    config = Config()

    # Name of the analysis must correspond to the folder name
    def __init__(self, analysis_name: str): 
        self.analysis_name = analysis_name
        self._paths = AnalysisPaths(self.analysis_name)
        self.labels = None
        self.df_all = None
        self.robust_scaler = None
        self._called_methods = set()

    def _make_df(self,
                 file_path: str,
                 dtypes=None,
                 index_col: int=0,
                 separator: str=',',
                 float_separator: str='.',
                 date_column_name: str='timestamp'
                 ) -> pd.DataFrame:
        
        """ Generate dataframe from csv file."""
    
        df = pd.read_csv(file_path,
                        dtype=dtypes,
                        index_col=index_col,
                        sep=separator,
                        decimal=float_separator,
                        parse_dates=[date_column_name]
                        )

        return df

    def create_df_all(self):

        """Create a single dataframe from all the raw csv files."""

        dataframes = []

        for file_path in self._paths.raw_file_paths:
            # Generate dataframe from file. Expect message if successful.
            df_cs = self._make_df(file_path=file_path,
                          dtypes=self.config.dtypes,
                          index_col=False,
                          separator=',',
                          float_separator='.',
                          date_column_name=self.time_column
                         )

            dataframes.append(df_cs)

        self.df_all = pd.concat(dataframes, ignore_index=True)

        self._called_methods.add("create_df_all")

        return self

    def _find_outliers(self, df):

        """Find outliers in dataframe based on standard deviation."""

        mus = df.mean()
        stds = df.std()
        over_stds_idx = (np.abs(df - mus) > stds * 4).any(axis=1)
        return over_stds_idx
    
    def clean_data(self):

        """Remove outliers from the dataframe."""

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
    
    def _create_tempo_idx_column(self, df, idx, time_column):

        """Create a temporal index column to be used in the multi index."""

        tempo_idx = self.tempo_idx

        df = df.sort_values(by=idx + [time_column], ascending=True)

        df[tempo_idx] = df.groupby(idx).cumcount()

        return df

    def create_multi_index(self):

        """Create a multi index for the dataframe."""

        df = self.df_all
        idx = self.idx
        new_idx = self.idx + [self.tempo_idx]
        time_column = self.time_column
        
        df = self._create_tempo_idx_column(df, idx, time_column)
        df = df.set_index(new_idx)
        df.index.names = new_idx

        self.df_all = df

        self._called_methods.add("create_multi_index")

        return self
    
    def create_labels(self,
                      score_column='non_defect_likelihood',
                      score_threshold=0.5,
                      threshold_count=500
                      ):
        
        """Create outlier labels for each component part."""

        df = self.df_all

        low_score = df[score_column] < score_threshold
        counts = low_score.groupby(level=[0, 1, 2]).sum()
        
        self.labels = counts > threshold_count

        self._called_methods.add("create_labels")

        return self

    def remove_columns(self):

        """Only keep the sensor columns in the dataframe."""

        df = self.df_all
        cols_to_keep = self.sensors
        df = df[cols_to_keep]
        self.df_all = df

        self._called_methods.add("remove_columns")

        return self

    def robust_scale_data(self):
        
        """Scale the data using a RobustScaler."""

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

        """Run all the methods in the correct order."""

        return(
            self.create_df_all()
            .clean_data()
            .create_multi_index()
            .create_labels()
            .remove_columns()
            .robust_scale_data()
        )

    def __repr__(self):

        """Return instance representaion with pending methods."""

        methods = {"create_df_all", "clean_data", "create_multi_index",
                   "create_labels", "remove_columns", "robust_scale_data",
                   "None"}
        pending_methods = methods - self._called_methods
        note = f"TransformRawData instance. Pending methods: {pending_methods}"
        return note