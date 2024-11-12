from collections import defaultdict
import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import pickle
from pythresh.thresholds.dsn import DSN
import pythresh
from sklearn.preprocessing import RobustScaler
import os

class AnalysisPaths:
    def __init__(self, analysis_name: str):
        # File paths to the data
        self.data_path = os.path.join(analysis_name, "data")
        self.raw_folder_path = os.path.join(self.data_path, 'raw_csv')
        self.raw_file_paths = [os.path.join(self.raw_folder_path, p) for p in os.listdir(self.raw_folder_path)]
        self.coil_serial_pkl_path = os.path.join(self.data_path, "coil_serials", "s{}_cs{}.pkl")
        self.coil_element_pkl_path = os.path.join(self.data_path, "coil_elements", "s{}_cs{}_ce{}.pkl")
        self.cleaned_scaled_pkl_path = os.path.join(self.data_path, "cleaned_scaled_coil_elements", "s{}_cs{}_ce{}.pkl")
        self.cleaned_pkl_path = os.path.join(self.data_path, "cleaned_coil_elements", "s{}_cs{}_ce{}.pkl")


class Config:

    # # File paths to the data
    # coil_serial_pkl_path = "data/coil_serials/s{}_cs{}.pkl"
    # coil_element_pkl_path = "data/coil_elements/s{}_cs{}_ce{}.pkl"
    # cleaned_scaled_pkl_path = "data/cleaned_scaled_coil_elements/s{}_cs{}_ce{}.pkl"
    # cleaned_pkl_path = "data/cleaned_coil_elements/s{}_cs{}_ce{}.pkl"

    # Confirm column names before generating dataframe
    column_names = ["material", "serial", "coil_name",
                "coil_serial", "coil_element_name",
                "avg_timestamp", "SSR", "CSP", "CNL",
                "CSI", "classifier_result"]

    # Confirm data types before generating dataframe
    dtypes = {'material':'str',
          'serial':'int32',
          'coil_name': 'str',
          'coil_serial':'int32',
          'coil_element_name':'str',
          'avg_timestamp':'int64',
          'SSR':'float64',
          'CSP':'float64',
          'CNL':'float64',
          'CSI':'float64',
          'classifier_result':'float64'
         }

    # serial, coil_serial and file path for all labeled (labels not included) coil serials
    dictionary_of_files = {
                           (196467, 2499):{"Filepath":"data/kusto_downloads/m11344916_s196467_HeadNeck_20_TCS_cs2499_bis20240609.csv"},
                           (196467, 3129):{"Filepath":"data/kusto_downloads/m11344916_s196467_HeadNeck_20_TCS_cs3129_bis20240609.csv"},
                           (196302, 1650):{"Filepath":"data/kusto_downloads/m11344916_s196302_HeadNeck_20_TCS_cs1650_bis20240609.csv"},
                           (196300, 1310):{"Filepath":"data/kusto_downloads/m11344916_s196300_HeadNeck_20_TCS_cs1310_bis20240609.csv"},
                           (196217, 1179):{"Filepath":"data/kusto_downloads/m11344916_s196217_HeadNeck_20_TCS_cs1179_bis20240609.csv"},
                           (196399, 2073):{"Filepath":"data/kusto_downloads/m11344916_s196399_HeadNeck_20_TCS_cs2073_bis20240609.csv"},
                           (196567, 3064):{"Filepath":"data/kusto_downloads/m11344916_s196567_HeadNeck_20_TCS_cs3064_bis20240609.csv"},
                           (196408, 2158):{"Filepath":"data/kusto_downloads/m11344916_s196408_HeadNeck_20_TCS_cs2158_bis20240609.csv"},
                           (196234, 1999):{"Filepath":"data/kusto_downloads/m11344916_s196234_HeadNeck_20_TCS_cs1999_bis20240609.csv"},
                           (196236, 1381):{"Filepath":"data/kusto_downloads/m11344916_s196236_HeadNeck_20_TCS_cs1381_bis20240609.csv"},
                           (196374, 1987):{"Filepath":"data/kusto_downloads/m11344916_s196374_HeadNeck_20_TCS_cs1987_bis20240609.csv"}, 
                           }

# returns indexes of values that are stds standard deviations away from the mean
# of the input array
def indexes_of_stds_devs(arr, stds):
    mean = np.mean(arr) # mean of the array
    std = np.std(arr) # standard deviation of the array
    over_stds = np.abs(arr - mean) > std * stds # boolean array of values that are stds standard deviations away from the mean
    index_of_trues = np.where(over_stds)[0] # indexes of original array where deviation is greater than stds standard deviations from the mean
    return index_of_trues

def dict_key_value_generator(gen_dict):
    for serial, coil_serials in gen_dict.items():
        for coil_serial, coil_elements in coil_serials.items():
            for coil_element in coil_elements:
                yield serial, coil_serial, coil_element

def create_s_cs_ce_name_generator():
    with open("data/structure_dict.pkl", "rb") as f:
        structure_dict = pickle.load(f)

    for serial, coil_serials in structure_dict.items():
        for coil_serial, coil_elements in coil_serials.items():
            for coil_element in coil_elements:
                yield serial, coil_serial, coil_element

class TransformRawData:
    def __init__(self, analysis_name: str): # Name of the analysis must correspond to the folder name
        self.analysis_name = analysis_name

    def create_analysis_paths(self):
        self.paths = AnalysisPaths(self.analysis_name)

    def _make_df(self,
            file_path: str,
            column_names: list[str],
            dtypes=None,
            index_col: int=0,
            separator: str=',',
            float_separator: str='.',
            date_column_name: str='avg_timestamp'
            ) -> pd.DataFrame:
    
        df = pd.read_csv(file_path,
                        dtype=dtypes,
                        index_col=index_col,
                        sep=separator,
                        decimal=float_separator,
                        parse_dates=[date_column_name]
                        )
        
        # ueberpruefe ob die Liste der Spaltennamen richtig ist
        enote = "Spaltennamen sind nicht korrekt"
        assert sorted(list(df.columns)) == sorted(column_names), enote
            
        note = "Erfolgreich: Daten erfolgreich in einen DataFrame umgewandelt"

        print(note)
        return df

    def check_for_duplicate_serials_and_coil_serials(self):

        # Check that file paths have been created
        enote = "File paths have not been created yet. Run instance.create_analysis_paths() first."
        assert hasattr(self, 'paths'), enote

        for file_path in self.paths.raw_file_paths:
            # Generate dataframe from file. Expect message if successful.
            df_cs = self._make_df(file_path=file_path,
                          column_names=Config.column_names,
                          dtypes=Config.dtypes,
                          index_col=False,
                          separator=';',
                          float_separator=',',
                          date_column_name="avg_timestamp"
                         )
            
            enote = f"More than one serial in dataframe for {file_path}"
            assert len(df_cs['serial'].unique()) == 1, enote
            
            enote = f"More than one coil serial in dataframe for {file_path}"
            assert len(df_cs['coil_serial'].unique()) == 1, enote

            print("No duplicate serials or coil serials found in:", file_path)

        print("Files checked for duplicate serials and coil serials.")

    def save_cs_dfs_as_pkls(self):
        self.s_cs_index_tupels = []

        for file_path in self.paths.raw_file_paths:
            # Generate dataframe from file. Expect message if successful.
            df_cs = self._make_df(file_path=file_path,
                          column_names=Config.column_names,
                          dtypes=Config.dtypes,
                          index_col=False,
                          separator=';',
                          float_separator=',',
                          date_column_name="avg_timestamp"
                         )
            serial = df_cs['serial'].unique()[0]
            coil_serial = df_cs['coil_serial'].unique()[0]
            self.s_cs_index_tupels.append((serial, coil_serial))

            cs_filepath = self.paths.coil_serial_pkl_path.format(serial, coil_serial)
            os.makedirs(os.path.dirname(cs_filepath), exist_ok=True)

            df_cs.to_pickle(cs_filepath)

        print("Index tupels for serial and coil_serial now available as attribute self.s_cs_index_tupels")

    def create_structure_dict(self):
        structure_dict = {}
        self.s_cs_ce_index_tupels = {}
        
        for serial, coil_serials in self.s_cs_index_tupels:
            # serial is the serial number itself
            # We actually only need the keys in this for-loop
            # Constructs keys for each serial number and an empty dictionary as their values
            structure_dict[serial] = {}
        for serial, coil_serials in self.s_cs_index_tupels:
            # A subset of keys for coil serials are added to each serial dictionary
            # The value of each serial / coil serial pair is an empty dictionary
            structure_dict[serial][coil_serials] = {}
        for serial, coil_serials in structure_dict.items():
            # for each serial / coil serial pair, the coil elements are gathered from the stored data frames
            # Each coil element is then added as a key in the coil serial dictionary
            # The value of each serial, coil serial, coil element combination is an empty dictionary.
            for coil_serial in coil_serials:
                df_cs = pd.read_pickle(self.paths.coil_serial_pkl_path.format(serial, coil_serial))
                coil_elements = df_cs['coil_element_name'].unique()
                for coil_element in coil_elements:
                    structure_dict[serial][coil_serial][coil_element] = {}
                    self.s_cs_ce_index_tupels[(serial, coil_serial, coil_element)] = {}
        
        with open(os.path.join(self.paths.data_path, "structure_dict.pkl"), "wb") as f:
            pickle.dump(structure_dict, f)

        with open(os.path.join(self.paths.data_path, "index_tupels.pkl"), "wb") as f:
            pickle.dump(self.s_cs_ce_index_tupels, f)

        print("Index tupels for serial, coil_serial and coil_element now available as attribute self.s_cs_ce_index_tupels")

    def save_ce_dfs_as_pkls(self):
        for serial, coil_serial, coil_element in self.s_cs_ce_index_tupels.keys():
            df_cs = pd.read_pickle(self.paths.coil_serial_pkl_path.format(serial, coil_serial))
            df_ce = df_cs.loc[df_cs['coil_element_name']==coil_element].reset_index(drop=True)
            # Define file path for each coil element
            ce_filepath = self.paths.coil_element_pkl_path.format(serial, coil_serial, coil_element)
            # Create directories if they do not exist
            os.makedirs(os.path.dirname(ce_filepath), exist_ok=True)
            df_ce.to_pickle(ce_filepath)

    # def save_ce_binary_labels(self):
    #     data_instance = GetData()
    #     gen = data_instance.gen_raw_ce_df(labels=True)

    #     labels = {}

    #     for df in gen:
    #         label = int(df['classifier_result'].max() < 0.95)
    #         mindex = (df.s_name, df.cs_name, df.ce_name)
    #         labels[mindex] = label # Label
        
    #     label_index = pd.MultiIndex.from_tuples(labels.keys(), names=['serial', 'coil_serial', 'coil_element'])
    #     self.df_ce_binary_labels_ = pd.DataFrame.from_dict(labels, orient='index', columns=['label'])
    #     self.df_ce_binary_labels_.index = label_index

    #     self.df_ce_binary_labels_.to_pickle("data/df_ce_binary_labels_.pkl")

    def clean_ce_data(self):
        for serial, coil_serial, coil_element in self.s_cs_ce_index_tupels.keys():
            df_ce_kpis = pd.read_pickle(self.paths.coil_element_pkl_path.format(serial, coil_serial, coil_element))

            # remove all datapoints more than standard_devs standard deviations from mean
            #standard_devs = 3
            #df_ce_kpis_cleaned = remove_stds(df_ce_kpis, standard_devs)

            # to-do capture removed datapoints
            # CURRENTLY DO NOTHING

            # Define file path for each coil element
            ce_cleaned_filepath = self.paths.cleaned_pkl_path.format(serial, coil_serial, coil_element)
            # Create directories if they do not exist
            os.makedirs(os.path.dirname(ce_cleaned_filepath), exist_ok=True)

            df_ce_kpis_cleaned = df_ce_kpis[["SSR", "CSP", "CNL", "CSI"]]
            df_ce_kpis_cleaned.to_pickle(ce_cleaned_filepath)

    def scale_ce_dfs(self):

        dict_scaled = self.s_cs_ce_index_tupels.copy()
        
        for serial, coil_serial, coil_element in self.s_cs_ce_index_tupels.keys():
            df = pd.read_pickle(self.paths.cleaned_pkl_path.format(serial, coil_serial, coil_element))
            dict_scaled[(serial, coil_serial, coil_element)] = df.values

        def ce_dict_to_df(my_dict):
            temp_dict = {(s, cs, ce, i): value 
                         for s, cs, ce in my_dict.keys()
                         for i, value in enumerate(my_dict[(s, cs, ce)])
                         }
            multi_index = pd.MultiIndex.from_tuples(temp_dict.keys(), names=['serial', 'coil_serial', 'coil_element', 'dataindex'])
            df_ces = pd.DataFrame.from_dict(temp_dict, orient='index', columns=df.columns)
            df_ces.index = multi_index

            return df_ces

        df_all = ce_dict_to_df(dict_scaled)
        df_all.to_pickle(os.path.join(self.paths.data_path, "df_all_cleaned.pkl"))

        self.transformer = RobustScaler().fit(df_all)
        df_all_cleaned_scaled = pd.DataFrame(self.transformer.transform(df_all), columns=df.columns)
        df_all_cleaned_scaled.to_pickle(os.path.join(self.paths.data_path, "df_all_cleaned_scaled.pkl"))

        for serial, coil_serial, coil_element in self.s_cs_ce_index_tupels.keys():
            df = pd.read_pickle(self.paths.cleaned_pkl_path.format(serial, coil_serial, coil_element))
            df_ce_kpis_scaled = pd.DataFrame(self.transformer.transform(df), columns=df.columns)

            # Define file path for each coil element
            ce_cleaned_scaled_filepath = self.paths.cleaned_scaled_pkl_path.format(serial, coil_serial, coil_element)
            
            # Create directories if they do not exist
            os.makedirs(os.path.dirname(ce_cleaned_scaled_filepath), exist_ok=True)

            df_ce_kpis_scaled.to_pickle(ce_cleaned_scaled_filepath)

    def create_labels(self):
        dict_labels = self.s_cs_ce_index_tupels.copy()

        for serial, coil_serial, coil_element in self.s_cs_ce_index_tupels.keys():
            df = pd.read_pickle(self.paths.coil_element_pkl_path.format(serial, coil_serial, coil_element))
            low_score = df['classifier_result'] < 0.5
            count_low_score = low_score.sum()
            label = count_low_score > 500

            dict_labels[(serial, coil_serial, coil_element)] = label
        
        s_labels = pd.Series(dict_labels)
        s_labels.index.names = ['serial', 'coil_serial', 'coil_element']
        
        with open(os.path.join(self.paths.data_path, 'labels.pkl'), 'wb') as f:
            pickle.dump(s_labels, f)
        
        return s_labels

class GetData:
    
    def __init__(self,
                 analysis_name: str,
                ):
        self.analysis_name = analysis_name
        self.paths = AnalysisPaths(analysis_name)
        self.s_cs_ce_index_tupels = self.get_tupels()
        self.structure_dict = self.get_structure_dict()

    def get_tupels(self):
        with open(os.path.join(self.paths.data_path, "index_tupels.pkl"), "rb") as f:
            index_tupels = pickle.load(f)
        return index_tupels
    
    def get_structure_dict(self):
        with open("data/structure_dict.pkl", "rb") as f:
            structure_dict = pickle.load(f)
        return structure_dict

    def df_gen_cleaned_scaled_ce(self):
        for serial, coil_serial, coil_element in self.s_cs_ce_index_tupels:
            df = self.get_cleaned_scaled_ce_df(serial, coil_serial, coil_element)
            yield df
    
    def gen_raw_ce_df(self, labels=False):
        for serial, coil_serial, coil_element in self.s_cs_ce_index_tupels:
            df = self.get_ce_df(serial, coil_serial, coil_element, labels)  
            yield df
    
    def cleaned_ce_df_generator(self):
        for serial, coil_serial, coil_element in self.s_cs_ce_index_tupels:
            df = self.get_cleaned_ce_df(serial, coil_serial, coil_element)
            yield df
    
    def ce_dict_to_df(self, my_dict): 
        temp_dict = {(s, cs, ce): my_dict[s][cs][ce]
                    for s in my_dict.keys()
                    for cs in my_dict[s].keys()
                    for ce in my_dict[s][cs].keys()
                    }
        self.df_ces = pd.DataFrame.from_dict(temp_dict, orient='index')
        self.df_ces.index.names = ['serial', 'coil_serial', 'coil_element']
    
    def _part_dict_to_df(self, my_dict):
        temp_dict = {(s, cs, ce, part): None
                    for s in my_dict.keys()
                    for cs in my_dict[s].keys()
                    for ce in my_dict[s][cs].keys()
                    for part in my_dict[s][cs][ce].keys()
                    }
        multi_index = pd.MultiIndex.from_tuples(temp_dict.keys(), names=['serial', 'coil_serial', 'coil_element', 'partition'])
        my_df = pd.DataFrame(index=multi_index)
        self.df_ce_partition_indexes = my_df      

    def create_df_ces(self, my_dict):
        temp_dict = {(s, cs, ce): None
                    for s in my_dict.keys()
                    for cs in my_dict[s].keys()
                    for ce in my_dict[s][cs].keys()
                    }
        multi_index = pd.MultiIndex.from_tuples(temp_dict.keys(), names=['serial', 'coil_serial', 'coil_element'])
        my_df = pd.DataFrame(index=multi_index)
        
        return my_df

    def create_dict_ce_partitions(self, partition_size=50):
        self.dict_ce_partitions = defaultdict(dict)

        for df in self.df_gen_cleaned_scaled_ce():
            remainder = len(df) % partition_size # remainder of the division of the length of the dataframe by the partition size

            if remainder == 0:
                df_no_remainder = df
            else:
                df_no_remainder = df.iloc[:-remainder] # remove the remainder to make the dataframe divisible by the partition size

            no_of_partitions = len(df_no_remainder) / partition_size 
            arr_partitions = np.array_split(df_no_remainder.values, no_of_partitions) # split the dataframe into partitions

            for i, arr_partition in enumerate(arr_partitions):
                df_part = pd.DataFrame(arr_partition, columns=df.columns)
                self.dict_ce_partitions[(df.serial, df.coil_serial, df.coil_element, i)] = df_part
        
        with open(os.path.join(self.paths.data_path, f'dict_ce_partitions_{partition_size}.pkl'), 'wb') as f:
            pickle.dump(self.dict_ce_partitions, f)

    def partition_generator(self, partition_size=50):
        self.create_dict_ce_partitions(partition_size=partition_size)
        
        def my_partition_generator():
            for tuple_index, df_partition in self.dict_ce_partitions.items():
                df_partition.part_name = f"{tuple_index[0]}_{tuple_index[1]}_{tuple_index[2]}_part{tuple_index[3]}" # name the partition
                df_partition.s_name = tuple_index[0]
                df_partition.cs_name = tuple_index[1]
                df_partition.ce_name = tuple_index[2]
                df_partition.partition_number = tuple_index[3]

                yield df_partition

        return my_partition_generator()

    def create_train_test_list(self, partition_size=50) -> list:

        # assert that self.dict_ce_partitions has been created
        enote = "dict_ce_partitions has not been created yet. Run instance.create_dict_ce_partitions() or partition_generator() first."
        assert hasattr(self, 'dict_ce_partitions'), enote

        pin = self.dict_ce_partitions.keys() # partition index
        # Return all possible combinations of train and test partitions within the same coil element
        # and without using the same partition for both train and test
        self.part_tr_te_indexes = [(tr, te) for tr in pin for te in pin if tr != te and tr[0:3] == te[0:3]]

        with open(os.path.join(self.paths.data_path, f'partition_tr_te_indexes_{partition_size}.pkl'), 'wb') as f:
            pickle.dump(self.part_tr_te_indexes, f)

        return self.part_tr_te_indexes


        # self.pairs = []
        # for key_train in self.dict_ce_partitions.keys():
        #     (s_train, cs_train, ce_train, part_train) = key_train
        #     for key_test in self.dict_ce_partitions.keys():
        #         (s_test, cs_test, ce_test, part_test) = key_test
        #         if key_train == key_test or (s_train, cs_train, ce_train) != (s_test, cs_test, ce_test):
        #             continue
        #         self.pairs.append([key_train, key_test])
        
        # return self.pairs

    def gen_train_test_partitions(self):
        pin = self.dict_ce_partitions.keys() # partition index
        # Return all possible combinations of train and test partitions within the same coil element
        # and without using the same partition for both train and test
        tr_te_indexes = ((tr, te) for tr in pin for te in pin if tr != te and tr[0:3] == te[0:3])
        return tr_te_indexes

    def time_series_splits(self, splitter):

        self.dict_ce_splits = copy.deepcopy(self.structure_dict)

        for df in self.ce_df_generator():
            for i, (X_index, y_index) in enumerate(splitter.split(df)):

                self.dict_ce_splits[df.serial][df.coil_serial][df.coil_element][i] = {'x': df.loc[X_index], 'y': df.loc[y_index]}

        def my_split_generator():
            for df in self.ce_df_generator():
                split_index_ = self.dict_ce_splits[df.serial][df.coil_serial][df.coil_element].keys() # array of split indexes for df_ce

                for s in split_index_:
                    arr_x_split = self.dict_ce_splits[df.serial][df.coil_serial][df.coil_element][s]['x'] # split array of x values
                    arr_y_split = self.dict_ce_splits[df.serial][df.coil_serial][df.coil_element][s]['y'] # split array of y values
                    
                    df_x = pd.DataFrame(arr_x_split, columns=df.columns) # partition array as a dataframe
                    df_x.name = f"{df.serial}_{df.coil_serial}_{df.coil_element}_xsplit{s}" # name the partition
                    df_x.serial = df.serial
                    df_x.coil_serial = df.coil_serial
                    df_x.coil_element = df.coil_element
                    df_x.split_number = s

                    df_y = pd.DataFrame(arr_y_split, columns=df.columns) # partition array as a dataframe
                    df_y.name = f"{df.serial}_{df.coil_serial}_{df.coil_element}_ysplit{s}" # name the partition
                    df_y.serial = df.serial
                    df_y.coil_serial = df.coil_serial
                    df_y.coil_element = df.coil_element
                    df_y.split_number = s

                    yield df_x, df_y

        return my_split_generator()

    def get_cs_df(self, serial, coil_serial, labels=False):
        df = pd.read_pickle(self.paths.coil_serial_pkl_path.format(serial, coil_serial))
        if labels == False:
            df = df.drop('classifier_result', axis=1)
        else:
            pass
        return df           

    def get_ce_df(self, serial, coil_serial, coil_element, labels=False):
        df = pd.read_pickle(self.paths.coil_element_pkl_path.format(serial, coil_serial, coil_element))
        if labels == False:
            df = df.drop('classifier_result', axis=1)
        else:
            pass
        df.s_name = serial
        df.cs_name = coil_serial
        df.ce_name = coil_element
        return df
    
    def get_cleaned_ce_df(self, serial, coil_serial, coil_element):
        df = pd.read_pickle(self.paths.cleaned_pkl_path.format(serial, coil_serial, coil_element))
        df.name = f"{serial}_{coil_serial}_{coil_element}"
        df.my_index_tupel = (serial, coil_serial, coil_element)
        df.s_name = serial
        df.cs_name = coil_serial
        df.ce_name = coil_element
        return df
        
    def get_cleaned_scaled_ce_df(self, serial, coil_serial, coil_element):
        df = pd.read_pickle(self.paths.cleaned_scaled_pkl_path.format(serial, coil_serial, coil_element))
        df.name = f"{serial}_{coil_serial}_{coil_element}"
        df.serial = serial
        df.coil_serial = coil_serial
        df.coil_element = coil_element
        df.s_name = serial
        df.cs_name = coil_serial
        df.ce_name = coil_element
        return df

    def get_df_all_cleaned(self):
        df_all = pd.read_pickle("data/df_all_cleaned.pkl")
        return df_all
    
    def get_df_all_cleaned_scaled(self):
        df_all = pd.read_pickle("data/df_all_cleaned_scaled.pkl")
        return df_all

    def get_index_tupels(self):
        with open("data/index_tupels.pkl", "rb") as f:
            index_tupels = pickle.load(f)

        return index_tupels

    # def get_labels(self):
    #     with open('data/labels.pkl', 'rb') as f:
    #         dict_labels = pickle.load(f)

        return dict_labels

    def get_outlier_classes(self, thresholder='mixmod'):
        with open(f'{self.analysis_name}/results/df_outlier_class.pkl', 'rb') as f:
            df_outlier_class = pickle.load(f)
        s_outlier_class = df_outlier_class[thresholder]
        return s_outlier_class

    def get_detected_outlier_dfs(self, thresholder: str):
        with open(f'{self.analysis_name}/results/df_outlier_class.pkl', 'rb') as f:
            df_outlier_class = pickle.load(f)
        outlier_dfs = []
        s_outlier_class = df_outlier_class[thresholder]
        outlier_indexes = s_outlier_class[s_outlier_class == 1].index
        for index in outlier_indexes:
            dfi = self.get_cleaned_scaled_ce_df(index[0], index[1], index[2])
            outlier_dfs.append([index, dfi])
        return outlier_dfs

    def get_labels(self):
        with open(f'{self.analysis_name}/data/labels.pkl', 'rb') as f:
            labels = pickle.load(f)
        return labels

    def get_false_positive_indexes(self, thresholder):
        labels = self.get_labels()
        s_outlier_class = self.get_outlier_classes(thresholder)
        fps = s_outlier_class & ~labels
        self.false_positive_indexes = list(fps[fps].index)

        return self.false_positive_indexes
    
    def get_false_positive_dfs(self, thresholder):
        fpis = self.get_false_positive_indexes(thresholder)
        fp_dfs = []
        for index in fpis:
            dfi = self.get_cleaned_scaled_ce_df(index[0], index[1], index[2])
            fp_dfs.append([index, dfi])
        return fp_dfs
    
    def get_false_negative_indexes(self, thresholder):
        labels = self.get_labels()
        s_outlier_class = self.get_outlier_classes(thresholder)
        fns = labels & ~s_outlier_class
        self.false_negative_indexes = list(fns[fns].index)

        return self.false_negative_indexes
        
    def get_false_negatives_dfs(self, thresholder):
        fnis = self.get_false_negative_indexes(thresholder)
        fn_dfs = []
        for index in fnis:
            dfi = self.get_cleaned_scaled_ce_df(index[0], index[1], index[2])
            fn_dfs.append([index, dfi])
        return fn_dfs
    
    def get_true_positive_indexes(self, thresholder):
        labels = self.get_labels()
        s_outlier_class = self.get_outlier_classes(thresholder)
        tps = labels & s_outlier_class
        self.true_positive_indexes = list(tps[tps].index)

        return self.true_positive_indexes

    def get_true_positives_dfs(self, thresholder):
        tps = self.get_true_positive_indexes(thresholder)
        tp_dfs = []
        for index in tps:
            dfi = self.get_cleaned_scaled_ce_df(index[0], index[1], index[2])
            tp_dfs.append([index, dfi])
        return tp_dfs
    
    def get_univariate_outlier_scores(self):
        with open(f'{self.analysis_name}/results/lof_outlier_scores.pkl', 'rb') as f:
            univariate_outlier_scores = pickle.load(f)
            univariate_outlier_scores.sort_values(ascending=False, inplace=True) 
        return univariate_outlier_scores
    
    def get_scores_percentiles(self, serial, coil_serial, coil_element):
        outlier_scores = pd.read_pickle(f'{self.analysis_name}/results/outlier_scores.pkl')
        df_ranks = outlier_scores.rank()
        df_ranks = round(df_ranks / len(df_ranks), ndigits=2)
        percentiles = df_ranks.loc[(serial, coil_serial, coil_element)].sort_values(ascending=False)

        return percentiles

    def gen_scores_percentiles(self, index_df_list):
        for item in index_df_list:
            yield self.get_scores_percentiles(item[0][0], item[0][1], item[0][2])

class UniThresh:
    def __init__(self, 
                 df_decision_scores_: pd.DataFrame,
                 thresh_model_instance,
                ) -> None:
        self.df_decision_scores_ = df_decision_scores_
        self.thresh_model_instance = thresh_model_instance
        self.df_labels_ = self.df_decision_scores_.copy().map(lambda x: np.nan)
        #self.df_ce_outliers = None

    def eval(self):
        if isinstance(self.df_decision_scores_, pd.Series):
            scores = self.df_decision_scores_.values
            self.df_labels_ = self.thresh_model_instance.eval(scores)
        elif isinstance(self.df_decision_scores_, pd.DataFrame):  
            for kpi in self.df_decision_scores_.columns:
                scores = self.df_decision_scores_[kpi].values
                self.df_labels_[kpi] = self.thresh_model_instance.eval(scores)
        #filter = self.df_labels_[self.df_labels_.columns]==1
        #self.df_ce_outliers = self.df_labels_[(filter).any(axis=1)]

    def save_results(self, model_name: str):
        self.df_labels_.to_pickle(f"results/df_uni_{model_name}_outlier_labels.pkl")
        #self.df_ce_outliers.to_pickle(f"results/df_uni_{model_name}_ce_outliers.pkl")

def update_outlier_scores(new_df, analysis_name):

    # Define file path for results
    path = os.path.join(analysis_name, 'results/outlier_scores.pkl')

    # Create directories if they do not exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    if not os.path.exists(path):
        with open(path, 'wb') as f:
            pickle.dump(new_df, f)
    
    else:
        with open(path, 'rb') as f:
            existing_df = pickle.load(f)

        # Add the new data to the existing DataFrame    
        for column in new_df:
            existing_df.loc[new_df.index, column] = new_df[column]

        # Save the updated DataFrame
        with open(path, 'wb') as f:
            pickle.dump(existing_df, f)

def ce_scatter_2x2_grid(indf_list):
    index = indf_list[0]
    df = indf_list[1]
    if df.shape[1] != 4:
        raise ValueError("DataFrame must have exactly 4 columns")
    
    # Creating the 2x2 grid of plots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

    # Flatten the axs array for easier iteration
    axs = axs.flatten()

    # Plotting each column in its own subplot as a scatter plot
    for i, column in enumerate(df.columns):
        axs[i].scatter(df.index, df[column], s=1)
        axs[i].set_ylabel(column)  # Set y-axis label to the column name
        axs[i].grid(True)

    # Set the overall title for the grid
    fig.suptitle(str(index), fontsize=16)
    
    # Remove x-axis labels
    for ax in axs:
        ax.set_xlabel('')  # No x-axis label
    print(index)
    plt.tight_layout()
    plt.show()

def gen_ce_scatter_2x2_grid(index_df_list):
    for indf_list in index_df_list:
        yield ce_scatter_2x2_grid(indf_list)

#Archive:

def SCSCEGenerator():
    data_instance = GetData()
    gen = dict_key_value_generator(data_instance.structure_dict)
    for serial, coil_serial, coil_element in gen:
        df = data_instance.get_cleaned_scaled_ce_df(serial, coil_serial, coil_element)
        df.name = f"{serial}_{coil_serial}_{coil_element}"
        df.serial = serial
        df.coil_serial = coil_serial
        df.coil_element = coil_element
        yield df

def window_generator(window_size=50):
    data_instance = GetData()
    gen = dict_key_value_generator(data_instance.structure_dict)
    for serial, coil_serial, coil_element in gen:
        df = data_instance.get_cleaned_scaled_ce_df(serial, coil_serial, coil_element)
        arr = df.values
        arr_windows = sliding_window_view(df.values, window_shape=(window_size, df.shape[1]))
        new_shape_arr_windows = arr_windows.reshape(-1, window_size, df.shape[1])
        sequence_count = new_shape_arr_windows.shape[0]
        for window_number in range(sequence_count):
            window = new_shape_arr_windows[window_number]
            df_window = pd.DataFrame(window, columns=df.columns)
            df_window.name = f"{serial}_{coil_serial}_{coil_element}_win{window_number}"
            df_window.serial = serial
            df_window.coil_serial = coil_serial
            df_window.coil_element = coil_element
            yield df_window

# For all rows in the dataframe with a variable value that is more than 3 (or some other specified number) 
# standard deviations away from the mean of that variable, remove the entire row from the dataframe.
# In the future, we want a class that maintains some of the intermiediate data objects as attributes.
def remove_stds(data: pd.DataFrame, 
                stds: int=3
               ) -> pd.DataFrame:
    df_kpis = data[['SSR', 'CSP', 'CNL', 'CSI']].copy()
    list_of_arrays = [] # list to store arrays of indexes to remove
    for column in df_kpis.columns:
        # get indexes of values that are stds standard deviations away from the mean of the column
        column_over_stdevs = indexes_of_stds_devs(df_kpis[column].values, stds)
        list_of_arrays.append(column_over_stdevs) # append the array of indexes to the list
    all_indexes_to_remove = np.concatenate(list_of_arrays) # concatenate all arrays of indexes to remove
    unique, counts = np.unique(all_indexes_to_remove, return_counts = True) # get sorted unique indexes and their counts
    df_kpis_cleaned = df_kpis.drop(unique).reset_index(drop=True) # drop rows containing anomalous data points and reset index

    enote = "Length of df_cleaned does not match original df_kpis minus anomalous data points" 
    assert len(df_kpis) - len(unique) == len(df_kpis_cleaned), enote # check that the length of the cleaned dataframe is correct

    dup = unique[counts > 1] # get indexes of rows that contain anomalous data points in more than one column (future attribute)
    
    return df_kpis_cleaned

def min_max_scale(df):

    df_kpis = df.copy()

    for column in df.columns:
        max = df[column].max()
        min = df[column].min()
        x_minus_min = df[column] - min
        df_kpis[column] = x_minus_min / (max - min)
    
    return df_kpis
