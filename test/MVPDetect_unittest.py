import unittest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from models.MVPDetect import MVPDetect
from models.MVPDetect import ParamLearnerLinReg
from test.test_data import my_data

class TestMVPDetect(unittest.TestCase):
    def test_fit_predict(self):
        # Data Generation
        data_generator = my_data()
        dict_partitions = data_generator.dict_partitions
        train_test_pairs = data_generator.train_test_idx

        # Model Initialization
        mvp_detector = MVPDetect(
            scorer=mean_squared_error,
            model=LinearRegression(),
            param_learner=ParamLearnerLinReg()
        )

        # Execution of fit_predict
        mvp_detector.fit_predict(
            dict_partitions=dict_partitions,
            train_test_pairs=train_test_pairs,
            iteration_limit=5  # Limit iterations for faster testing
        )

        # Check if collected_scores_ is populated
        self.assertGreater(len(mvp_detector.collected_scores_), 0, "No scores collected.")

        # Check if aggregated_scores_ is populated
        self.assertTrue(hasattr(mvp_detector, 'aggregated_scores_'), "aggregated_scores_ not found.")
        self.assertGreater(len(mvp_detector.aggregated_scores_), 0, "No aggregated scores.")

        # Check if df_ce_outlier_scores is a non-empty DataFrame
        self.assertTrue(hasattr(mvp_detector, 'df_ce_outlier_scores'), "df_ce_outlier_scores not found.")
        self.assertFalse(mvp_detector.df_ce_outlier_scores.empty, "df_ce_outlier_scores is empty.")

        # Optional: Validate the contents of df_ce_outlier_scores
        # For example, ensure all scores are non-negative
        self.assertTrue((mvp_detector.df_ce_outlier_scores >= 0).all().all(),
                        "Outlier scores contain negative values.")

        # # Print outputs for manual inspection (optional)
        # print("Aggregated Scores:")
        # print(mvp_detector.aggregated_scores_)
        # print("\nOutlier Scores DataFrame:")
        # print(mvp_detector.df_ce_outlier_scores)

# Run the test
if __name__ == '__main__':
    unittest.main()
