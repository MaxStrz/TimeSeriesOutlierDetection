from models.HBOSagg import HBOSAgg, HBOS_dynamic_bins
import numpy as np
import pytest

@pytest.fixture(scope='session')
def hbos_agg():
    return HBOSAgg()

def test_hbos_max_bins(my_data_fixture, hbos_agg):
    data = my_data_fixture.four_dim_pareto
    
    hbos_agg.fit(data)

    # Max value of array of bin counts should be less than sqrt of data length
    max_bins = max(hbos_agg._arr_bin_counts)
    assert max_bins < int(np.sqrt(len(data))), "Max bin count too high."

def test_hbosagg_instantiation(hbos_agg):
    """Test that hbosagg can be correctly default instantiated."""
    
    assert hbos_agg.instances == 100
    assert hbos_agg.dynamic_bins == True

def test_hbosagg_custom_instance():
    """Test that arguments correctly instantiate the class."""
    hbos_agg = HBOSAgg(instances=50, dynamic_bins=False)
    assert hbos_agg.instances == 50
    assert hbos_agg.dynamic_bins == False

def test_hbosagg_fit(hbos_agg):
    """Test that fit function fuctions."""
    X = np.random.rand(200, 5) # 200 samples, 5 features
    hbos_agg.fit(X)
    assert hasattr(hbos_agg, 'decision_scores_'), "missing decision_scores_"
    assert hbos_agg.results['decision_scores'].shape == (100, 200)
    assert hbos_agg.decision_scores_.shape == (200,)
    assert len(hbos_agg.arr_rank) == 200

def test_invalid_bin_count():
    """Test that an error is raised when an invalid bin count is provided."""
    
    with pytest.raises(ValueError):
        hbosdyn = HBOS_dynamic_bins(dynamic_bins=True, n_bins=1)
        hbosdyn.fit(np.random.rand(200, 5))