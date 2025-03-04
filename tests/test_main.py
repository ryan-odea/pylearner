import os
import numpy as np
import pytest
from numpy.testing import assert_allclose
from pylearner import learner, cv_learner, dlearner

def load_dataset(filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.abspath(os.path.join(current_dir, "..", "datasets"))
    file_path = os.path.join(datasets_dir, filename)
    data = np.load(file_path)
    return {key: data[key] for key in data.files}

@pytest.fixture
def dat_highsim():
    return load_dataset("dat_highsim.npz")

@pytest.fixture
def dat_modsim():
    return load_dataset("dat_modsim.npz")

def test_cv_learner_sequential_no_error(dat_highsim):
    try:
        result = cv_learner(
            Y_source=dat_highsim["Y_source"],
            Y_target=dat_highsim["Y_target"],
            lambda1_all=[1, 10],
            lambda2_all=[1, 10],
            step_size=0.003
        )
    except Exception as e:
        pytest.fail(f"cv_learner (sequential) failed with error: {e}")

def test_cv_learner_parallel_no_error(dat_highsim):
    try:
        result = cv_learner(
            Y_source=dat_highsim["Y_source"],
            Y_target=dat_highsim["Y_target"],
            lambda1_all=[1, 10],
            lambda2_all=[1, 10],
            step_size=0.003,
            n_cores=2
        )
    except Exception as e:
        pytest.fail(f"cv_learner (parallel) failed with error: {e}")

def test_cv_learner_output(dat_highsim):
    expected_cv = np.array([
        [5497.094679, 5308.327932],
        [5344.612   , 5235.475182],
        [5019.494013, 5022.248501]
        ])
    
    result = cv_learner(
        Y_source=dat_highsim["Y_source"],
        Y_target=dat_highsim["Y_target"],
        lambda1_all=[1, 10, 100],
        lambda2_all=[1, 10],
        step_size=0.003,
        n_cores = 1,
        seed = 12345
    )
    mse_result = result["mse_all"]
    assert_allclose(mse_result, expected_cv, atol=1e-3)

def test_learner_no_error(dat_highsim):
    try:
        result = learner(
            Y_source=dat_highsim["Y_source"],
            Y_target=dat_highsim["Y_target"],
            lambda1=1, lambda2=1,
            step_size=0.003
        )
    except Exception as e:
        pytest.fail(f"learner failed with error: {e}")

def test_learner_output(dat_highsim):
    expected_learner = np.array([
        [0.1578405, -1.54883716,  1.2205437,   0.03377010,  0.0767873],
        [0.1189457, -0.12793612, -0.9755801,   0.01372606, -0.3902793],
        [-0.4384613,  1.21518704, -1.0718671,   0.60388502, -0.4104101],
        [1.4426139, -3.13260948,  0.9175863,  -1.79041482,  0.5015381],
        [0.5602868, -0.05068081, -1.3878535,  -0.78502749, -0.1315952]
    ])
    result = learner(
        Y_source=dat_highsim["Y_source"],
        Y_target=dat_highsim["Y_target"],
        lambda1=1, lambda2=1,
        step_size=0.003
    )
    learner_est = result["learner_estimate"]
    assert_allclose(learner_est[:5, :5], expected_learner, atol=1e-5)

def test_dlearner_no_error(dat_highsim):
    try:
        result = dlearner(
            Y_source=dat_highsim["Y_source"],
            Y_target=dat_highsim["Y_target"]
        )
    except Exception as e:
        pytest.fail(f"dlearner failed with error: {e}")

def test_dlearner_output(dat_highsim):
    expected_dlearner = np.array([
        [0.0959171, -1.72143637,  1.15500149,  0.005478273,  0.3258085],
        [0.1077137, -0.32243480, -1.03557177, -0.108195636, -0.4093049],
        [-0.7237275,  0.86634922, -0.43528206,  1.105802194, -0.5961351],
        [1.6347696, -2.91985925,  0.04876288, -2.357504089,  0.8127403],
        [0.4757450,  0.05665543, -1.50373470, -0.744076401, -0.2876033]
    ])
    result = dlearner(
        Y_source=dat_highsim["Y_source"],
        Y_target=dat_highsim["Y_target"]
    )
    dlearner_est = result["dlearner_estimate"]
    assert_allclose(dlearner_est[:5, :5], expected_dlearner, atol=1e-5)
