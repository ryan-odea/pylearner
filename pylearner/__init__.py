import warnings
from pylearner import learner_ext

learner = learner_ext.learner
cv_learner = learner_ext.cv_learner
omp_max_threads = learner_ext.omp_max_threads

import importlib
dlearner = importlib.import_module("pylearner.dlearner").dlearner

if omp_max_threads() < 2:
    warning_message = (
        "*******\n"
        "This installation of learner has not detected OpenMP support\n"
        "It will still work but will not support multithreading via the `n_cores` argument\n"
        "If you plan to use multithreading, please ensure you have OpenMP installed\n"
        "*******"
    )
    warnings.warn(warning_message, UserWarning)

__all__ = [
    "learner",
    "cv_learner",
    "dlearner"
]

__version__ = "1.0.2"
