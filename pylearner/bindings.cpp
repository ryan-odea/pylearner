#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <stdexcept>
#include <algorithm>

namespace py = pybind11;

// Forward declarations
py::dict learner_cpp(const Eigen::MatrixXd &Y_source,
                     const Eigen::MatrixXd &Y_target,
                     int r, double lambda1, double lambda2,
                     double step_size, int max_iter,
                     double threshold, double max_value);

py::dict cv_learner_cpp(const Eigen::MatrixXd &Y_source,
                        const Eigen::MatrixXd &Y_target,
                        const std::vector<double> &lambda1_all,
                        const std::vector<double> &lambda2_all,
                        double step_size, int n_folds,
                        int max_iter, double threshold,
                        int n_cores, int r, double max_value,
                        unsigned int seed);

int omp_max_threads();

PYBIND11_MODULE(learner_ext, m) {
    m.doc() = "LEARNER function bindings (C++ backend via pybind11)";

    // Expose learner function
    m.def(
        "learner",
        [](const Eigen::MatrixXd &Y_source,
           const Eigen::MatrixXd &Y_target,
           py::object r,
           double lambda1,
           double lambda2,
           double step_size,
           py::dict control) -> py::dict
        {
            if (Y_source.rows() != Y_target.rows() ||
                Y_source.cols() != Y_target.cols()) {
                throw std::invalid_argument("Y_source and Y_target must have the same dimensions.");
            }

            for (int i = 0; i < Y_source.rows(); i++) {
                for (int j = 0; j < Y_source.cols(); j++) {
                    if (std::isnan(Y_source(i, j))) {
                        throw std::invalid_argument("Y_source cannot have NA values.");
                    }
                }
            }

            // Determine rank: if r is None, use adaptive thresholding in (local) screenot.
            // when screeNOT is working as a python package, perhaps switch there.
            int r_val;
            if (r.is_none()) {
                py::module screenot = py::module::import("pylearner.screenot");
                double max_rank = std::min(Y_source.rows(), Y_source.cols()) / 3.0;
                py::object result_obj = screenot.attr("adaptiveHardThresholding")(Y_source, max_rank);
                py::tuple result_tuple = result_obj.cast<py::tuple>();
                r_val = std::max(result_tuple[2].cast<int>(), 1);
            } else {
                r_val = r.cast<int>();
            }

            int max_iter  = control.contains("max_iter") ? control["max_iter"].cast<int>() : 100;
            double threshold = control.contains("threshold") ? control["threshold"].cast<double>() : 0.001;
            double max_value = control.contains("max_value") ? control["max_value"].cast<double>() : 10.0;

            return learner_cpp(Y_source, Y_target, r_val,
                               lambda1, lambda2, step_size,
                               max_iter, threshold, max_value);
        },
        R"pbdoc(
Latent space-based transfer learning

This function applies the LEARNER method to leverage source population data 
to improve low-rank estimation of the target population matrix.

Parameters
----------
Y_source : numpy.ndarray
    Matrix containing the source population data.
Y_target : numpy.ndarray
    Matrix containing the target population data.
r : int or None
    Rank specification. If None, adaptive hard thresholding from screenot.py is used.
lambda1 : float
    Regularization parameter lambda1.
lambda2 : float
    Regularization parameter lambda2.
step_size : float
    Step size for the optimization algorithm.
control : dict, optional
    Dictionary controlling optimization criteria:
      - "max_iter": maximum number of iterations (default 100)
      - "threshold": convergence threshold (default 0.001)
      - "max_value": maximum allowed value for the objective function (default 10).

Returns
-------
dict
    A dictionary containing:
      - "learner_estimate": numpy.ndarray, the LEARNER estimate.
      - "objective_values": list of float, the objective function values.
      - "convergence_criterion": int, the stopping condition indicator.
      - "r": int, the rank used.
)pbdoc",
        py::arg("Y_source"),
        py::arg("Y_target"),
        py::arg("r") = py::none(),
        py::arg("lambda1"),
        py::arg("lambda2"),
        py::arg("step_size"),
        py::arg("control") = py::dict()
    );

    // Expose cv_learner function
    m.def(
        "cv_learner",
        [](const Eigen::MatrixXd &Y_source,
        const Eigen::MatrixXd &Y_target,
        py::object r,
        py::list lambda1_all,
        py::list lambda2_all,
        double step_size,
        int n_folds,
        int n_cores,
        py::dict control,
        unsigned int seed) -> py::dict
        {
            int r_val;
            if (r.is_none()) {
                py::module screenot = py::module::import("pylearner.screenot");
                double max_rank = std::min(Y_source.rows(), Y_source.cols()) / 3.0;
                py::object result_obj = screenot.attr("adaptiveHardThresholding")(Y_source, max_rank);
                py::tuple result_tuple = result_obj.cast<py::tuple>();
                r_val = std::max(result_tuple[2].cast<int>(), 1);
            } else {
                r_val = r.cast<int>();
            }
            int max_iter  = control.contains("max_iter") ? control["max_iter"].cast<int>() : 100;
            double threshold = control.contains("threshold") ? control["threshold"].cast<double>() : 0.001;
            double max_value = control.contains("max_value") ? control["max_value"].cast<double>() : 10.0;

            std::vector<double> vec_lambda1 = lambda1_all.cast<std::vector<double>>();
            std::vector<double> vec_lambda2 = lambda2_all.cast<std::vector<double>>();

            return cv_learner_cpp(Y_source, Y_target,
                                vec_lambda1, vec_lambda2,
                                step_size,
                                n_folds,
                                max_iter,
                                threshold,
                                n_cores,
                                r_val,
                                max_value,
                                seed);
        },
        R"pbdoc(
    Cross-validation for LEARNER

    Performs k-fold cross-validation to select optimal nuisance parameters
    (lambda1, lambda2) for the LEARNER method.

    Parameters
    ----------
    Y_source : numpy.ndarray
        Matrix with source data.
    Y_target : numpy.ndarray
        Matrix with target data.
    r : int or None
        Rank specification. If None, adaptive hard thresholding is used.
    lambda1_all : list of float
        Candidate values for lambda1.
    lambda2_all : list of float
        Candidate values for lambda2.
    step_size : float
        Step size for the optimization algorithm.
    n_folds : int
        Number of folds.
    n_cores : int
        Number of cores for parallel computation.
    control : dict, optional
        Dictionary controlling optimization parameters.
    seed : int, optional
        Seed for random number generation (default 1636).

    Returns
    -------
    dict
        A dictionary containing:
        - "lambda_1_min": candidate lambda1 with the smallest MSE.
        - "lambda_2_min": candidate lambda2 with the smallest MSE.
        - "mse_all": numpy.ndarray of MSE values.
        - "r": the rank used.
    )pbdoc",
        py::arg("Y_source"),
        py::arg("Y_target"),
        py::arg("r") = py::none(),
        py::arg("lambda1_all"),
        py::arg("lambda2_all"),
        py::arg("step_size"),
        py::arg("n_folds") = 4,
        py::arg("n_cores") = 1,
        py::arg("control") = py::dict(),
        py::arg("seed") = 1636 
    );


    // Expose omp_max_threads function
    m.def(
        "omp_max_threads",
        &omp_max_threads,
        R"pbdoc(
Return the maximum number of OpenMP threads available.
)pbdoc"
    );
}
