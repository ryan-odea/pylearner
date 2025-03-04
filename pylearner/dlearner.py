import numpy as np
from .screenot import adaptiveHardThresholding

def dlearner(Y_source, Y_target, r=None):
    """
    Latent space-based transfer learning

    This function applies the Direct project LatEnt spAce-based tRaNsfer lEaRning (D-LEARNER) method 
    (McGrath et al. 2024) to leverage data from a source population to improve estimation of a low-rank matrix 
    in an underrepresented target population.

    Parameters
    ----------
    Y_source : numpy.ndarray or pandas.DataFrame
        Matrix containing the source population data.
    Y_target : numpy.ndarray or pandas.DataFrame
        Matrix containing the target population data.
    r : int, optional
        Rank specification for the knowledge graphs. If not provided, screenot is applied to the source 
        population to select the rank.

    Returns
    -------
    dict
        A dictionary with the following components:
          - dlearner_estimate: numpy.ndarray (or pandas.DataFrame), the D-LEARNER estimate of the target 
            population knowledge graph.
          - r: int, the rank value used.

    Details
    -------
    The data consists of a matrix in the target population, Y₀ ∈ ℝ^(p×q), and the source population, 
    Y₁ ∈ ℝ^(p×q). Let the truncated SVD of Yₖ (k = 0, 1) be given by Uₖ Λₖ Vₖᵀ. This method estimates 
    the target population knowledge graph, Θ₀, by:

        dlearner_estimate = U₁ U₁ᵀ Y₀ V₁ V₁ᵀ

    where U₁ and V₁ are computed from the SVD of Y_source.

    References
    ----------
    Donoho, D., Gavish, M. and Romanov, E. (2023). screenot: Exact MSE-optimal singular value thresholding in 
    correlated noise. The Annals of Statistics, 51(1), 122-148.

    Examples
    --------
    >>> import numpy as np
    >>> Y_source = np.random.rand(100, 50)
    >>> Y_target = np.random.rand(100, 50)
    >>> result = dlearner(Y_source, Y_target)
    >>> print(result["dlearner_estimate"])
    """
    if Y_source.shape != Y_target.shape:
        raise ValueError("Y_source and Y_target must have the same dimensions")
    
    if np.isnan(Y_source).any():
        raise ValueError("Y_source cannot have NA values.")
    
    if np.isnan(Y_target).any():
        raise ValueError("Y_target cannot have NA values.")

    p, q = Y_source.shape

    # If r is not provided, compute it using adaptiveHardThresholding.
    if r is None:
        max_rank = int(min(p, q) / 3)
        Xest, Topt, rank_val = adaptiveHardThresholding(Y_source, k=max_rank)
        r = max(int(rank_val), 1)

    U, s, Vh = np.linalg.svd(Y_source, full_matrices=False)
    U_r = U[:, :r]         # (p, r)
    V_r = Vh[:r, :].T       # (q, r)

    # Compute the D-LEARNER estimate: U_r @ (U_r.T @ Y_target @ V_r) @ V_r.T
    dlearner_estimate = U_r @ (U_r.T @ Y_target @ V_r) @ V_r.T

    try:
        import pandas as pd
        if isinstance(Y_source, pd.DataFrame):
            dlearner_estimate = pd.DataFrame(
                dlearner_estimate, index=Y_source.index, columns=Y_source.columns
            )
    except ImportError:
        pass

    return {"dlearner_estimate": dlearner_estimate, "r": r}
