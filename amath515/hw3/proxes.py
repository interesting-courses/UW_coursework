# this file contains collections of proxes we learned in the class
import numpy as np
from scipy.optimize import bisect

# =============================================================================
# TODO Complete the following prox for simplex
# =============================================================================

# Prox of capped simplex
# -----------------------------------------------------------------------------
def prox_csimplex(z, k):
    """
    Prox of capped simplex
        argmin_x 1/2||x - z||^2 s.t. x in k-capped-simplex.

    input
    -----
    z : arraylike
        reference point
    k : int
        positive number between 0 and z.size, denote simplex cap

    output
    ------
    x : arraylike
        projection of z onto the k-capped simplex
    """
    # safe guard for k
    assert 0<=k<=z.size, 'k: k must be between 0 and dimension of the input.'

    # TODO do the computation here
    # Hint: 1. construct the scalar dual object and use `bisect` to solve it.
    #		2. obtain primal variable from optimal dual solution and return it.
    #
    
    # find root of f'(la) using bisection method
    la_root = bisect(lambda la: -k + np.sum(np.clip(z-la,0,1)),np.min(z)-1,np.max(z))
    
    # return optimal x given root of f'(la)
    return np.clip(z-la_root,0,1)

