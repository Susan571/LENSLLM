import numpy as np
from scipy.optimize import minimize
import random
import warnings

PARAMS_NUM = 4
warnings.filterwarnings("ignore", category=RuntimeWarning)

def ntk_loss(f0_X, y, ntk_matrix, t, eta):
    """
    Compute NTK-based test loss according to equation (9)
    """
    exp_term = np.exp(-eta * ntk_matrix * t)
    diff = f0_X - y
    return np.linalg.norm(np.dot(exp_term, diff), ord=2)**2

def lens_llm_transform(log_x, B, beta, E, f0_X, y, ntk_matrix, t, eta):
    """
    Transform using LensLLM model based on equation (10)
    Input x is in log scale, representing dataset size D
    """
    F_term = ntk_loss(f0_X, y, ntk_matrix, t, eta)
    D = np.exp(log_x)  # Convert log scale back to original scale
    denominator = F_term + D**beta
    return np.log(B/denominator + E)

def rec_llm_transform(log_x, A, B, beta, E):
    """
    Transform the input x to the prediction y using rectified law based on four parameters A, B, beta, and E.
    Note that input x is in log scale.
    """
    expnum = beta * log_x + B
    return np.log((A-abs(E)) /(1 + np.exp(expnum)) + abs(E))

def fit_lens_llm(log_x, log_y, f0_X, y, ntk_matrix, case="phase_fitting", t_fixed=None):
    def optimize_phase_fitting(params):
        B, beta, E = params
        eta = 0.01
        
        def loss(params):
            pred = lens_llm_transform(log_x, B, beta, E, f0_X, y, ntk_matrix, t_fixed, eta)
            return np.square(pred - log_y).mean()
        
        result = minimize(loss, params, bounds=((0, None), (0, None), (0, None)))
        return result.x, result.fun

    def optimize_prediction(params):
        B, beta, E, t = params
        eta = 0.01
        
        def loss(params):
            pred = lens_llm_transform(log_x, B, beta, E, f0_X, y, ntk_matrix, t, eta)
            return np.square(pred - log_y).mean()
        
        result = minimize(loss, params, bounds=((0, None), (0, None), (0, None), (0, None)))
        return result.x, result.fun

    init_params = np.array([1.0, 1.0, 0.1])

    if case == "phase_fitting":
        # Case 1: Phase-fitting with fixed t
        if t_fixed is None:
            raise ValueError("t_fixed must be provided for phase_fitting case")
        return optimize_phase_fitting(init_params)
    else:
        # Case 2: Test loss prediction with all parameters
        init_params_with_t = np.append(init_params, 500)  # Initial t value
        return optimize_prediction(init_params_with_t)

def fit_rec_llm(log_x, log_y):
    """
    Fit rectified law to the data.
    Note that both x and y are in log scale.
    """
    def surrogate_optimize(params):
        params = np.concatenate([init_params, np.ones_like(log_x)])

        def first_loss(surr_params):
            Z = surr_params[PARAMS_NUM:]
            Z = Z**2 # Z = exp(beta*x + B) > 0
            A, B, beta, E = surr_params[:PARAMS_NUM]

            # MSE loss 
            Z = Z
            y_loss = np.square(np.log(A + np.abs(E) * Z) - np.log(1 + Z) - log_y).mean() if Z.shape[0] > 0 else 0
            x_loss = np.square(np.log(Z) - beta * log_x - B).mean() if Z.shape[0] > 0 else 0

            return x_loss + y_loss

        result = minimize(first_loss, params)
        params = result.x
        
        A, B, beta, E = params[:PARAMS_NUM]
        def second_loss(intercepts):
            _A, _E = intercepts
            log_y_pred = rec_llm_transform(log_x, _A, B, beta, _E)

            y_loss = np.square(log_y_pred - log_y).mean()

            return y_loss

        result = minimize(second_loss, [A, E])
        A, E = result.x
        params = [A, B, beta, E]
        return params, result.fun

    best_loss = float('inf')
    best_params = None
    u, l =np.max(log_y), np.min(log_y)

    for i in range(3):
        A, E = np.exp(u), np.exp((u + l) / 2)
        init_params = np.array([A, 0.0, 1.0, E])
        
        params, lss = surrogate_optimize(init_params)

        if lss < best_loss and params[0] > params[3]:
            best_loss = lss
            best_params = params

    return best_params, best_loss