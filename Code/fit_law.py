import numpy as np
from scipy.optimize import minimize
import random
import warnings

PARAMS_NUM = 4
warnings.filterwarnings("ignore", category=RuntimeWarning)

def our_law_transform(log_x, B, beta, E, F_term):
    """
    Transform the input x to the prediction y using our law based on four parameters A, B, beta, and E.
    Note that input x is in log scale.
    """
    denominator = F_term + np.exp(beta * log_x)
    return np.log(B/denominator + abs(E))

def rec_law_transform(log_x, A, B, beta, E):
    """
    Transform the input x to the prediction y using rectified law based on four parameters A, B, beta, and E.
    Note that input x is in log scale.
    """
    expnum = beta * log_x + B
    return np.log((A-abs(E)) /(1 + np.exp(expnum)) + abs(E))

def fit_our_law(log_x, log_y, F_term):
    """
    Fit rectified law to the data.
    Note that both x and y are in log scale.
    """
    def surrogate_optimize(params):
        params = np.concatenate([init_params, np.ones_like(log_x)])
        
        def first_loss(surr_params):
            Z = surr_params[PARAMS_NUM:]
            Z = Z * 2  # ensure positivity
            B, beta, E = surr_params[:PARAMS_NUM]
            
            denominator = F_term + Z
            y_loss = np.square(np.log(B/denominator + abs(E)) - log_y).mean()
            x_loss = np.square(np.log(Z) - beta * log_x).mean()
            
            return x_loss + y_loss
        
        result = minimize(first_loss, params)
        params = result.x
        
        # Second optimization stage for B and E
        B, beta, E = params[:PARAMS_NUM]
        def second_loss(intercepts):
            _B, _E = intercepts
            log_y_pred = lens_llm_transform(log_x, _B, beta, _E, F_term)
            return np.square(log_y_pred - log_y).mean()
            
        result = minimize(second_loss, [B, E])
        B, E = result.x
        params = [B, beta, E]
        return params, result.fun

    best_loss = float('inf')
    best_params = None
    u, l = np.max(log_y), np.min(log_y)

    for i in range(3):
        B, E = np.exp(u), np.exp((u + l) / 2)
        init_params = np.array([B, 1.0, E])
        
        params, lss = surrogate_optimize(init_params)
        
        if lss < best_loss:
            best_loss = lss
            best_params = params

    return best_params, best_loss

def fit_rec_law(log_x, log_y):
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

            return x_loss + y_loss # + 0.001 * np.abs(E)

        result = minimize(first_loss, params)
        params = result.x
        
        # Fix beta and D and re-tune A and E
        A, B, beta, E = params[:PARAMS_NUM]
        def second_loss(intercepts):
            _A, _E = intercepts
            log_y_pred = our_law_transform(log_x, _A, B, beta, _E)

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