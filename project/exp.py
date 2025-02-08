import numpy as np
import itertools
import time
from tqdm import tqdm

# [Previous initialize_parameters, backward, m_step functions remain the same]

def forward_log(y, pi, gamma, r, T, K, epsilon=1e-300):
    """Forward algorithm in log space"""
    log_alpha = np.zeros((T, K))
    
    # Initialize first timestep
    log_alpha[0] = np.log(pi + epsilon) + np.log(r[y[0]] + epsilon)
    
    # Forward pass
    for t in range(1, T):
        for k in range(K):
            # Compute in log space using logsumexp trick
            log_probs = log_alpha[t-1] + np.log(gamma[:, k] + epsilon)
            max_log_prob = np.max(log_probs)
            log_alpha[t, k] = max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob))) + np.log(r[y[t], k] + epsilon)
    
    return log_alpha

def compute_log_likelihood(log_alpha):
    """Compute log likelihood using the forward variables"""
    # Use logsumexp trick for the final sum
    final_timestep = log_alpha[-1]
    max_log = np.max(final_timestep)
    return max_log + np.log(np.sum(np.exp(final_timestep - max_log)))

def run_single_em(y, N, K, max_iter=100, tol=1e-6):
    """Run a single instance of EM algorithm with given initial parameters"""
    T = len(y)
    pi, gamma, r = initialize_parameters(N, K)
    prev_log_likelihood = -np.inf
    
    pbar = tqdm(range(max_iter), desc="EM iterations")
    for iteration in pbar:
        # E-step
        zeta, xi = e_step(y, pi, gamma, r, T, K)
        
        # M-step
        pi, gamma, r = m_step(y, zeta, xi, N, K)
        
        # Compute log-likelihood in log space
        log_alpha = forward_log(y, pi, gamma, r, T, K)
        current_log_likelihood = compute_log_likelihood(log_alpha)
        
        # Update progress bar
        pbar.set_postfix({'log_likelihood': f'{current_log_likelihood:.2f}'})
        
        # Check convergence
        if abs(current_log_likelihood - prev_log_likelihood) < tol:
            print(f"\nConverged at iteration {iteration} with log-likelihood {current_log_likelihood:.2f}")
            break
            
        prev_log_likelihood = current_log_likelihood
    
    return pi, gamma, r, current_log_likelihood

def em_algorithm(y, N, K, max_iter=100, tol=1e-6, n_restarts=5):
    """Run EM algorithm with multiple restarts"""
    T = len(y)
    best_log_likelihood = -np.inf
    best_params = None
    
    for restart in range(n_restarts):
        print(f"\nRestart {restart + 1}/{n_restarts}")
        
        # Run a single instance of EM
        pi, gamma, r, log_likelihood = run_single_em(y, N, K, max_iter, tol)
        
        # Print parameter values for debugging
        print("\nCurrent parameters:")
        print("Pi:", pi)
        print("Gamma column sums:", np.sum(gamma, axis=0))
        print("R column sums:", np.sum(r, axis=0))
        
        # Update best parameters if current likelihood is better
        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_params = (pi, gamma, r)
            print(f"New best log-likelihood: {best_log_likelihood:.2f}")
    
    print(f"\nBest log-likelihood across all restarts: {best_log_likelihood:.2f}")
    return best_params

# Usage example:
# N = 10  # neurons
# K = 3   # states
# T = 2000  # time steps
# y = ...  # your observed data
# pi_est, gamma_est, r_est = em_algorithm(y, N, K, max_iter=100, tol=1e-6, n_restarts=5)

def generate_hmm_data(T, N, K, pi, gamma, r):
    z = np.zeros(T, dtype=int)  
    y = np.zeros(T, dtype=int)  

    # Initialization t=0. I extract the value of K following the init prob
    z[0] = np.random.choice(K, p=pi)

    #data generation
    for t in range(T):
        if t > 0:
            # I take the column of gamma corresponding to z[t-1] and extract from this distribution
            z[t] = np.random.choice(K, p=gamma[:, z[t-1]])

        # I take the column of r corresponding to z[t] and extract from this prob distribution
        y[t] = np.random.choice(N + 1, p=r[:, z[t]])

    return y, z

# Example usage
N = 10  # neurons
K = 3   # states
T = 2000  # time steps

# Generate some sample data
pi_true, gamma_true, r_true = initialize_parameters(N, K, seed=1234)
y, z = generate_hmm_data(T, N, K, pi_true, gamma_true, r_true)

# Run the improved EM algorithm
pi_est, gamma_est, r_est = em_algorithm(y, N, K, max_iter=100, tol=1e-6, n_restarts=5)

print("Real pi:", pi_true)
print("Real gamma:", gamma_true)
print("Real r:", r_true)

print("Estimated pi:", pi_est)
print("Estimated gamma:", gamma_est)
print("Estimated r:", r_est)