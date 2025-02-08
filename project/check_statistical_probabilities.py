
import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(N, K, seed=None):
    #Initialize parameters randomly

    if seed is not None:
        np.random.seed(seed)

    pi = np.random.dirichlet(alpha=np.ones(K))    

    check = np.allclose(sum(pi), 1, atol=1e-6)
    if check != 1:
        print("Inconsistent boundary conditions")


    # Transition matrix. it has K rows and K columns. In particular
    # the sum of probabilities from the same initial state should be 1
    gamma = np.random.dirichlet(alpha=np.ones(K), size=K).T

    sum_columns = np.sum(gamma, axis = 0)
    check = np.allclose(sum_columns, np.ones(K), atol=1e-6)
    if check != 1:
        print("Inconsistent boundary conditions")


    # I initialize the r matrix in this in order to satisfy the boundary condition (sum of 
    # columns must be 1 foreach column)
    r = np.random.dirichlet(alpha=np.ones(N + 1), size=K).T

    sum_columns = np.sum(r, axis = 0)
    check = np.allclose(sum_columns, np.ones(K), atol=1e-6)
    if check != 1:
        print("Inconsistent boundary conditions")

    return pi, gamma, r

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

T=100000
N=10
K=3

pi,gamma,r=initialize_parameters(N,K)
y,z=generate_hmm_data(T, N, K, pi, gamma, r)

# Firings

for i in range(K):

    mask=z==i

    y_i=y[mask]

    # Normalized histogram

    plt.hist(y_i, bins=range(N+2), label=f'Real counts of firings for state {str(i)}', alpha=0.5)

    # Compute expected counts

    expected_counts = [int(x) for x in r[:, i] * y_i.size]

    expected_outs = []  # Convert expected counts to a list of outputs

    for j, count in enumerate(expected_counts):
        expected_outs += [j] * count

    plt.hist(expected_outs, bins=range(N+2), label=f'Expected counts for state {str(i)}', alpha=0.5)
    plt.xlabel('Neuron')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()

# Transitions

transition_counts = np.zeros((K, K), dtype=int)

for t in range(T - 1):
    transition_counts[z[t], z[t + 1]] += 1

# Convert counts to probabilities and set the matrix as T

row_sums = transition_counts.sum(axis=1, keepdims=True)
empirical_gamma = (transition_counts / row_sums).T

# Compare with the expected gamma
print("Expected Transition Matrix (gamma):\n", gamma)
print("\nEmpirical Transition Matrix:\n", empirical_gamma)
print("\nDifference:\n", empirical_gamma - gamma)

# Plot transition matrices

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(gamma, cmap='Blues', vmin=0, vmax=1)
axes[0].set_title("Expected Transition Matrix")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(empirical_gamma, cmap='Oranges', vmin=0, vmax=1)
axes[1].set_title("Empirical Transition Matrix")
fig.colorbar(im2, ax=axes[1])

plt.show()