########################################################################################
# IMPORT ZONE
########################################################################################

import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score as nmi 

########################################################################################
# DATA GENERATION
########################################################################################

def initialize_parameters(N, K, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Probability vector of length K
    pi = np.random.dirichlet(alpha=np.ones(K))    

    assert(np.allclose(sum(pi), 1, atol=1e-6) == 1)

    # Transition matrix (K x K), where each column sums to 1, because a column
    # represents the jump probabilities from a fixed initial state.
    gamma = np.random.dirichlet(alpha=np.ones(K), size=K)

    assert(np.allclose(np.sum(gamma, axis = 1), np.ones(K), atol=1e-6) == 1)

    # r matrix (N+1 x K), where each column sums to 1, because a column
    # represents the firing probability of a neuron in a fixed state.
    r = np.random.dirichlet(alpha=np.ones(N + 1), size=K).T

    assert(np.allclose(np.sum(r, axis = 0), np.ones(K), atol=1e-6) == 1)

    return pi, gamma, r

########################################################################################

def generate_hmm_data(T, N, K, pi, gamma, r):
    z = np.zeros(T, dtype=int)
    y = np.zeros(T, dtype=int)

    # Initialization t=0. I extract the value of K following the init prob
    z[0] = np.random.choice(K, p=pi)
    y[0] = np.random.choice(N + 1, p=r[:, z[0]])

    #data generation
    for t in range(1, T):
        # I take the column of gamma corresponding to z[t-1] and extract from this distribution
        z[t] = np.random.choice(K, p=gamma[z[t-1]])
        
        # I take the column of r corresponding to z[t] and extract from this prob distribution
        y[t] = np.random.choice(N + 1, p=r[:, z[t]])

    return y, z

########################################################################################
# SAMPLING FUNCTIONS
########################################################################################

def sample_pi(v):
    return np.random.dirichlet(v)

########################################################################################

def sample_gamma(u):
    K = len(u)
    gamma_matrix = np.zeros((K, K))
    
    # Here is normalized BY ROWS!!!
    for k in range(K):
        gamma_matrix[k] = np.random.dirichlet(u[k])
    return gamma_matrix

########################################################################################

def sample_r(w):
    K = len(w[0])
    N = len(w) - 1
    r_matrix = np.zeros((N + 1, K))

    # Here is normalized BY COLUMNS!!!
    for k in range(K):
        r_matrix[:, k] = np.random.dirichlet(w[:, k])
    return r_matrix

########################################################################################

def sample_z(y, z_prev, r, gamma):
    T = len(z_prev)
    K = len(gamma)
    z_new = np.zeros(T)
    eta = np.zeros((T, K))
    
    for k in range(K):
        eta[0, k] = r[y[0], k] * gamma[k, int(z_prev[1])] 
    
    eta[0, :] /= np.sum(eta[0, :])
    z_new[0] = np.random.choice(K, p=eta[0, :])
    
    for t in range(1, T-1):
        for k in range(K):
            m = int(z_new[t-1])
            l = int(z_prev[t+1])
            eta[t, k] = r[y[t], k] * gamma[m, k] * gamma[k, l]
        
        eta[t, :] /= np.sum(eta[t, :])
        
        z_new[t] = np.random.choice(K, p=eta[t, :])
    
    for k in range(K):
        eta[T-1, k] = r[y[T-1], k] * gamma[int(z_new[T-2]), k]
    
    eta[T-1, :] /= np.sum(eta[T-1, :])
    z_new[T-1] = np.random.choice(K, p=eta[T-1, :])
    
    return z_new

########################################################################################
# UPDATING FUNCTIONS
########################################################################################

def update_v(v, z):
    v[int(z[0])] += 1
    
    return v

########################################################################################

def update_u(u, z):
    T = len(z)
    
    for t in range(T-1):
        u[int(z[t]), int(z[t+1])] += 1
        
    return u

########################################################################################

def update_w(w, y, z):
    T = len(z)
    
    for t in range(T):
        w[int(y[t]), int(z[t])] += 1
        
    return w

########################################################################################
# GIBBS SAMPLING
########################################################################################

def gibbs_sampling(y, v, u, w, num_iterations, threshold=1e-6):
    pi = sample_pi(v)
    gamma = sample_gamma(u)
    r = sample_r(w)
    K = len(v)
    z0 = np.random.randint(0, K, K)
    z = sample_z(y, z0, r, gamma)
    
    pi_init = pi
    gamma_init = gamma
    r_init = r
    
    running_nmi = []
    running_perc_corr_class = []
    
    for iter in range(num_iterations):
        if iter != 0 and iter%100==0:
            print(f"Iteration {iter}...")
        
        v = update_v(v, z)
        u = update_u(u, z)
        w = update_w(w, y, z)
        
        pi_up = sample_pi(v)
        gamma_up = sample_gamma(u)
        r_up = sample_r(w)
        
        z_up = sample_z(y, z, r_up, gamma_up)
        running_nmi.append(nmi(labels_true = z, labels_pred = z_up))
        running_perc_corr_class.append(np.sum(z_up==z)/len(z)*100)
        
        delta_pi = np.linalg.norm(pi_up - pi)
        delta_gamma = np.linalg.norm(gamma_up - gamma, ord='nuc')
        delta_r = np.linalg.norm(r_up - r, ord='nuc')
        
        if delta_pi < threshold and delta_gamma < threshold and delta_r < threshold:
            print(f"Converged at iteration {iter}")
            break
        
        pi = pi_up
        gamma = gamma_up
        r = r_up
        z = z_up
        
    return pi_up, gamma_up, r_up, running_nmi, running_perc_corr_class, pi_init, gamma_init, r_init


########################################################################################
# RUNNING SIMULATIONS
########################################################################################

def create_folder():
    i = 1
    while os.path.exists(f'run_{i}'):
        i += 1

    run_folder = f'run_{i}'
    os.makedirs(run_folder)
    return run_folder
    
# Function to plot the data of the nmi against the training steps, coloured based on a distance metric
    
def plot_nmi_vs_iterations(run_folder, tot_nmi, tot_dist, title, colorbar_label):
    plt.figure(figsize=(10, 6))

    # Normalize the distance values for the color mapping
    
    norm = plt.Normalize(vmin=np.min(tot_dist), vmax=np.max(tot_dist))
    colormap = plt.cm.viridis

    # Create a ScalarMappable to use for the colorbar (doesn't work otherwise)
    
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Required for compatibility with colorbar

    # Loop through each run and plot it with color based on the required distance
    
    for i, (nmi_vals, dist) in enumerate(zip(tot_nmi, tot_dist)):
        plt.plot(range(len(nmi_vals)), nmi_vals, label=f"Run {i+1}", color=colormap(norm(dist)), linewidth=2)

    # Add colorbar explicitly associated with the correct figure
    
    cbar = plt.colorbar(sm, ax=plt.gca())  # Attach to current Axes
    cbar.set_label(colorbar_label)  # Label for the colorbar

    label_1 = title.split()[0]
    label_2 = colorbar_label.split()[0]

    plt.title(title)
    plt.xlabel("Iteration Number")
    plt.ylabel("NMI")
    plt.legend()  # Show legend to distinguish different runs, it's not really useful because who cares the order
    plt.savefig(f"{run_folder}/{label_1}_{label_2}.png")
    plt.close()





def run_simulation(T, N, K, seed, num_rep, pi, gamma, r, type_run):

    run_folder = create_folder()

    col_sums = r.sum(axis=0)
    col_sums[col_sums == 0] = 1
    r = r / col_sums

    y, z = generate_hmm_data(T, N, K, pi, gamma, r)

    print("Data is generated according to an HMM with the following parameters:")

    print("Pi:\n", pi)
    print("Gamma:\n", gamma)
    print("R:\n", r)

    print(f"{T} data points are generated, which are:")

    print("Generated firings:", y)
    print("Generated states:", z)
    
    # CHANGE Z
    
    
    # Initialize everything

    pis_est = []
    gammas_est = []
    rs_est = []

    tot_nmi=[]
    tot_perc=[]
    tot_delta_gamma=[]
    tot_delta_r=[]

    # Nrep is the number of attempts you want to do, the result depends on the random initialization
    # of the matrices so we need multiple attemps

    for i in range(num_rep):
        print(f"Rep {i}...")
        v = np.ones(K)
        u = np.ones((K, K))
        w = np.ones((N + 1, K))

        pi_est, gamma_est, r_est, running_nmi, running_perc_corr_class, pi_init, gamma_init, r_init = gibbs_sampling(y, v, u, w, num_iterations=50)

        init_delta_pi = np.linalg.norm(pi_init - pi)
        init_delta_gamma = np.linalg.norm(gamma_init - gamma, ord='fro')
        init_delta_r = np.linalg.norm(r_init - r, ord='fro')
    
        tot_nmi.append(running_nmi)
        tot_perc.append(running_perc_corr_class)
        tot_delta_gamma.append(init_delta_gamma)
        tot_delta_r.append(init_delta_r)

        print("##########################")
        print("Estimated values for the parameters")
        print("Estimated Pi = ", pi_est)
        print("Estimated Gamma = ", gamma_est)
        print("Estimated R = ", r_est)

        pis_est.append(pi_est)
        gammas_est.append(gamma_est)
        rs_est.append(r_est)

        """
        plt.plot(range(len(running_perc_corr_class)),running_perc_corr_class)
        plt.title("Percentage of correctly classified states Vs Interation number")
        plt.show()
        """
    """ 
    print(tot_delta_gamma)
    print(tot_delta_r)
    """

    tot_delta_gamma=np.array(tot_delta_gamma)
    tot_delta_r=np.array(tot_delta_r)
    tot_dist=tot_delta_gamma+tot_delta_r

    # Plot different cases with the corrected function
    plot_nmi_vs_iterations(run_folder, tot_nmi, tot_dist, "NMI Vs Iteration Number for Multiple Runs", "Total Distance Measure")
    plot_nmi_vs_iterations(run_folder, tot_nmi, tot_delta_gamma, "NMI Vs Iteration Number for Multiple Runs", "Gamma Distance Measure")
    plot_nmi_vs_iterations(run_folder, tot_nmi, tot_delta_r, "NMI Vs Iteration Number for Multiple Runs", "R Distance Measure")

    plot_nmi_vs_iterations(run_folder, tot_perc, tot_dist, "Percentage correct classified state Vs Iteration for Multiple Runs", "Total Distance Measure")
    plot_nmi_vs_iterations(run_folder, tot_perc, tot_delta_gamma, "Percentage correct classified state Vs Iteration for Multiple Runs", "Gamma Distance Measure")
    plot_nmi_vs_iterations(run_folder, tot_perc, tot_delta_r, "Percentage correct classified state Vs Iteration for Multiple Runs", "R Distance Measure")

    with open(os.path.join(run_folder, "run_parameters.txt"), "w") as f:
        f.write(type_run + '\n')
        f.write("=" * 50 + "\n\n")

        f.write(f"N (neurons): {N}\n")
        f.write(f"K (states): {K}\n")
        f.write(f"T (time steps): {T}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Number of repetitions: {num_rep}\n\n")

        f.write("Initialization of parameters for HMM:\n")
        f.write(f"Pi: {pi}\n")
        f.write(f"Gamma: {gamma}\n")
        f.write(f"R: {r}\n\n")

        f.write("Generated data:\n")
        f.write(f"Generated firings (y): {y[:100]}... (showing first 100 entries)\n")
        f.write(f"Generated states (z): {z[:100]}... (showing first 100 entries)\n\n")


        f.write("Estimated parameters:\n")
        f.write(f"Estimated Pi: {pis_est}\n")
        f.write(f"Estimated Gamma: {gammas_est}\n")
        f.write(f"Estimated R: {rs_est}\n\n")

        f.write(f"Total NMI: {tot_nmi}\n")
        f.write(f"Total Percentage: {tot_perc}\n")
        f.write(f"Total Delta Gamma: {tot_delta_gamma}\n")
        f.write(f"Total Delta R: {tot_delta_r}\n")


# define a new run_simulation() function for analyzing real data

def create_folder_real():
    i = 1
    while os.path.exists(f'run__real_{i}'):
        i += 1

    run_folder = f'run__real_{i}'
    os.makedirs(run_folder)
    return run_folder

def run_simulation_real(y, T, N, K, seed, num_rep, type_run):

    run_folder = create_folder_real()

    # Initialize everything

    pis_est = []
    gammas_est = []
    rs_est = []

    tot_nmi=[]
    tot_perc=[]
    tot_delta_gamma=[]
    tot_delta_r=[]

    # Nrep is the number of attempts you want to do, the result depends on the random initialization
    # of the matrices so we need multiple attemps

    for i in range(num_rep):
        print(f"Rep {i}...")
        v = np.ones(K)
        u = np.ones((K, K))
        w = np.ones((N + 1, K))

        pi_est, gamma_est, r_est, running_nmi, running_perc_corr_class, pi_init, gamma_init, r_init = gibbs_sampling(y, v, u, w, num_iterations=50)
        
        init_delta_pi = np.linalg.norm(pi_init - pi_est)
        init_delta_gamma = np.linalg.norm(gamma_init - gamma_est, ord='fro')
        init_delta_r = np.linalg.norm(r_init - r_est, ord='fro')
    
        tot_nmi.append(running_nmi)
        tot_perc.append(running_perc_corr_class)
        tot_delta_gamma.append(init_delta_gamma)
        tot_delta_r.append(init_delta_r)

        print("##########################")
        print("Estimated values for the parameters")
        print("Estimated Pi = ", pi_est)
        print("Estimated Gamma = ", gamma_est)
        print("Estimated R = ", r_est)

        pis_est.append(pi_est)
        gammas_est.append(gamma_est)
        rs_est.append(r_est)

        """
        plt.plot(range(len(running_perc_corr_class)),running_perc_corr_class)
        plt.title("Percentage of correctly classified states Vs Interation number")
        plt.show()
        """
    """ 
    print(tot_delta_gamma)
    print(tot_delta_r)
    """

    tot_delta_gamma=np.array(tot_delta_gamma)
    tot_delta_r=np.array(tot_delta_r)
    tot_dist=tot_delta_gamma+tot_delta_r

    # Plot different cases with the corrected function
    plot_nmi_vs_iterations(run_folder, tot_nmi, tot_dist, "NMI Vs Iteration Number for Multiple Runs", "Total Distance Measure")
    plot_nmi_vs_iterations(run_folder, tot_nmi, tot_delta_gamma, "NMI Vs Iteration Number for Multiple Runs", "Gamma Distance Measure")
    plot_nmi_vs_iterations(run_folder, tot_nmi, tot_delta_r, "NMI Vs Iteration Number for Multiple Runs", "R Distance Measure")

    plot_nmi_vs_iterations(run_folder, tot_perc, tot_dist, "Percentage correct classified state Vs Iteration for Multiple Runs", "Total Distance Measure")
    plot_nmi_vs_iterations(run_folder, tot_perc, tot_delta_gamma, "Percentage correct classified state Vs Iteration for Multiple Runs", "Gamma Distance Measure")
    plot_nmi_vs_iterations(run_folder, tot_perc, tot_delta_r, "Percentage correct classified state Vs Iteration for Multiple Runs", "R Distance Measure")

    with open(os.path.join(run_folder, "run_parameters.txt"), "w") as f:
        f.write(type_run + '\n')
        f.write("=" * 50 + "\n\n")

        f.write(f"N (neurons): {N}\n")
        f.write(f"K (states): {K}\n")
        f.write(f"T (time steps): {T}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Number of repetitions: {num_rep}\n\n")

        f.write("Estimated parameters:\n")
        f.write(f"Estimated Pi: {pis_est}\n")
        f.write(f"Estimated Gamma: {gammas_est}\n")
        f.write(f"Estimated R: {rs_est}\n\n")

        f.write(f"Total NMI: {tot_nmi}\n")
        f.write(f"Total Percentage: {tot_perc}\n")
        f.write(f"Total Delta Gamma: {tot_delta_gamma}\n")
        f.write(f"Total Delta R: {tot_delta_r}\n")