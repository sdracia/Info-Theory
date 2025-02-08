import numpy as np
import itertools
import time
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score as nmi
import matplotlib.pyplot as plt
import os



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

def likelihood_comp(T, K, pi, gamma, r, y):

    likelihood = 0
    # all possible sequences z
    all_sequences = itertools.product(range(K), repeat=T)

    for z in all_sequences: 

        p_1 = 0
        p_2 = pi[z[0]]

        for t in range(T-1):
            p_1 *= r[y[t], z[t]]
            p_2 *= gamma[z[t], z[t+1]]

        p_1 *= r[y[T-1], z[T-1]]

        likelihood += (p_1*p_2)

    return likelihood

def forward(y, pi, gamma, r, T, K, epsilon=1e-10):
    
    alpha = np.zeros((T, K))   # alpha has T elements of dimension K each

    alpha[0] = pi * r[y[0]]   # r[y[0]] is the first row of r: r_y1,k  
    alpha[0] /= np.sum(alpha[0])

    for t in range(1, T):   
        for k in range(K):
            alpha[t, k] = (r[y[t], k] * np.sum(alpha[t - 1, :] * gamma[:, k])) + epsilon     #gamma[k] is the k'th row of gamma
        alpha[t] /= np.sum(alpha[t])

    """
    print(alpha)
    print(pi)
    print(r[y[0]])
    """
    
    return alpha

def backward(y, gamma, r, T, K, epsilon=1e-10):
    
    beta = np.zeros((T, K))
    beta[-1] = [1]*K

    for t in range(T - 2, -1, -1):
        for k in range(K):
            beta[t, k] = (np.sum(beta[t + 1, :] * gamma[k, :] * r[y[t + 1], :])) + epsilon
        beta[t] /= np.sum(beta[t])  

    #print(beta)

    return beta

def e_step(y, z, pi, gamma, r, T, K):
    
    alpha = forward(y, pi, gamma, r, T, K)
    beta = backward(y, gamma, r, T, K)
    
    """
    print("alpha = ", alpha)
    print("beta = ", beta)
    """
    
    # it must have T-1 elements because it's the P(z_t = k and z_t+1 = l | ...)
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        for k in range(K):
            for l in range(K):
                xi[t, k, l] = alpha[t, k] * gamma[l, k] * r[y[t + 1], l] * beta[t + 1, l]   #this formula might be wrong in the notes
        #renormalization
        xi[t,:,:] /= np.sum(xi[t,:,:])

    zeta = np.zeros((T, K))

    """
    for t in range(T):
        for k in range(K):
            zeta[t,k]=np.sum(xi[t, k, :])

        zeta[t,:] /= np.sum(zeta[t,:])
    """
    for t in range(T):
        for k in range(K):
            zeta[t,k] = alpha[t, k] * beta[t, k]

        zeta[t,:] /= np.sum(zeta[t,:])
    
    # Assign the state at timestep t with the argmax of the one point posterior
    
    z_pred=np.argmax(zeta,axis=1)
    nmi_value=nmi(labels_true=z,labels_pred=z_pred)
    perc_corr_class=np.sum(z_pred==z)/len(z)*100
    
    """
    print("zeta = ", zeta)
    print("xi = ", xi)
    """
    """
    print("z pred = ",z_pred)  
    print("z real = ",z)
    """
    print(f"Mutual Info between z and z pred = {nmi_value}")  
    print(f"Percentage of correctly classified timesteps = {perc_corr_class}% ")

    return zeta, xi, nmi_value, perc_corr_class


def m_step(y, zeta, xi, N, K):
    
    T = len(y)
    # update of pi
    pi = np.zeros(K)
    pi = zeta[0,:]    
    #print(pi)

    # update of gamma
    gamma = np.zeros((K, K))

    for l in range(K):  # Iterate over columns
        for k in range(K):
            gamma[k, l] = np.sum(xi[:, k, l])  # Keep numerator as-is

    for i in range(gamma.shape[1]):
        gamma[:, i] /= np.sum(gamma[:, i])

    # update of r 
    r = np.zeros((N + 1, K))

    # Accumulate contributions to r[i, k]
    for i in range(N + 1):
        for k in range(K):
            for t in range(T):
                if y[t] == i:
                    # Sum over time steps where y[t] == i, weighted by zeta[t, k]

                    """
                    print("i = ", i)
                    print("k = ", k)
                    print("t = ", t)
                    print("zeta = ", zeta[t, k])
                    """

                    r[i, k] += zeta[t, k]

    # Normalize each column of r (ensure the sum of each column equals 1)
    r /= np.sum(r, axis=0, keepdims=True)
    
    return pi, gamma, r

def em_algorithm(y, z, pi, gamma, r, N, K, max_iter=100, tol=1e-6):
    
    T = len(y)
    pi_est, gamma_est, r_est = initialize_parameters(N, K)
    
    # Normalized mutual information and percentage of correctly classified states across iterations
    # Actually the second one has little significance, if K=2 the model might classify them the other way around
    # and reaching 0% accuracy is just as good as reaching 100%, when K>2 its much harder to say, the interesting one is NMI
    
    running_nmi=[]
    running_perc_corr_class=[]

    """
    pi = np.full(K, 1 / K)  # Uniform initial state distribution
    gamma = np.full((K, K), 1 / K)  # Uniform transition probabilities
    r = np.full((N + 1, K), 1 / (N + 1)) 
    """

    print("##########################")
    print("Starting values for the estimation of the parameters")
    print("Initialized Pi = ", pi_est)
    print("Initialized Gamma = ", gamma_est)
    print("Initialized R = ", r_est)
    
    # Calculate and print the difference between the initialized matrices (random) and the real ones
    
    init_delta_pi = np.linalg.norm(pi_est - pi)
    init_delta_gamma = np.linalg.norm(gamma_est - gamma, ord='fro')
    init_delta_r = np.linalg.norm(r_est - r, ord='fro')
    
    # print(f"Dist between starting pi and real = {init_delta_pi}")
    # print(f"Dist between starting gamma and real = {init_delta_gamma}")
    # print(f"Dist between starting r and real = {init_delta_r}")
    

    for iteration in tqdm(range(max_iter),desc=f"Running EM algorithm with {max_iter} iterations"):
        
        #print("Running iteration: ",iteration)
        # E-step
        zeta, xi, nmi, perc_corr_class = e_step(y, z, pi_est, gamma_est, r_est, T, K)
        
        # Keep track of nmi and perc
        
        running_nmi.append(nmi)
        running_perc_corr_class.append(perc_corr_class)

        #print("zeta = ", zeta)
        #print("xi = ", xi)

        # M-step
        pi_updated, gamma_updated, r_updated = m_step(y, zeta, xi, N, K)

        # print("#################################")
        # print("pi_upd = ", pi_updated)
        # print("gamma_upd = ", gamma_updated)
        # print("r_upd = ", r_updated)
        
    
        # i compute the delta using the relative distance, and using the Frobenius norm
        # delta_pi = np.linalg.norm(pi_updated - pi, ord='fro') / (np.linalg.norm(pi, ord='fro') + 1e-10)
        # This might work but for sure the tol must be higher
        
        delta_pi = np.linalg.norm(pi_updated - pi_est)
        delta_gamma = np.linalg.norm(gamma_updated - gamma_est, ord='fro')
        delta_r = np.linalg.norm(r_updated - r_est, ord='fro')
        

        # if delta_pi < tol and delta_gamma < tol and delta_r < tol:
        if delta_gamma < tol and delta_r < tol and delta_pi < tol:
            print(f"Converged at iteration {iteration}")
            break

        pi_est = pi_updated
        gamma_est = gamma_updated
        r_est = r_updated

    return pi_updated, gamma_updated, r_updated, running_nmi, running_perc_corr_class, init_delta_gamma, init_delta_r, init_delta_pi



def create_run_folder():
    
    i = 1
    while os.path.exists(f'synthetic_runs_hmm/run_{i}'):
        i += 1
    

    run_folder = f'synthetic_runs_hmm/run_{i}'
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





def run_simulation(N, K, T, seed, num_rep, pi, gamma, r, type_run):

    run_folder = create_run_folder()

    # To make it simpler to set up one can just set 1 and 0 based on which we want to be active, and then they get normalized
    col_sums = r.sum(axis=0)
    col_sums[col_sums == 0] = 1
    r = r / col_sums

    #output: generation of the observed and hidden data
    y, z = generate_hmm_data(T, N, K, pi, gamma, r)

    print("Data is generated according to an HMM with the following parameters:")

    print("Pi:", pi)
    print("Gamma:", gamma)
    print("R:", r)

    print(f"{T} data points are generated, which are:")

    print("Generated firings:", y)
    print("Generated states:", z)

    # likelihood = likelihood_comp(T, K, pi, gamma, r, y)
    # print(likelihood)

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

        pi_est, gamma_est, r_est, running_nmi, running_perc_corr_class, delta_gamma, delta_r, delta_pi = em_algorithm(y, z, pi, gamma, r, N, K, max_iter=25)

        tot_nmi.append(running_nmi)
        tot_perc.append(running_perc_corr_class)
        tot_delta_gamma.append(delta_gamma)
        tot_delta_r.append(delta_r)

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

    
    return pis_est, gammas_est, rs_est
