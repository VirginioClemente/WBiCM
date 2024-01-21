from WBiCM import weighted_bipartite


# Importing the weighted bipartite network matrix
weighted_matrix = np.loadtxt('bicm_matW.csv', delimiter=',',dtype=str)
weighted_matrix =  weighted_matrix[1:, 1:].astype(float)


# Defining the binary part
topologia = weighted_matrix.astype(bool).astype(int)

# Calculating the statistics of interest
s_row = weighted_matrix.sum(1)
s_col = weighted_matrix.sum(0)

k_row = weighted_matrix.astype(bool).astype(int).sum(1)
k_col = weighted_matrix.astype(bool).astype(int).sum(0)


# Initializing the class:

# 1 If the topology is not known.
bipartito = weighted_bipartite(s_row,s_col,k_rows=k_row,k_cols=k_col)

# 2 If the topology is known
#bipartito = weighted_bipartite(s_row,s_col,topology=topologia)

# Finding the solution to the problem
bipartito.solve()

beta = bipartito.beta
alfa = bipartito.alfa

# Creating the folder 'samples_WBiCM' with 50 samples of the network inside
bipartito.sampler(50)
