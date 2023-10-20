from Weighted_Bipartite import weighted_bipartite


# Importo la matrice della rete bipartita pesata
weighted_matrix = np.loadtxt('bicm_matW.csv', delimiter=',',dtype=str)
weighted_matrix =  weighted_matrix[1:, 1:].astype(float)


# Definisco la parte binaria
topologia = weighted_matrix.astype(bool).astype(int)

# Calcolo le statistiche che mi interessano
s_row = weighted_matrix.sum(1)
s_col = weighted_matrix.sum(0)

k_row = weighted_matrix.astype(bool).astype(int).sum(1)
k_col = weighted_matrix.astype(bool).astype(int).sum(0)


# Inizializzo la classe:

# 1 Se la topologia non è nota.
bipartito = weighted_bipartite(s_row,s_col,k_rows=k_row,k_cols=k_col)

# 2 Se la topologia è nota
#bipartito = weighted_bipartite(s_row,s_col,topology=topologia)

# Trovo la soluzione al problema
bipartito.solve()

beta = bipartito.beta
alfa = bipartito.alfa

# Creo la cartella "samples_WBiCM" con 50 campioni della rete al suo interno
bipartito.sampler(50)