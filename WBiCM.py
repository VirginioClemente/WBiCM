import numpy as np
from numba import jit
from scipy.optimize import least_squares
import os
from bicm import *

class weighted_bipartite():

    def __init__(self, s_rows, s_cols, k_rows = False, k_cols = False, topology = False):


        self.topology = topology
        self.n_rows = len(s_rows)
        self.n_cols = len(s_cols)

        self.k_rows = k_rows
        self.k_cols = k_cols
        self.s_rows = s_rows
        self.s_cols = s_cols

        self.beta = np.zeros(self.n_rows)
        self.alfa = np.zeros(self.n_cols)

        self.P = self.__Compute_P()


    def __Compute_P(self):
        """ Defining the binary part.

            Depending on the availability or not of the topology
            it returns the topology itself or the probability of a link between
            each node in the network.

            In the last case, the probability is computed using the BiCM.

        """
        if type(self.topology) != bool:
            return self.topology

        elif (type(self.k_rows)!=bool and type(self.k_cols)!=bool):

            myGraph = BipartiteGraph()
            myGraph.set_degree_sequences((self.k_rows, self.k_cols))
            myGraph.solve_tool(print_error=False)

            return myGraph.get_bicm_matrix()

        else:
            print('To reconstruct the binary part, it is needed either the degree sequence or the topology.')
            return

    @jit
    def equations_to_solve_WBiCM(self,p, s_row, s_col,ave_mat):
        """WBiCM equations for numerical solver.

            Args:
                p: list of independent variables [beta alfa]
                s_row, s_col: numpy array of observed strengths on each dimension

            Returns:
                numpy array of observed stengths - expected stengths

        """
        n_rows = len(s_row)
        n_cols = len(s_col)

        p = np.array(p)

        beta = p[0:n_rows]
        alfa = p[n_rows:len(p)]


        # Expected degrees
        s_row_exp = np.zeros(n_rows)
        s_col_exp = np.zeros(n_cols)

        for i in range(n_rows):
            for j in range(n_cols):
                if i != j:
                    s_row_exp[i] += (ave_mat[i,j]) / (beta[i] + alfa[j])

        for j in range(n_cols):
            for i in range(n_rows):
                if i != j:
                    s_col_exp[j] += (ave_mat[i,j]) / (beta[i] +  alfa[j])


        f1 = s_row - s_row_exp
        f2 = s_col - s_col_exp


        return np.concatenate((f1, f2))


    def solve(self):
        """Solves the WBiCM numerically with least squares.
           The Weighted Bipartite Configuration Model is solved using the
           system of equations. The optimization is done using
           scipy.optimize.least_squares on the system of equations.

           Returns:
               list of independent variables [beta alfa]
        """

        beta_initial_values = np.random.rand(len(self.s_rows))
        alfa_initial_values = np.random.rand(len(self.s_cols))


        initial_values = np.concatenate((alfa_initial_values, beta_initial_values))
        x_solved = least_squares(fun=self.equations_to_solve_WBiCM,
                                 x0=initial_values,
                                 jac='3-point',
                                 args=(self.s_rows, self.s_cols,self.P),
                                 max_nfev=100000,
                                 loss='soft_l1',
                                 ftol=1e-10, xtol=1e-10, gtol=1e-10)
        print(x_solved.cost, x_solved.message)
        # Numerical solution checks
        assert x_solved.cost < 0.1, 'Numerical convergence problem: final cost function evaluation > 1'


        p = x_solved.x
        p = np.array(p)

        self.beta = p[0:len(self.s_rows)]
        self.alfa = p[len(self.s_rows):len(p)]

        return



    def sampler(self,n_samples,folder_name = 'samples_WBiCM'):

        try:
            os.mkdir(folder_name)
        except:
            print('\033[91m'+'The folder',folder_name, 'exists. Before running the sampler, change the folder name or delete the existing folder.'+'\033[0m')
            return
        for ii in range(n_samples):
            sample = np.zeros(np.shape(self.P))

            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    if np.random.rand() < self.P[i,j]:
                        sample[i,j] = np.random.exponential(1/abs(self.beta[i]+self.alfa[j]))

            np.savetxt(folder_name + '/sample_' + repr(ii) + '.csv', sample, fmt='%s', delimiter=',')
