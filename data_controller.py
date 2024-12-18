import random
import math
import time
import numpy as np
import os.path

import matplotlib.pyplot as plt

from numpy.random import randn
from numpy.linalg import norm
import persim
from sklearn.metrics import pairwise_distances
import gudhi
from gph import ripser_parallel
from ripser import ripser

class DataController:
    def __init__(self, nPts, dim, trial, persistence, output_path=None, is_alpha=True):
        self.nPts = nPts  ## number of points sampled from each hypershpere
        self.trial = trial
        self.persistence = persistence
        self.dim = dim # extrinsic dimension
        self.radii = np.array([1])  ##radii of the hyperspheres
        self.output_path = output_path if output_path is not None else "."
        self.is_alpha = is_alpha

    def generate_nsphere(self):
        # Generate data. All spheres are centerd at the origin
        foo = np.random.randn(self.nPts, self.dim + 1)
        self.Data = self.radii * (foo / norm(foo, axis=1, keepdims=True))

        return self.Data

    def sampling_nball(self):
        # https://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
        # http://nojhan.free.fr/metah/
        def get_point_on_nsphere():
            R = np.random.uniform(low=0.0, high=1.0, size=self.dim+1)
            A = 2*np.pi * np.random.uniform(low=0.0, high=1.0, size=self.dim+1)
            Ri = R ** (1/(self.dim+1))
            suma = np.sum(A)
            p = (Ri * A) / suma

            return p
        sphere = [get_point_on_sphere() for _ in range(self.nPts)]
        self.Data = np.array(sphere)

        del sphere
        return self.Data

    def sampling_with_gram_schmidt(self):
        # get an orthornormal basis
        e = np.random.uniform(low=0, high=1, size=(self.nPts, self.dim))
        e_prime = self.orthonormalize(e)


    def orthonormalize(vectors):
        import torch
        """    
            Orthonormalizes the vectors using gram schmidt procedure.    

            Parameters:    
                vectors: torch tensor, size (dimension, n_vectors)    
                        they must be linearly independant    
            Returns:    
                orthonormalized_vectors: torch tensor, size (dimension, n_vectors)    
        """
        assert (vectors.size(1) <= vectors.size(0)), 'number of vectors must be smaller or equal to the dimension'
        orthonormalized_vectors = torch.zeros_like(vectors)
        orthonormalized_vectors[:, 0] = vectors[:, 0] / torch.norm(vectors[:, 0], p=2)

        for i in range(1, orthonormalized_vectors.size(1)):
            vector = vectors[:, i]
            V = orthonormalized_vectors[:, :i]
            PV_vector = torch.mv(V, torch.mv(V.t(), vector))
            orthonormalized_vectors[:, i] = (vector - PV_vector) / torch.norm(vector - PV_vector, p=2)

        return orthonormalized_vectors

    def sampling_nsphere(self):
        # https://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
        pass

    def execute(self):
        self.Data = self.generate_nsphere()

        howmany = self.computing_draw_analyze_ph()

        #if howmany:
        #    self.visualize_data(self.Data)
        return howmany

    def has_desired_persistence(self, diags = None):
        def pi_assessment(pi):
            if np.isinf(pi[1]):
                return False
            if self.is_alpha:
                pi = np.sqrt(pi)
            if (pi[1] - pi[0]) > self.persistence:
                return True
            return False

        if hasattr(self, "simplex_tree") and self.simplex_tree is not None:
            diags = self.simplex_tree.persistence_intervals_in_dimension(self.dim)
            del self.simplex_tree
        else:
            if diags is None:
                return False

        for pi in diags:
            if pi_assessment(pi):
                return True
        return False

    def computing_draw_analyze_ph(self):
        # compute homology according VR or alpha
        result_diags = None
        if self.is_alpha:
            diags = self.compute_alpha_persistence()
        else:
            #diags = self.compute_sparse_rips_persistence(metric="manhattan")
            result_diags = self.compute_ripser_persistence(metric="manhattan")
        howmany = self.has_desired_persistence(diags=result_diags)

        # if howmany:
        #     ax=gudhi.plot_persistence_diagram(diags)
        #     ax.set_title(f"Persistence diagram of a {self.dim}-sphere")
        #     ax.set_aspect("equal")  # forces to be square shaped
        #     plt.savefig(f"{self.output_path}/pd_spheres_npts({self.nPts})_dim({self.dim})_t({self.trial}).png")
        #     plt.close("all")

        return howmany

    def compute_alpha_persistence(self):
        complex = gudhi.AlphaComplex(points=self.Data, precision="fast")
        self.simplex_tree = complex.create_simplex_tree()#(max_alpha_square=0.51**2)
        diags = self.simplex_tree.persistence()

        del complex

        return diags

    def compute_ripser_persistence(self, metric):
        D = pairwise_distances(self.Data, n_jobs=-1, metric=metric)
        result_dict = ripser(D, distance_matrix=True, maxdim=self.dim)

        diags = []
        for d, Hi in enumerate(result_dict["dgms"]):
            if d != self.dim:
                continue
            for pi in Hi:
                diags.append(pi)

        del result_dict
        return diags

    def compute_sparse_rips_persistence(self, metric):
        '''
        We compute the persistence diagram in case it is not exists

        When the PD is computed we also save useful information such as:
        Persistence diagram, Barcode, and the diagrams information
        :return:
        '''
        D = pairwise_distances(self.Data, n_jobs=-1, metric=metric)
        max_value = np.quantile(D, 0.75) if len(self.Data) > 1000 else None # we use the 0.75 quantile as max value
        max_value2 = np.average(D) if len(self.Data) > 1000 else None # we use the 0.75 quantile as max value
        max_value = min(max_value, max_value2)
        enable_max_edge = max_value is not None
        enable_collapse = False
        max_dim = self.dim #+ 1
        if enable_max_edge:
            '''
            See Gudhi Rips complex python documentation https://gudhi.inria.fr/python/latest/rips_complex_user.html
            for a detailed exposition.
            According to Gudhi sparse value should not surpass 1, so we use 0.5
            '''
            complex = gudhi.RipsComplex(distance_matrix=D, max_edge_length=max_value)
        else:
            complex = gudhi.RipsComplex(distance_matrix=D)
        if enable_collapse is None or not enable_collapse:
            self.simplex_tree = complex.create_simplex_tree(max_dimension=max_dim)
        else:
            self.simplex_tree = complex.create_simplex_tree(max_dimension=float(1))
            self.simplex_tree.collapse_edges(1)
            self.simplex_tree.expansion(max_dim)

        diags = self.simplex_tree.persistence()

        del D
        del complex

        return diags

    def visualize_data(self, data):
        fig = plt.figure()
        plt_dim = data.shape[1]

        if plt_dim == 4:
            for point in data:
                point /= point[3]
        elif plt_dim > 5:
            from dimensionality_reduction import DimensionalityReduction

            rtype = DimensionalityReduction.PCA
            components = 3
            data = DimensionalityReduction.execute(X=data, rtype=rtype, components=components)
            plt_dim = components

        ax = fig.add_subplot(111, projection='3d') if plt_dim > 2 else fig.add_subplot(111)

        alphas = [1]

        if plt_dim > 2:
            ax.scatter(data[0: self.nPts, 0], data[0:self.nPts, 1],
                       data[0:self.nPts, 2], alpha=alphas[0])
        else:
            ax.scatter(data[0: self.nPts, 0], data[0:self.nPts, 1], alpha=alphas[0])
        plt.savefig(f"{self.output_path}/spheres_npts({self.nPts})_dim({self.dim})_t({self.trial}).png")
        plt.close("all")
