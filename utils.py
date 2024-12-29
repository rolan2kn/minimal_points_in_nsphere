import random
import math
import time
import numpy as np
import os.path

import matplotlib.pyplot as plt

from numpy.random import randn
from numpy.linalg import norm
import gudhi as gd

def sampling_nsphere(n_points, dim):
    """ Generate nsphere data. All nspheres are centerd at the origin
    
        https://compneuro.uwaterloo.ca/files/publications/voelker.2017.pdf
    """

    foo = np.random.randn(n_points, dim + 1)
    points = (foo / norm(foo, axis=1, keepdims=True))

    del foo
    return points


def get_total_points(n_points, m):
    """ Returns the total number of points in a m-layered (n+1)-sphere resulting from the suspension 
        of a nsphere with n_points. This was extracted from sample_layered_sphere
    """
    total = 0

    angles = np.linspace(0, np.pi, 2 * m + 3)  # Uniform in angle space
    heights = np.cos(angles)#*k/2  # Convert to heights (t_i)
    
    
    for t in heights:
        if t == 1 or t == -1:  # North pole
            total += 1
            
        r = np.sqrt(1 - t**2)
        
        total += int(n_points * r)
        
    return total

def sample_layered_sphere(n_points, dim, m, k):
    """Sample points on an (n+1)-sphere with geometry-aware layer distribution
        Args:
            n_points: Base points for equator
            dim: Target sphere dimension 
            m: Layers per side
            k: Desired persistence
        Returns:
            Points on layer as array
    """
    layers = []
    
    # Compute angles for layers (including poles and equator)
    angles = np.linspace(0, np.pi, 2 * m + 3)  # Uniform in angle space
    heights = np.cos(angles)  # Convert to heights (t_i)
    
    for t in heights:
        if t == 1:  # North pole
            layers.append(np.array([[0]*dim + [1]]))
        elif t == -1:  # South pole
            layers.append(np.array([[0]*dim + [-1]]))
        else:
            # Radius at this height
            r = np.sqrt(1 - t**2)
            points_in_layer = int(n_points * r)  # Scale density by radius

            # Sample uniformly on S^(dim-1)
            X = np.random.randn(points_in_layer, dim)
            X /= np.linalg.norm(X, axis=1)[:, np.newaxis]

            # Scale and add height
            X = np.column_stack([X * r, np.full(points_in_layer, t)])
            layers.append(X)
    
    return np.vstack(layers)

def test_persistence(points, dim, k):
    """Test if points achieve persistence k in dimension dim"""
    alpha_complex = gd.AlphaComplex(points=points)
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.persistence()
    
    diag = simplex_tree.persistence_intervals_in_dimension(dim)
    
    # Check for persistence interval ≥ k in dimension dim
    for birth, death in diag:
        if np.sqrt(death) - np.sqrt(birth) >= k:   #  
            return True
    return False
