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
from data_controller import DataController

'''
Given an n-dimensional sphere unit S^n, the homology groups are Z in
dimension 0 and n, and 0 otherwise.

Now, suppose we sample points from uniform distribution in S^n. If I
am able to sample many many many points, I expect then a persistence
interval [0,1] in dimension n.

The question is, how many points (n) do I need to sample in order to
have, with a high probability, an interval of a length k \iin [0,1] in
n dimensional persistence. We can, at the beginning, set k to 0.5,

Now, the tricky part is what do I mean by high probability? Let us
define it in Monte Carlo fashion - say that, in the collection of N
samples, 95% of cases we observe such a interval.
So, is N = 100, if I sample n points 100 times, 95 times we will
observe a persistence interval of a length greater of equal k in
dimension n.

I want you to find n as a function of k and the significance level (95%)
'''
if __name__ == '__main__':
    np.random.seed(42)

    Dmax = 30 # Largest dimension
    N = int(100) # Monte Carlo samples
    confidence = int((N * 95)//100)
    expected_fails = N-confidence
    k = 0.5

    output_path = time.strftime("./%y.%m.%d__%H.%M.%S_results")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    result = open(f"{output_path}/minimal_points_per_dim.txt", "w")
    if not result:
        exit(1)
    old_pts = 2
    for dim in range(1,Dmax):
        min_nums = []
        tmp_output_path = f"{output_path}/dim{dim}"
        if not os.path.isdir(tmp_output_path):
            os.makedirs(tmp_output_path)

        for nPts in range(old_pts, 100000, old_pts):
            # nPts, dim, trial, sphere_type = None, output_path=None
            # failures should be less than expected_fails
            failures = 0
            for t in range(N):
                dc = DataController(nPts = nPts,
                                    dim=dim,
                                    trial=t,
                                    persistence=k,
                                    output_path=tmp_output_path,
                                    is_alpha=True)
                howmany = dc.execute()
                if not howmany:
                    failures += 1
                #else:
                #    print(f"H_{dim}: true for {nPts} in trial {t}")
                if failures > expected_fails:
                    # we should increase the samples
                    break

                del dc
            if failures <= expected_fails:
                file_line = f"dim: {dim} nPts {nPts} with confidence {N-failures}\n"
                old_pts = nPts
                result.write(file_line)
                result.flush()
                print(file_line)
                break
    result.close()