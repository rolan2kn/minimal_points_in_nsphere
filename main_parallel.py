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

import os
import sys
import math
import time
import os.path
import random
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from numpy.random import randn
from numpy.linalg import norm
from sklearn.metrics import pairwise_distances

from utils import sampling_nsphere, test_persistence 


def process_trial(items):
    """ execute the experiment in the current trial """
    nPts, dim, t, k = items
    
    points = sampling_nsphere(nPts, dim)
    
    answer = test_persistence(points, dim, k)

    del points

    return answer


def process_points(nPts, dim, N, k, expected_fails):
    # failures should not be greater than the expected_fails

    with ProcessPoolExecutor() as executor:
        failures = 0
        items = [(nPts, dim, t, k) for t in range(N)]
        
        # submit all tasks
        futures = [executor.submit(process_trial, item) for item in items]
        
        for future in as_completed(futures):
            try:
                if not future.result():
                    failures += 1
                if failures > expected_fails:
                    # as soon as we fail enough we kill the executor
                    # this raises a lot of exceptions due to forcing processes to finish
                    # we redirect those exceptions to par_err.log file
                    for f in futures:
                        f.cancel()                  # Cancel any unfinished tasks
                    executor.shutdown(wait=False)   # cancel remaining tasks
                    
                    break
            except Exception as e:
                failures += 1
            
    answer = failures <= expected_fails
    return answer, failures

def process_points_old(nPts, dim, N, k, expected_fails):
    # failures should not be greater than the expected_fails
    failures = 0
    no_cores = len(os.sched_getaffinity(0))
    iterations = (N+no_cores-1) // no_cores
    remaining = N

    for i in range(iterations):
        num = min(no_cores, remaining)
        passn = N - remaining
        with multiprocessing.Pool(no_cores) as pool:
            items = [(nPts, dim, t + passn, k) for t in range(num)]
            for _answer in pool.imap_unordered(process_trial, items):
                if not _answer:
                    failures += 1
                if failures > expected_fails:
                    # as soon as we fail enough we kill the pool
                    pool.terminate()
                    # we should increase the samples
                    return False, failures
        remaining -= num
    answer = failures <= expected_fails
    return answer, failures

def find_points_galloping(dim, output_path, old_pts, N, k, expected_fails, result):
    lower_bound = old_pts
    upper_bound = old_pts
    empirical_upper_bound = 10 ** (dim+1) + old_pts - 1
    
    # First, find a valid upper bound with a galloping search
    success = False
    fails = 0
    while not success  and upper_bound < empirical_upper_bound:
        upper_bound *= 2
        success, fails = process_points(upper_bound, dim, N, k, expected_fails)
    min_points = upper_bound
    conf = N - fails
    ans = False
    while lower_bound <= upper_bound:
        mid = lower_bound + (upper_bound-lower_bound) // 2
        # Now each test runs N trials
        ans, fails = process_points(mid, dim, N, k, expected_fails)
        if ans:
            conf = N - fails
            min_points = mid
            upper_bound = mid - 1
        else:
            lower_bound = mid + 1
    file_line = f"dim: {dim} nPts {min_points} with confidence {conf}\n"
    result.write(file_line)
    result.flush()
    print(file_line)

    return min_points

def find_points_linear(dim, output_path, old_pts, N, k, expected_fails, result):
    empirical_upper_bound = 10 ** (dim+1) + old_pts - 1
    tmp_output_path = f"{output_path}/dim{dim}"
    if not os.path.isdir(tmp_output_path):
        os.makedirs(tmp_output_path)

    for nPts in range(old_pts, empirical_upper_bound, old_pts):
        answer, fails = process_points(nPts, dim, N, k, expected_fails)
        if answer:
            file_line = f"dim: {dim} nPts {nPts} with confidence {N - fails}\n"
            old_pts = nPts
            result.write(file_line)
            result.flush()
            print(file_line)
            return old_pts

    return old_pts

if __name__ == '__main__':
    np.random.seed(42)
    # redirecting errors from cancelling unfinished tasks to par_err.log
    sys.stderr = open('par_err.log', 'w')

    Dmax = 30 # Largest dimension
    N = int(100) # Monte Carlo samples
    confidence = int((N * 95)//100)
    expected_fails = N-confidence
    k = 0.5

    output_path = time.strftime("./%y.%m.%d__%H.%M.%S_parallel_results")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    result = open(f"{output_path}/minimal_points_per_dim.txt", "w")
    if not result:
        exit(1)
    old_pts = 2
    for dim in range(1, Dmax):

        old_pts = find_points_linear(dim,
        #old_pts = find_points_galloping(dim,
                              output_path,
                              old_pts,
                              N,
                              k,
                              expected_fails,
                              result)

    result.close()