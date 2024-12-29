import numpy as np
import sys
import time
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
import gudhi as gd
from utils import get_total_points, sample_layered_sphere, test_persistence


def run_experiment(params):
    n_points, dim, m, k = params
    points = sample_layered_sphere(n_points, dim, m, k)
    
    return test_persistence(points, dim, k)

def test_configuration(n_points, dim, m, k, n_trials=100, confidence=0.95):
    """Test configuration with parallel processing"""
    with ProcessPoolExecutor() as executor:
        failure = 0
        success = 0
        rate = 0
        total_npoints = get_total_points(n_points, m)
        expected_fails = n_trials - confidence*100
        params = [(n_points, dim, m, k) for _ in range(n_trials)]
        
        # submit all tasks
        futures = [executor.submit(run_experiment, param) for param in params]
        # process results as they complete (unordered)
        for future in as_completed(futures):
            try:
                answ = future.result() 
                if not answ:
                    failure += 1
                else:
                    success += 1
                    
                # Early termination if failures exceed threshold
                if failure > expected_fails:
                    # as soon as we fail enough we kill the executor
                    # this raises a lot of exceptions due to forcing processes to finish
                    # we redirect those exceptions to susp_err.log file
                    for f in futures:
                        f.cancel()  # Cancel any unfinished tasks
                    executor.shutdown(wait=True) # cancel remaining tasks
                    break
            except Exception as e:
            #    # handle any exceptions during execution
                failure += 1
                raise sys.exc_info()[0](traceback.format_exc())
            
        # compute success rate
        rate = success / n_trials
        return (rate >= confidence), rate, total_npoints
    return (False, 0, total_npoints)


def find_points_linear(old_npts, dim, m, k):
    """ 
    finds the minimal amount of points that produces a persistence interval in Hd with lifetime k.
    We iterate linearly the possible number of points
    """
    empirical_upper_bound = 10 ** (dim+1) + old_npts - 1
    for npts in range(old_npts, empirical_upper_bound, old_npts):
        success, success_rate, total_points = test_configuration(npts, dim, m, k)
        if success:
            return npts, success_rate, total_points
    return None, 0, 0


def find_points_galloping(old_npts, dim, m, k):
    """ 
    finds the minimal amount of points that produces a persistence interval in Hd with lifetime k.
    We perform a galloping search to find the possible number of points
    """
    lower_bound = old_npts
    upper_bound = old_npts  # Start conservative
    empirical_upper_bound = 10 ** (dim+1) + old_npts - 1
    
    # First, find a valid upper bound with a galloping search
    success = False
    success_rate = 0
    total_points = lower_bound
    while not success:# and upper_bound < empirical_upper_bound:
        upper_bound *= 2
        success, success_rate, total_points = test_configuration(upper_bound, dim, m, k)
    
    # Binary search with tighter bounds
    min_points = upper_bound
    old_rate = success_rate
    old_total_npts = total_points
    while lower_bound <= upper_bound:
        mid = (lower_bound + upper_bound) // 2
        success, success_rate, total_points = test_configuration(mid, dim, m, k)
        if success:
            min_points = mid
            old_rate = success_rate
            old_total_npts = total_points
            upper_bound = mid - 1
        else:
            lower_bound = mid + 1
     
    return min_points, old_rate, old_total_npts


if __name__ == '__main__':
    # redirecting errors from cancelling unfinished tasks to susp_err.log
    sys.stderr = open('susp_err.log', 'w')
    # Test different configurations
    k_values = [0.5]#[0.25, 0.5, 0.75]
    m_values = [2, 3, 4, 5]
    dimensions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
    np.random.seed(42)
    
    output_path = time.strftime("./%y.%m.%d__%H.%M.%S_suspension_results")
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    result = open(f"{output_path}/minimal_points_per_dim.txt", "w")
    if not result:
        exit(1)

    for k in k_values:
        old_npts = 2
        for dim in dimensions:
            for m in m_values:
                #npts, rate, total_points = find_points_linear(old_npts, dim, m, k)
                npts, rate, total_points = find_points_galloping(old_npts, dim, m, k)
                
                if npts is not None:
                    file_line = f"k={k}, dim={dim}, m={m}, success_rate={rate} base_nPts={npts} total_npts={total_points}\n" 
                    old_npts = total_points
                    result.write(file_line)
                    result.flush()
                    print(file_line)
                    break
                else:
                    print(f"FAIL k={k}, dim={dim}, m={m}, success_rate={rate} base_nPts={npts} total_npts={total_points}")
    result.close()