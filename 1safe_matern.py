import GPy
import numpy as np
import safeopt.gp_opt
import multiprocessing
from multiprocessing.pool import Pool
import time
import sys
import safeopt.utilities

def main(): 
    # global T, save_path, num_functions, num_processes, num_trials, disc, \
    #     bounds, noise_var, noise_var2, stop, parameter_set
    multiprocessing.set_start_method('forkserver')
    
    T = int(sys.argv[1]) # time horizon
    num_processes = int(sys.argv[2]) # number of processes
    num_functions = int(sys.argv[3]) # number of functions per process    
    num_trials = int(sys.argv[4]) # number of trials per function
    save_path = sys.argv[5] # path to save trial data    
    disc = int(sys.argv[6]) # discretization factor
    stop = int(sys.argv[7])
    
    bounds = [(0., 1.), (0., 1.)]
    noise_var = 0.05 ** 2
    noise_var2 = 0.05 ** 2
    parameter_set = safeopt.linearly_spaced_combinations(bounds, disc)

    params = {'T': T,
              'num_processes': num_processes,
              'num_functions': num_functions,
              'num_trials': num_trials,
              'disc': disc,
              'bounds': bounds,
              'noise_var': noise_var,
              'noise_var2': noise_var2,
              'stop': stop,
              'parameter_set': parameter_set,
              'save_path': save_path}
    
    #run_trial(0)
    start_time = time.time()                                    
    p = Pool(processes = num_processes)    
    p.map_async(run_trial, zip(range(num_processes),
                               [params] * num_processes)).get(10000000)
    p.close()
    p.join()                                    
    print(time.time() - start_time)    
                     
def run_trial(args):
    process = args[0]
    params = args[1]
    T = params['T']
    num_processes = params['num_processes']
    num_functions = params['num_functions']
    num_trials = params['num_trials']
    disc = params['disc']
    bounds = params['bounds']
    noise_var = params['noise_var']
    noise_var2 = params['noise_var2']
    stop = params['stop']
    parameter_set = params['parameter_set']
    save_path = params['save_path']
    for function in range(num_functions):
        func_idx = process * num_functions + function    
        kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=0.5,
                              lengthscale=0.5, ARD=True)
        kernel2 = kernel.copy()
        def sample_safe_fun():
            safe_seeds = []
            while len(safe_seeds) < num_trials:
                fun = safeopt.utilities.sample_gp_function(kernel, bounds, noise_var, disc)
                fun2 = safeopt.utilities.sample_gp_function(kernel2, bounds,
                                                  noise_var2, disc)
                def combined_fun(x, noise=True):
                    return np.hstack([fun(x, noise), fun2(x, noise)])
                safe_vals = [fun2(p, noise = False)[0][0] for
                             p in parameter_set]
                thresh = np.mean(safe_vals) + 0.5 * np.std(safe_vals)
                seed_thresh = np.mean(safe_vals) + np.std(safe_vals)

                safe_seeds = []
                for p in parameter_set:
                    if fun2(p, noise=False)[0][0] > seed_thresh:
                        safe_seeds.append(p)
                if len(safe_seeds) > num_trials:
                    return combined_fun, safe_seeds, thresh
            return None
        # sample GP
        fun, safe_seeds, thresh = sample_safe_fun()
        safe_seeds = np.array(safe_seeds)
        seeds = safe_seeds[np.random.choice(np.arange(len(safe_seeds)),
                                            size=num_trials, replace=False)]
        for i, seed in enumerate(seeds):
            try:
                print('function', func_idx, 'trial', i)
                x0 = np.array([seed])
                y0 = fun(x0)            
                gp = GPy.models.GPRegression(x0, y0[:, 0, None],
                                             kernel, noise_var=noise_var)
                gp2 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel2, noise_var=noise_var2)
                safestage_reward = np.zeros(T)
                safeopt_reward = np.zeros(T)
                # StageOpt
                opt = safeopt.gp_opt.SafeStage([gp, gp2], parameter_set,
                                               [-np.inf, thresh], lipschitz=None,
                                               threshold=thresh)
                curr_max = -np.inf
                for t in range(stop):
                    x_next = opt.optimize()
                    # Get a measurement from the real system
                    y_meas = fun(x_next)
                    # Add this to the GP model
                    opt.add_new_data_point(x_next, y_meas)   
                    y_actual = fun(x_next, noise=False)[0][0]
                    #print(y_actual)
                    if y_actual > curr_max:
                        curr_max = y_actual
                    safestage_reward[t] += curr_max
                for t in range(stop, T):
                    x_next = opt.optimize(exploit=True)
                    # Get a measurement from the real system
                    y_meas = fun(x_next)
                    # Add this to the GP model
                    opt.add_new_data_point(x_next, y_meas)   
                    y_actual = fun(x_next, noise=False)[0][0]
                    if y_actual > curr_max:
                        curr_max = y_actual
                    safestage_reward[t] += curr_max  
                # SafeOpt
                gp = GPy.models.GPRegression(x0, y0[:, 0, None], kernel,
                                             noise_var=noise_var)
                gp2 = GPy.models.GPRegression(x0, y0[:, 1, None], kernel2,
                                              noise_var=noise_var2)
                opt = safeopt.gp_opt.SafeOpt([gp, gp2], parameter_set,
                                             [-np.inf, thresh], lipschitz=None,
                                             threshold=thresh)
                curr_max = -np.inf
                for t in range(100):
                    x_next = opt.optimize()
                    # Get a measurement from the real system
                    y_meas = fun(x_next)
                    # Add this to the GP model
                    opt.add_new_data_point(x_next, y_meas)   
                    y_actual = fun(x_next, noise=False)[0][0]
                    if y_actual > curr_max:
                        curr_max = y_actual
                    safeopt_reward[t] += curr_max

                # save rewards
                pref = save_path + 'function' + str(func_idx) + \
                       'trial' + str(i)
                np.save(pref + 'stagerew', safestage_reward)
                np.save(pref + 'optrew', safeopt_reward)
            except EnvironmentError:
                continue
            
if __name__ == '__main__':
    main()
