import GPy
import numpy as np
import safeopt.gp_opt
import multiprocessing
from multiprocessing.pool import Pool
import time
import sys
import safeopt.utilities
from scipy.stats import norm

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
    stop = int(sys.argv[7]) # iteration at which to return to stage T
    is_dynamic = int(sys.argv[8]) # dynamically stop safe expansion?
    lp = int(sys.argv[9])
    
    bounds = [(0., 1.), (0., 1.)]
    noise_var = 0.025
    noise_var2 = 0.025
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
              'save_path': save_path,
              'lp': lp}
    
    #run_trial([0, params])
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
    lp = params['lp']
    for function in range(num_functions):
        func_idx = process * num_functions + function    
        kernel = GPy.kern.Matern32(input_dim=len(bounds), variance=0.5,
                                   lengthscale=0.5, ARD=False)
        kernel2 = GPy.kern.Matern32(input_dim=len(bounds), variance=5.,
                                    lengthscale=0.2, ARD=False)
        kernel3 = GPy.kern.Matern32(input_dim=len(bounds), variance=5.,
                                    lengthscale=0.4, ARD=False)
        kernel4 = GPy.kern.Matern32(input_dim=len(bounds), variance=5.,
                                    lengthscale=0.6, ARD=False)
        
        def sample_safe_fun():
            safe_seeds = []
            while len(safe_seeds) < num_trials:
                fun = safeopt.utilities.sample_gp_function(kernel, bounds, noise_var, disc)
                fun2 = safeopt.utilities.sample_gp_function(kernel2, bounds,
                                                  noise_var2, disc)
                fun3 = safeopt.utilities.sample_gp_function(kernel3, bounds,
                                                  noise_var2, disc)
                fun4 = safeopt.utilities.sample_gp_function(kernel4, bounds,
                                                  noise_var2, disc)                
                def combined_fun(x, noise=True):
                    return np.hstack([fun(x, noise), fun2(x, noise),
                                      fun3(x, noise), fun4(x, noise)])
                safe_vals1 = [fun2(p, noise = False)[0][0] for p in parameter_set]
                safe_vals2 = [fun3(p, noise = False)[0][0] for p in parameter_set]
                safe_vals3 = [fun4(p, noise = False)[0][0] for p in parameter_set]
                
                thresh1 = np.mean(safe_vals1)
                seed_thresh1 = np.mean(safe_vals1) + 0.25 * np.std(safe_vals1)
                thresh2 = np.mean(safe_vals2)
                seed_thresh2 = np.mean(safe_vals2) + 0.25 * np.std(safe_vals2)
                thresh3 = np.mean(safe_vals3) 
                seed_thresh3 = np.mean(safe_vals3) + 0.25 * np.std(safe_vals3)
                
                safe_seeds = []
                for p in parameter_set:
                    if fun2(p, noise=False)[0][0] > seed_thresh1 and \
                       fun3(p, noise=False)[0][0] > seed_thresh2 and \
                       fun4(p, noise=False)[0][0] > seed_thresh3:
                        safe_seeds.append(p)
                if len(safe_seeds) >= num_trials:
                    return combined_fun, safe_seeds, thresh1, thresh2, thresh3
        # sample GP
        fun, safe_seeds, thresh1, thresh2, thresh3 = sample_safe_fun()
        safe_seeds = np.array(safe_seeds)
        seeds = safe_seeds[np.random.choice(np.arange(len(safe_seeds)),
                                            size=num_trials, replace=False)]
        #print(len(safe_seeds))
        #print(thresh)
        for i, seed in enumerate(seeds):
            try:
                print('function', func_idx, 'trial', i)
                CEI_regret = np.zeros(T)
                x0 = np.array([seed])
                y0 = fun(x0)            
                gp = GPy.models.GPRegression(x0, y0[:, 0, None],
                                             kernel, noise_var=noise_var)
                gp2 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel2, noise_var=noise_var2)
                gp3 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel3, noise_var=noise_var2)
                gp4 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel4, noise_var=noise_var2)
                safestage_reward = np.zeros(T)
                safeopt_reward = np.zeros(T)
                cei_reward = np.zeros(T)
                # Constrained EI
                curr_max = y0[0][0]
                cei_opt = safeopt.gp_opt.GaussianProcessOptimization([gp, gp2, gp3, gp4], parameter_set, num_contexts=0, threshold=[-np.inf, thresh1, thresh2, thresh3])
                cei_ss = np.zeros(T)
                for t in range(T):
                    # compute CEI index
                    max_cei = -np.inf
                    max_x = parameter_set[0]
                    safe = 0
                    max_xpr = parameter_set[0]
                    max_pr = 0
                    for p in parameter_set:
                        y_util = fun(p, noise=False)[0][0]
                        y_safe = fun(p, noise=False)[0][1]
                        u_mean, u_var = gp.predict_noiseless(np.array([p]))
                        s_mean1, s_var1 = gp2.predict_noiseless(np.array([p]))
                        s_mean2, s_var2 = gp2.predict_noiseless(np.array([p]))
                        s_mean3, s_var3 = gp2.predict_noiseless(np.array([p]))
                        
                        s_std1 = np.sqrt(s_var1)
                        s_std2 = np.sqrt(s_var2)
                        s_std3 = np.sqrt(s_var3)
                                                
                        u_std = np.sqrt(u_var)
                        z_util = (curr_max - u_mean) / u_std
                        p1 = (1 - norm.cdf((thresh1-s_mean1)/s_std1))
                        p2 = (1 - norm.cdf((thresh2-s_mean2)/s_std2))
                        p3 = (1 - norm.cdf((thresh3-s_mean3)/s_std3))
                        if p1 * p2 * p3 > max_pr:
                            max_pr = p1 * p2 * p3
                            max_xpr = p
                        if p1 > 0.999 and p2 > 0.999  and p3 > 0.999:
                            safe += 1
                        else:
                            continue
                        CEI = p1 * p2 * p3 * u_std * \
                              (z_util * norm.cdf(z_util) + norm.pdf(z_util))
                        if CEI > max_cei:
                            max_cei = CEI
                            max_x = p
                    if max_cei < -1000000:
                        max_x = max_xpr
                    cei_ss[t] += safe
                    y_meas = fun(max_x)
                    # Add this to the GP model
                    cei_opt.add_new_data_point(max_x, y_meas)                          
                    y_actual = fun(max_x, noise=False)[0][0]
                    if y_actual > curr_max:
                        curr_max = y_actual
                    cei_reward[t] += curr_max
                                            
                gp = GPy.models.GPRegression(x0, y0[:, 0, None],
                                             kernel, noise_var=noise_var)
                gp2 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel2, noise_var=noise_var2)
                gp3 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel3, noise_var=noise_var2)
                gp4 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel4, noise_var=noise_var2)
                # StageOpt
                opt = safeopt.gp_opt.SafeStage([gp, gp2, gp3, gp4], parameter_set,
                                               [-np.inf, thresh1, thresh2, thresh3], lipschitz=None,
                                               threshold=0)
                curr_max = -np.inf
                safe_sizes = np.zeros(T)
                last = stop
                for t in range(stop):
                    x_next, ss = opt.optimize()
                    # Get a measurement from the real system
                    y_meas = fun(x_next)
                    # Add this to the GP model
                    opt.add_new_data_point(x_next, y_meas)   
                    y_actual = fun(x_next, noise=False)[0][0]
                    #print(y_actual)
                    if y_actual > curr_max:
                        curr_max = y_actual
                    safestage_reward[t] += curr_max
                    # check if we should stop dynamically
                    safe_sizes[t] = ss
                    if t >= lp and ss == safe_sizes[t-lp]:
                        last = t
                        break
                # print(last, stop)
                # print(safe_sizes)
                for t in range(min(last + 1, stop), T):
                    x_next, ss = opt.optimize(exploit=True)
                    # Get a measurement from the real system
                    y_meas = fun(x_next)
                    # Add this to the GP model
                    opt.add_new_data_point(x_next, y_meas)   
                    y_actual = fun(x_next, noise=False)[0][0]
                    if y_actual > curr_max:
                        curr_max = y_actual
                    safestage_reward[t] += curr_max
                    safe_sizes[t] = ss
                # SafeOpt
                opt = safeopt.gp_opt.SafeOpt([gp, gp2, gp3, gp4], parameter_set,
                                               [-np.inf, thresh1, thresh2, thresh3], lipschitz=None,
                                               threshold=0)                
                gp = GPy.models.GPRegression(x0, y0[:, 0, None],
                                             kernel, noise_var=noise_var)
                gp2 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel2, noise_var=noise_var2)
                gp3 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel3, noise_var=noise_var2)
                gp4 = GPy.models.GPRegression(x0, y0[:, 1, None],
                                              kernel4, noise_var=noise_var2)
                curr_max = -np.inf
                opt_ss = np.zeros(T)
                for t in range(100):
                    x_next, ss = opt.optimize()
                    # Get a measurement from the real system
                    y_meas = fun(x_next)
                    # Add this to the GP model
                    opt.add_new_data_point(x_next, y_meas)   
                    y_actual = fun(x_next, noise=False)[0][0]
                    if y_actual > curr_max:
                        curr_max = y_actual
                    safeopt_reward[t] += curr_max
                    opt_ss[t] = ss

                # save rewards
                pref = save_path + 'function' + str(func_idx) + \
                       'trial' + str(i)
                np.save(pref + 'ceisizes', cei_ss)
                np.save(pref + 'ceirew', cei_reward)
                np.save(pref + 'stagerew', safestage_reward)
                np.save(pref + 'optrew', safeopt_reward)
                np.save(pref + 'stagestop', last)
                np.save(pref + 'stagesizes', safe_sizes)
                np.save(pref + 'optsizes', opt_ss)
            except EnvironmentError:
                continue
            
if __name__ == '__main__':
    main()
