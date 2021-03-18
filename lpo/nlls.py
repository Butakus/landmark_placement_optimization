#!/usr/bin/env python3

""" TODO: docstring """

import time
from tqdm import tqdm

import multiprocessing as mp

import numpy as np
# Set numpy random seed
# np.random.seed(42)

import lmfit
import matplotlib.pyplot as plt
import corner

from landmark_detection import landmark_detection
import lie_algebra as lie

# landmarks = np.array([
#     # [1.0, 5.0, 0.0],
#     # [1.0, 10.0, 0.0],
#     # [1.0, -5.0, 0.0],
#     # [1.0, -10.0, 0.0],
#     [-5.0, 5.0, 0.0],
#     [5.0, 5.0, 0.0],
#     [5.0, -5.0, 0.0],
#     [-5.0, -5.0, 0.0],
#     # [0.0, 0.0, 0.0],
#     # [-2.0, 2.0, 0.0],
#     # [2.0, 2.0, 0.0],
#     # [2.0, -2.0, 0.0],
#     # [-2.0, -2.0, 0.0],
# ])

landmarks = np.array([
    [5.0, 40.0, 0.0],
    [40.0, 50.0, 0.0],
    [55.0, 25.0, 0.0],
    [65.0, 55.0, 0.0],
    [85.0, 60.0, 0.0],
    [100.0, 30.0, 0.0],
    [120.0, 60.0, 0.0],
    [125.0, 23.0, 0.0],
])


def distance(x, y):
    """ TODO: docstring """
    return np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)


def residual(params, landmarks, measurements, measurement_covs):
    """ TODO: docstring """
    # print("##################################################")
    # print("params: {}".format(params))
    trans = np.array([params['x'], params['y'], 0.0])
    theta = params['theta']
    theta = theta % (2*np.pi)
    rot = lie.so3_from_rpy([0.0, 0.0, theta])
    pose = lie.se3(t=trans, r=rot)
    # print("[x, y, theta]: {}, {}, {}".format(trans[0], trans[1], params['theta']+0.0))

    # Compute log-likelihood of each measurement
    predicted_measurements, _ = landmark_detection(pose, landmarks, std=0.0)

    # measurement_likelihood = np.empty((measurements.shape[0], 1))
    measurement_likelihood = []
    for l in range(measurements.shape[0]):
        if np.isfinite(predicted_measurements[l]).any() and np.isfinite(measurements[l]).any():
            # TODO: Compute error on manifold
            # TODO: Find the proper error equation
            e = measurements[l, :2, 3] - predicted_measurements[l, :2, 3]
            # print("e: {}".format(e))
            # measurement_likelihood[l] = distance(e / measurement_covs[l].diagonal(), (0, 0))
            likelihood = distance(e / np.sqrt(measurement_covs[l].diagonal()), (0, 0))
            measurement_likelihood.append(likelihood)
            # measurement_inf = np.linalg.inv(measurement_covs[l])
            # measurement_likelihood[l] = np.matmul(np.matmul(e.T, measurement_inf), e)
            # print("e: {}".format(e))
            # print("d(e/cov): {}".format(distance(e / measurement_covs[l].diagonal(), (0, 0))))
            # print("d(e/sqrt(cov)): {}".format(distance(e / np.sqrt(measurement_covs[l].diagonal()), (0, 0))))
            # print("d(e.T*H*e): {}".format(np.matmul(np.matmul(e.T, measurement_inf), e))
        # else:
            # Very tricky.....
            # measurement_likelihood.append(1000.0)
    # print("measurement_likelihood:\n{}".format(measurement_likelihood))
    return np.array(measurement_likelihood)


def nlls_estimation(initial_guess=None, args=(), output=True):
    """ TODO: docstring """

    # Setup parameters
    params = lmfit.Parameters()
    params.add('x', min=0.0, max=150.0)
    params.add('y', min=0.0, max=90.0)
    params.add('theta', min=0.0, max=2*np.pi)

    # Create NLLS minimizer
    minimizer = lmfit.Minimizer(residual, params, fcn_args=args)

    # Set initial guess, or else compute it with Nelder-Mead algorithm
    if initial_guess is not None:
        params['x'].value = initial_guess[0]
        params['y'].value = initial_guess[1]
        params['theta'].value = initial_guess[2]
    else:
        initial_result = minimizer.minimize(method='nelder')
        params = initial_result.params
        if output:
            print("Initial guess estimation from Nelder-Mead:")
            print(lmfit.fit_report(initial_result))

    # if output:
        # print("Initial guess:")
        # print(params.pretty_print())

    # Run Levenberg-Marquardt algorithm to find the optimal solution
    result = minimizer.minimize(method='leastsq', params=params)

    if output:
        print("Levenberg-Marquardt result:")
        print(result.params.pretty_print())
        # print("Result covariance (invalid):")
        # print(result.covar)
        print(lmfit.fit_report(result))
        # ci = lmfit.conf_interval(minimizer, result)
        # lmfit.printfuncs.report_ci(ci)

    return result


def mcmc_posterior_estimation(params, args, steps=1000, output=False, plot=False):
    """ TODO: docstring """

    # One walker for each measurement?
    nwalkers = 2*landmarks.shape[0]
    # Run emcee
    emcee_result = lmfit.minimize(residual, method='emcee', nan_policy='omit',
                                  burn=300, steps=steps, thin=1, nwalkers=nwalkers,
                                  params=params, is_weighted=True, progress=output, args=args,
                                  seed=1)

    if plot:
        corner.corner(emcee_result.flatchain, labels=emcee_result.var_names,
                      truths=list(emcee_result.params.valuesdict().values()))

    if output:

        print("Emcee Result:")
        print(emcee_result.params.pretty_print())
        print(lmfit.fit_report(emcee_result))

        highest_prob = np.argmax(emcee_result.lnprob)
        hp_loc = np.unravel_index(highest_prob, emcee_result.lnprob.shape)
        mle_soln = emcee_result.chain[hp_loc]
        for i, par in enumerate(params):
            params[par].value = mle_soln[i]

        # print('\nMaximum Likelihood Estimation from emcee       ')
        # print('-------------------------------------------------')
        # print('Parameter  MLE Value   Median Value   Uncertainty')
        # fmt = '  {:5s}  {:11.5f} {:11.5f}   {:11.5f}'.format
        # for name, param in params.items():
        #     print(fmt(name, param.value, emcee_result.params[name].value,
        #               emcee_result.params[name].stderr))

        print('\nError estimates from emcee:')
        print('------------------------------------------------------')
        print('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma')

        for name in params.keys():
            quantiles = np.percentile(emcee_result.flatchain[name],
                                      [2.275, 15.865, 50, 84.135, 97.275])
            median = quantiles[2]
            err_m2 = quantiles[0] - median
            err_m1 = quantiles[1] - median
            err_p1 = quantiles[3] - median
            err_p2 = quantiles[4] - median
            fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
            print(fmt(name, err_m2, err_m1, median, err_p1, err_p2))

    return emcee_result


def test_nlls_posterior():
    """ Perform multiple runs of the NLLS estimation with different measurement noise.
        Store the final estimation of each run and get the statistics of the set of results.
    """
    # Number of runs
    N = 100

    X = np.array([80.0, 40.0, np.pi/2])
    X_se3 = lie.se3(t=[X[0], X[1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, X[2]]))

    results = []

    t0 = time.time()
    for i in tqdm(range(N)):
        # Initialize initial guess
        initial_guess = X.copy()
        # Add noise to initial solution
        initial_guess[:2] += np.random.normal(0.0, 0.5, 2)
        initial_guess[2] += np.random.normal(0.0, 0.2)
        # Get the landmark measurements
        measurements, measurement_covs = landmark_detection(X_se3, landmarks)
        # Estimate the NLLS solution
        nlls_result = nlls_estimation(args=(landmarks, measurements, measurement_covs),
                                      initial_guess=initial_guess, output=False)
        results.append(nlls_result)
    t1 = time.time()
    print("Total time: {}".format(t1 - t0))
    print("Avg time per fcn call: {}".format((t1 - t0) / N))
    x = np.array([r.params['x'] for r in results])
    y = np.array([r.params['y'] for r in results])
    theta = np.array([r.params['theta'] for r in results])
    print("x mean: {}".format(np.mean(x)))
    print("x median: {}".format(np.median(x)))
    print("x std: {}".format(np.std(x)))
    print("y mean: {}".format(np.mean(y)))
    print("y median: {}".format(np.median(y)))
    print("y std: {}".format(np.std(y)))
    print("theta mean: {}".format(np.mean(theta)))
    print("theta median: {}".format(np.median(theta)))
    print("theta std: {}".format(np.std(theta)))

    print("Quantiles:")
    print('------------------------------------------------------')
    print('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma')
    data_dicts = {'x': x, 'y': y, 'theta': theta}
    for p in ['x', 'y', 'theta']:
        quantiles = np.percentile(data_dicts[p],
                                  [2.275, 15.865, 50, 84.135, 97.275])
        median = quantiles[2]
        err_m2 = quantiles[0] - median
        err_m1 = quantiles[1] - median
        err_p1 = quantiles[3] - median
        err_p2 = quantiles[4] - median
        fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
        print(fmt(p, err_m2, err_m1, median, err_p1, err_p2))


def nlls_proxy(pose):
    pose_se3 = lie.se3(t=[pose[0], pose[1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, pose[2]]))
    # Initialize initial guess
    initial_guess = pose.copy()
    # Add noise to initial solution
    initial_guess[:2] += np.random.normal(0.0, 0.5, 2)
    initial_guess[2] += np.random.normal(0.0, 0.2)
    # Get the landmark measurements
    measurements, measurement_covs = landmark_detection(pose_se3, landmarks)  # std=0.01
    # Estimate the NLLS solution
    nlls_result = nlls_estimation(args=(landmarks, measurements, measurement_covs),
                                  initial_guess=initial_guess, output=False)
    return nlls_result


def nlls_sample(pose, sample_size):
    results = []
    for i in tqdm(range(sample_size)):
        nlls_result = nlls_proxy(pose)
        results.append(nlls_result)

    x = np.array([r.params['x'] for r in results])
    y = np.array([r.params['y'] for r in results])
    theta = np.array([r.params['theta'] for r in results])

    return (x, y, theta)


def nlls_pool_sample(pose, sample_size):
    async_results = []
    pool = mp.Pool(1)
    # pool = mp.Pool(mp.cpu_count())
    for i in range(sample_size):
        async_result = pool.apply_async(nlls_proxy, args=(pose,))
        async_results.append(async_result)
    pool.close()
    pool.join()

    x = np.array([r.get().params['x'] for r in async_results])
    y = np.array([r.get().params['y'] for r in async_results])
    theta = np.array([r.get().params['theta'] for r in async_results])

    return (x, y, theta)


def test_sim_accuracy():
    X = np.array([80.0, 40.0, np.pi/2])

    # sample_sizes = [50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 2000, 3000, 6000]
    sample_sizes = []
    sample_sizes += list(range(5, 50, 5))
    sample_sizes += list(range(50, 200, 10))
    # sample_sizes += list(range(200, 500, 20))
    # sample_sizes += list(range(500, 1000, 50))
    # sample_sizes += list(range(1000, 3000, 100))
    # sample_sizes += list(range(3000, 6000, 300))
    # sample_sizes += list(range(6000, 10000, 1000))
    sample_sizes = np.array(sample_sizes)

    x_errors = np.empty(sample_sizes.shape)
    y_errors = np.empty(sample_sizes.shape)
    errors = np.empty(sample_sizes.shape)
    x_stds = np.empty(sample_sizes.shape)
    y_stds = np.empty(sample_sizes.shape)
    stds = np.empty(sample_sizes.shape)
    times = np.empty(sample_sizes.shape)

    for i in range(sample_sizes.shape[0]):
        print("##################################")
        N = sample_sizes[i]
        print("N: {}".format(N))
        # results = []

        t0 = time.time()
        x, y, theta = nlls_pool_sample(pose=X, sample_size=N)
        t1 = time.time()
        print("Total time: {}".format(t1 - t0))
        print("Avg time per fcn call: {}".format((t1 - t0) / N))
        # x = np.array([r.params['x'] for r in results])
        # y = np.array([r.params['y'] for r in results])
        # theta = np.array([r.params['theta'] for r in results])
        print("x mean: {}".format(np.mean(x)))
        print("x median: {}".format(np.median(x)))
        print("x std: {}".format(np.std(x)))
        print("y mean: {}".format(np.mean(y)))
        print("y median: {}".format(np.median(y)))
        print("y std: {}".format(np.std(y)))
        print("theta mean: {}".format(np.mean(theta)))
        print("theta median: {}".format(np.median(theta)))
        print("theta std: {}".format(np.std(theta)))

        # print("Quantiles:")
        # print('------------------------------------------------------')
        # print('Parameter  -2sigma  -1sigma   median  +1sigma  +2sigma')
        # data_dicts = {'x': x, 'y': y, 'theta': theta}
        # for p in ['x', 'y', 'theta']:
        #     quantiles = np.percentile(data_dicts[p],
        #                               [2.275, 15.865, 50, 84.135, 97.275])
        #     median = quantiles[2]
        #     err_m2 = quantiles[0] - median
        #     err_m1 = quantiles[1] - median
        #     err_p1 = quantiles[3] - median
        #     err_p2 = quantiles[4] - median
        #     fmt = '  {:5s}   {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f}'.format
        #     print(fmt(p, err_m2, err_m1, median, err_p1, err_p2))
        x_error = abs(np.mean(x) - X[0])
        y_error = abs(np.mean(y) - X[1])
        x_errors[i] = x_error
        y_errors[i] = y_error
        errors[i] = np.sqrt(x_error**2 + y_error**2)
        x_stds[i] = np.std(x)
        y_stds[i] = np.std(y)
        stds[i] = np.sqrt(np.std(x)**2 + np.std(y)**2)
        times[i] = t1 - t0
        # x_errors.append(x_error)
        # y_errors.append(y_error)
        # errors.append(np.sqrt(x_error**2 + y_error**2))
        # x_stds.append(np.std(x))
        # y_stds.append(np.std(y))
        # errors.append(np.sqrt(np.std(x)**2 + np.std(y)**2))
        # times.append(t1 - t0)

    print("x_errors: {}".format(x_errors))
    print("y_errors: {}".format(y_errors))
    print("errors: {}".format(errors))
    print("x_stds: {}".format(x_stds))
    print("y_stds: {}".format(y_stds))
    print("stds: {}".format(stds))
    print("times: {}".format(times))

    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("NLLS sim")
    ax[0].set_title("Error (mm)")
    # ax[0].plot(sample_sizes, x_errors, color='r')
    # ax[0].plot(sample_sizes, y_errors, color='b')
    ax[0].plot(sample_sizes, errors*1000.0, color='b')

    ax[1].set_title("Std (mm)")
    # ax[1].plot(sample_sizes, x_stds, color='r')
    # ax[1].plot(sample_sizes, y_stds, color='b')
    ax[1].plot(sample_sizes, stds*1000.0, color='b')

    ax[2].set_title("Processing Time (s)")
    ax[2].plot(sample_sizes, times, color='b')


def mcmc_proxy(pose, N):
    t0 = time.time()
    pose_se3 = lie.se3(t=[pose[0], pose[1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, pose[2]]))
    measurements, measurement_covs = landmark_detection(pose_se3, landmarks, std=0.01)
    nlls_result = nlls_estimation(args=(landmarks, measurements, measurement_covs),
                                  initial_guess=None, output=False)
    emcee_result = mcmc_posterior_estimation(params=nlls_result.params, steps=N,
                                             args=(landmarks, measurements, measurement_covs),
                                             output=False)
    t1 = time.time()
    return (
        emcee_result.params['x'].stderr,
        emcee_result.params['y'].stderr,
        t1 - t0
    )


def test_mcmc_accuracy():
    X = np.array([80.0, 40.0, np.pi/2])
    # X_se3 = lie.se3(t=[X[0], X[1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, X[2]]))

    # sample_sizes = [50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 2000, 3000, 6000]
    chain_steps = []
    # chain_steps += list(range(300, 500, 50))
    chain_steps += list(range(500, 1000, 50))
    chain_steps += list(range(1000, 2000, 200))
    # chain_steps += list(range(2000, 5000, 100))
    chain_steps = np.array(chain_steps)

    x_stds = np.empty(chain_steps.shape)
    y_stds = np.empty(chain_steps.shape)
    stds = np.empty(chain_steps.shape)
    times = np.empty(chain_steps.shape)

    pool = mp.Pool(1)
    async_results = []
    # pool = mp.Pool(mp.cpu_count())
    for i in range(chain_steps.shape[0]):
        async_result = pool.apply_async(mcmc_proxy, args=(X, chain_steps[i]))
        async_results.append(async_result)
    pool.close()
    pool.join()

    for i in range(chain_steps.shape[0]):
        x_stds[i] = async_results[i].get()[0]
        y_stds[i] = async_results[i].get()[1]
        stds[i] = np.sqrt(x_stds[i]**2 + y_stds[i]**2)
        times[i] = async_results[i].get()[2]

    # for i in range(chain_steps.shape[0]):
    #     print("##################################")
    #     N = chain_steps[i]
    #     print("N: {}".format(N))
    #     t0 = time.time()
    #     measurements, measurement_covs = landmark_detection(X_se3, landmarks, std=0.01)
    #     nlls_result = nlls_estimation(args=(landmarks, measurements, measurement_covs),
    #                                        initial_guess=None, output=False)
    #     emcee_result = mcmc_posterior_estimation(params=nlls_result.params, steps=N,
    #                                              args=(landmarks, measurements, measurement_covs),
    #                                              output=False)
    #     t1 = time.time()
    #     x_stds[i] = emcee_result.params['x'].stderr
    #     y_stds[i] = emcee_result.params['y'].stderr
    #     times[i] = t1 - t0

    x_std_diffs = np.abs(x_stds - 0.0038)
    y_std_diffs = np.abs(y_stds - 0.0038)
    std_diffs = np.abs(stds - 0.00525)

    print("x_stds: {}".format(x_stds))
    print("y_stds: {}".format(y_stds))
    print("x_std_diffs: {}".format(x_std_diffs))
    print("y_std_diffs: {}".format(y_std_diffs))
    print("times: {}".format(times))

    fig, ax = plt.subplots(3, 1, sharex=True)
    plt.suptitle("MCMC sim")
    ax[0].set_title("Std (mm)")
    # ax[0].plot(chain_steps, x_stds, color='r')
    # ax[0].plot(chain_steps, y_stds, color='b')
    ax[0].plot(chain_steps, stds*1000.0, color='b')

    ax[1].set_title("Std diff (mm)")
    # ax[1].plot(chain_steps, x_std_diffs, color='r')
    # ax[1].plot(chain_steps, y_std_diffs, color='b')
    ax[1].plot(chain_steps, std_diffs*1000.0, color='b')

    ax[2].set_title("Processing Time (s)")
    ax[2].plot(chain_steps, times, color='b')


def main():
    X = np.array([80.0, 40.0, np.pi/2])
    initial_guess = X.copy()

    # Add noise to initial solution
    initial_guess[:2] += np.random.normal(0.0, 0.5, 2)
    initial_guess[2] += np.random.normal(0.0, 0.2)

    # Add extra deviation for initial solution
    # initial_guess[:2] += 3.0
    # initial_guess[2] += np.pi

    # Normalize angle
    initial_guess[2] = initial_guess[2] % (2*np.pi)

    print("True X: {}".format(X))
    print("Initial guess for X: {}".format(initial_guess))

    # Get the landmark measurements
    X_se3 = lie.se3(t=[X[0], X[1], 0.0], r=lie.so3_from_rpy([0.0, 0.0, X[2]]))
    measurements, measurement_covs = landmark_detection(X_se3, landmarks)
    # print("measurements:\n{}".format(measurements))
    # print("measurement_covs:\n{}".format(measurement_covs))

    # Estimate the NLLS solution
    # nlls_result = nlls_estimation(args=(landmarks, measurements, measurement_covs),
    #                               initial_guess=X, output=True)

    # # Run emcee
    # emcee_result = mcmc_posterior_estimation(params=nlls_result.params,
    #                                          args=(landmarks, measurements, measurement_covs),
    #                                          output=True)
    # print(emcee_result.params)
    # print(emcee_result.params['x'].stderr)

    # Run posterior tests to validate the posterior density estimation obtained from emcee
    # test_nlls_posterior()
    test_sim_accuracy()
    # test_mcmc_accuracy()

    plt.show()


if __name__ == "__main__":
    main()
