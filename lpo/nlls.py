#!/usr/bin/env python3

""" TODO: docstring """

import time
from tqdm import tqdm

import numpy as np
# Set numpy random seed
# np.random.seed(1)

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

    measurement_likelihood = np.empty((measurements.shape[0], 1))
    for l in range(measurements.shape[0]):
        # TODO: Compute error on manifold
        # TODO: Find the proper error equation
        e = measurements[l, :2, 3] - predicted_measurements[l, :2, 3]
        # print("e: {}".format(e))
        # measurement_likelihood[l] = distance(e / measurement_covs[l].diagonal(), (0, 0))
        measurement_likelihood[l] = distance(e / np.sqrt(measurement_covs[l].diagonal()), (0, 0))
        # measurement_inf = np.linalg.inv(measurement_covs[l])
        # measurement_likelihood[l] = np.matmul(np.matmul(e.T, measurement_inf), e)
        # print("e: {}".format(e))
        # print("d(e/cov): {}".format(distance(e / measurement_covs[l].diagonal(), (0, 0))))
        # print("d(e/sqrt(cov)): {}".format(distance(e / np.sqrt(measurement_covs[l].diagonal()), (0, 0))))
        # print("d(e.T*H*e): {}".format(np.matmul(np.matmul(e.T, measurement_inf), e))
    # print("measurement_likelihood:\n{}".format(measurement_likelihood))
    return measurement_likelihood


def nlls_estimation(initial_guess=None, args=(), output=True):
    """ TODO: docstring """

    # Setup parameters
    params = lmfit.Parameters()
    params.add('x', min=-500.0, max=500)
    params.add('y', min=-500.0, max=500)
    params.add('theta', min=0.0, max=2*np.pi)

    # Create NLLS minimizer
    minimizer = lmfit.Minimizer(residual, params, fcn_args=args)

    # Set initial guess, or else compute it with Nelder-Mead algorithm
    if initial_guess is not None:
        params['x'].value =initial_guess[0]
        params['y'].value =initial_guess[1]
        params['theta'].value =initial_guess[2]
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

def mcmc_posterior_estimation(params, args, output=False, plot=False):
    """ TODO: docstring """

    # One walker for each measurement?
    nwalkers = 2*landmarks.shape[0]
    # Run emcee
    emcee_result = lmfit.minimize(residual, method='emcee', nan_policy='omit',
                                  burn=300, steps=1000, thin=1, nwalkers=nwalkers,
                                  params=params, is_weighted=True, progress=output, args=args)

    if plot:
        emcee_plot = corner.corner(emcee_result.flatchain, labels=emcee_result.var_names,
                                   truths=list(emcee_result.params.valuesdict().values()))

    if output:
        # print("x mean: {}".format(np.mean(emcee_result.flatchain['x'])))
        # print("x median: {}".format(np.median(emcee_result.flatchain['x'])))
        # print("x std: {}".format(np.std(emcee_result.flatchain['x'])))
        # print("y mean: {}".format(np.mean(emcee_result.flatchain['y'])))
        # print("y median: {}".format(np.median(emcee_result.flatchain['y'])))
        # print("y std: {}".format(np.std(emcee_result.flatchain['y'])))
        # print("theta mean: {}".format(np.mean(emcee_result.flatchain['theta'])))
        # print("theta median: {}".format(np.median(emcee_result.flatchain['theta'])))
        # print("theta std: {}".format(np.std(emcee_result.flatchain['theta'])))

        print("Emcee Result:")
        print(emcee_result.params.pretty_print())
        # print(lmfit.fit_report(emcee_result))

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
    N = 1000

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
    nlls_result = nlls_estimation(args=(landmarks, measurements, measurement_covs),
                                  initial_guess=X, output=True)

    # Run emcee
    emcee_result = mcmc_posterior_estimation(params=nlls_result.params,
                                             args=(landmarks, measurements, measurement_covs),
                                             output=True)
    print(emcee_result.params)
    print(emcee_result.params['x'].stderr)
    # Run posterior tests to validate the posterior density estimation obtained from emcee
    # test_nlls_posterior()

    plt.show()


if __name__ == "__main__":
    main()
