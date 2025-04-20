def print_results(results):
    print('BEST PARAMS: {}\n'.format(results.best_params_))

    means = results.cv_results_['mean_test_score']
    stds = results.cv_results_['std_test_score']
    params =  results.cv_results_['params']
    for mean, std, param_dict in zip(means, stds, params):
        print('{} (+/-{}) for {}'.format(round(mean, 3), round(std * 2, 3), param_dict))