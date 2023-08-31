# -*- coding: utf-8 -*-
from math import comb
import numpy as np

def l_moments_gpd(series, threshold):
    ### From Stedinger 1993 around page 7 (section 18.6.1) ###
    n = len(series)
    b_values = list()
    # This method could be streamlined, but as-is, it's easily debugged
    for r in range(4):
        running_sum = list()
        for j in range(n-r):
            cur = (comb(n - (j + 1), r) * series[j]) / comb(n - 1, r)
            running_sum.append(cur)
        running_sum = sum(running_sum)
        b = running_sum / n
        b_values.append(b)
    
    l1 = b_values[0]
    l2 = (2 * b_values[1]) - b_values[0]

    k_gpd = ((l1 - threshold) / l2) - 2
    alpha_gpd = (l1 - threshold) * (k_gpd + 1)

    return k_gpd, alpha_gpd

def gpd_to_gev(k_gpd, alpha_gpd, arrival_lambda, threshold):
    xi_gev = threshold + ((alpha_gpd * (1 - (arrival_lambda ** -k_gpd))) / k_gpd)
    alpha_gev = alpha_gpd * (arrival_lambda ** -k_gpd)
    k_gev = k_gpd
    return xi_gev, alpha_gev, k_gev


def fit_marginal(events, record_length):
    # Fit Pareto
    peaks = events.query(f'type == 1')['max_flow']
    threshold = events['threshold'].iloc[0]
    n = len(peaks)
    arrival_rate = n / record_length
    series = sorted(peaks, reverse=True)
    k_gpd, alpha_gpd = l_moments_gpd(series, threshold)
    
    # Convert to GEV
    xi_gev, alpha_gev, k_gev = gpd_to_gev(k_gpd, alpha_gpd, arrival_rate, threshold)
    return xi_gev, alpha_gev, k_gev

def fit_conditional(events):
    # Preprocess events
    positive_events = events.query('type == 1')
    positive_summary = positive_events.groupby(positive_events['threshold']).agg(mean_duration=('duration', 'mean'),
                                                                        median_duration=('duration', 'median'),
                                                                        mean_volume=('volume', 'mean'),
                                                                        median_volume=('volume', 'median'),
                                                                        count=('duration', 'count'))
    thresholds = positive_summary.index
    mean_durations = positive_summary['mean_duration']
        
    # Fit power law
    power_fit, power_residuals, *_ = np.polyfit(np.log(thresholds), np.log(mean_durations), deg=1, full=True)
    power_fit[1] = np.exp(power_fit[1])
    power_predictions = power_fit[1] * (thresholds ** power_fit[0])
    exp_residuals = np.exp(np.log(mean_durations.to_numpy()) - np.log(power_predictions.to_numpy()))
    smearing = np.mean(exp_residuals)
    power_dict = {'fit': power_fit, 'residuals': power_residuals, 'predictions': power_predictions, 'smearing': smearing}

    return power_dict
