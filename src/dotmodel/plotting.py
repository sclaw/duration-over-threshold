# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib import cm
import numpy as np


def plot_marginal_fit(peaks, record_length, xi, alpha, k, save_path, ri_bounds=(0.1, 500), x_axis='ri'):
    fig = plt.figure(figsize=(6.5, 4.5), layout='tight')
    gs = fig.add_gridspec(1,1)
    marginal = fig.add_subplot(gs[0, 0])

    # Preprocess data
    n = len(peaks)
    peaks = sorted(peaks, reverse=True)
    plotting_positions = [(record_length) / (i + 1) for i in range(n)]

    recurrence_intervals = np.linspace(ri_bounds[0], ri_bounds[1], 1000)
    flow_space = xi + ((alpha * (1 - ((1 / recurrence_intervals) ** k))) / k)
    recurrence_intervals = recurrence_intervals[flow_space > 0]
    flow_space = flow_space[flow_space > 0]

    marginal.plot(recurrence_intervals, flow_space, alpha=0.95, zorder=12, c='royalblue', label='PDS Parametric Arrival Rate')
    marginal.scatter(plotting_positions, peaks, alpha=0.8, ec='lightcoral', fc='none', s=8, zorder=11, label='PDS Empirical Arrival Rate')
    marginal.set_xscale('log')
    if x_axis == 'ri':
        marginal.set_xlabel('Arrival Rate (years)')
    elif x_axis == 'aep':
        marginal.set_xlabel('AEP (%)')
    marginal.set_ylabel('Discharge (cfs)')
    plt.legend()
    if x_axis == 'aep':
        plt.gca().invert_xaxis()
    
    plt.grid(which='both', zorder=0)
    fig.savefig(save_path)
    plt.close(fig)

def plot_conditional_fit(events, misc_data, save_path):
    # Preprocess data
    positive_events = events.query('type == 1')
    positive_summary = positive_events.groupby(positive_events['threshold']).agg(mean_duration=('duration', 'mean'),
                                                                        median_duration=('duration', 'median'),
                                                                        count=('duration', 'count'))
    thresholds = positive_summary.index

    power_a = misc_data['power_a'].item()
    power_b = misc_data['power_b'].item()
    smearing = misc_data['smearing'].item()
    power_predictions = smearing * power_a * (thresholds ** power_b)

    # Plot
    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 5]}, figsize=(7, 4.5))

    axs[0].plot(thresholds, positive_summary['count'], color='k')
    axs[0].set_ylabel('Event Count')
    axs[0].set_yticks([min(positive_summary['count']), max(positive_summary['count'])])
    axs[0].set_xscale('log')

    axs[1].scatter(positive_events['threshold'], positive_events['duration'], c='k', s=5, alpha=0.1, label='Events')
    axs[1].scatter(thresholds, positive_summary['mean_duration'], marker='o', fc='w', ec='slateblue', s=25, label='Mean')
    axs[1].scatter(thresholds, positive_summary['median_duration'], marker='X', fc='w', ec='slateblue', s=25, label='Median')

    axs[1].plot(thresholds, power_predictions, c='gray', alpha=0.8, ls='dashed', label='power law fit', zorder=11)
    axs[1].text(0.05, 0.1, f"Power Law Constants: A = {int(power_a)}; B = {round(power_b, 4)}", c='gray', ha='left', alpha=0.8, transform=axs[1].transAxes)
    
    axs[1].set_xlabel('Event Threshold (cfs)')
    axs[1].set_ylabel('Event Duration (Minutes)')
    axs[1].set_ylim(10, 1.3 * (10 ** 6))
    axs[1].set_yscale('log')

    secax = axs[1].secondary_yaxis('right', functions=(lambda x: x / (60 * 24), lambda x: x * (60 * 24)))
    tick_list = [0.1, 1, 10]
    secax.yaxis.set_major_locator(mticker.FixedLocator(tick_list))
    secax.set_yticklabels(tick_list)
    secax.set_ylabel('Event Duration (Days)')

    site_no = misc_data['site_no'].item()
    name = misc_data['station_nm'].item()
    da = round(float(misc_data['drain_area_va'].item()), 1)
    axs[0].set_title(f'{site_no} | {name} | {da} sqmi', loc='left')
    axs[1].legend(loc='upper right', fontsize='xx-small')
    plt.tight_layout()

    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_model_contour(a, b, xi, alpha, k, resolution=3000, plot_live=False, save_path=None, units='imperial'):
    # set up interpolation grid
    max_q = ((1 - (0.002 ** k)) * (alpha / k)) + xi  # flow at 500-yr event
    max_q *= 1.05
    min_q = ((1 - (6 ** k)) * (alpha / k)) + xi  # flow at 0.25-yr event
    min_q = max(min_q, 0)

    q1 = ((1 - (1 ** k)) * (alpha / k)) + xi
    max_d = np.log(0.002 ** 2) * (-a * np.power(q1, b))  # 500 year event duration at 1-yr flowrate

    q_space = np.linspace(min_q, max_q, resolution)
    d_space = np.linspace(10, max_d, resolution)
    q, d = np.meshgrid(q_space, d_space)

    # calculate arrival rates
    marginal_rates = (1 - ((q - xi) * (k / alpha))) ** (1 / k)
    full_lambda = marginal_rates * np.exp(d / (-a * np.power(q, b)))
    full_lambda[full_lambda == 0] = np.nan
    full_lambda[full_lambda < 0.0001] = np.nan
    full_lambda = 1 / full_lambda

    # rescale d
    rescale = (60 * 24)
    d = d / rescale

    # rescale q
    if units == 'imperial':
        y_label = 'Flowrate (cfs)'
    elif units == 'metric':
        q /= 35.3147
        y_label = 'Flowrate (cms)'

    fig, ax = plt.subplots()
    levels = {0.2: '0.2', 1: '1', 2:'2', 5:'5', 10:'10', 25:'25', 50:'50', 100:'100', 200:'200', 500:'500'}
    min_level = np.nanmin(full_lambda)
    levels = {l[0]:l[1] for l in levels.items() if l[0] > min_level}
    c = ax.contour(d, q, full_lambda, levels=[i for i in levels.keys()], colors='slateblue')
    for contour, level in zip(c.allsegs, levels):
        x0 = contour[0][-1, 0]
        y0 = contour[0][-1, 1]
        y = np.interp(20 / rescale, contour[0][:, 0][::-1], contour[0][:, 1][::-1])
        ax.text(20 / rescale, y, level, va='center', ha='center', size=8, path_effects=[pe.withStroke(linewidth=4, foreground="whitesmoke")])

    custom_line = Line2D([0], [0], color='slateblue', lw=2)
    fig.legend(handles=[custom_line], labels=['Recurrence Interval (yrs)'], bbox_to_anchor=(0.97, 0.96))

    ax.set_ylabel(y_label)
    ax.set_xlabel('Duration (days)')
    ax.set_xlim(10 / rescale, max_d / rescale)
    ax.set_xscale('log')
    ax.set_facecolor("whitesmoke")
    ax.grid(c='k')
    
    fig.tight_layout()
    if plot_live:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.close(fig)

def plot_model_3d(a, b, xi, alpha, k, resolution=3000, plot_live=False, save_path=None, units='imperial'):
    # set up interpolation grid
    max_q = ((1 - (0.002 ** k)) * (alpha / k)) + xi  # flow at 500-yr event
    max_q *= 1.05
    min_q = ((1 - (6 ** k)) * (alpha / k)) + xi  # flow at 0.25-yr event
    min_q = max(min_q, 0)

    q1 = ((1 - (1 ** k)) * (alpha / k)) + xi
    max_d = np.log(0.002) * (-a * np.power(q1, b))  # 500 year event duration at 1-yr flowrate

    q_space = np.linspace(min_q, max_q, resolution)
    d_space = np.linspace(10, max_d, resolution)
    q, d = np.meshgrid(q_space, d_space)

    # calculate arrival rates
    marginal_rates = (1 - ((q - xi) * (k / alpha))) ** (1 / k)
    full_lambda = marginal_rates * np.exp(d / (-a * np.power(q, b)))
    full_lambda[full_lambda == 0] = np.nan
    full_lambda[full_lambda < 0.001] = np.nan
    full_lambda = 1 / full_lambda

    # rescale d
    rescale = (60 * 24)
    d = d / rescale

    # rescale q
    if units == 'imperial':
        y_label = 'Flowrate (cfs)'
    elif units == 'metric':
        q /= 35.3147
        y_label = 'Flowrate (cms)'

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(d, q, full_lambda, cmap=cm.plasma, linewidth=0, antialiased=False, zorder=1, rcount=70, ccount=70)

    ax.set_ylabel(y_label)
    ax.set_xlabel('Duration (days)')
    ax.set_zlabel('Arrival Rate (years)')
    ax.view_init(azim=225)
    
    fig.tight_layout()
    if plot_live:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.close(fig)

def plot_timeseries(data, misc_data, save_path):
    fig, ax = plt.subplots(figsize=(35, 6))

    ax.grid(axis='x')
    ax.set_xlabel('Time')
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1, month=1, day=1))
    ax.set_ylabel('Flowrate (cfs)')

    ax.plot(data['datetime'], data['flowrate'], lw=0.5, c='k', zorder=2, label='instantaneous flowrate')

    site_no = misc_data['site_no'].item()
    name = misc_data['station_nm'].item()
    da = round(float(misc_data['drain_area_va'].item()), 1)

    ax.set_title(f'{site_no} | {name} | {da} sqmi', loc='left')
    ax.legend(markerscale=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def exponential_check(events, misc_data, save_path):
    site_no = misc_data['site_no'].item()
    name = misc_data['station_nm'].item()
    da = round(float(misc_data['drain_area_va'].item()), 1)
    title = f'{site_no} | {name} | {da} sqmi | QQ Plot for Exponential Distribution'

    thresholds = events['threshold'].unique()

    nrows = int(np.floor(len(thresholds) ** 0.5))
    ncols = int(len(thresholds) / nrows)

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95, wspace=0.15, hspace=0.15)
    fig.set_figheight(8.5)
    fig.set_figwidth(11)
    fig.text(0.5, 0.05, 'Theoretical Distribution Quantile', ha='center', va='center')
    fig.text(0.05, 0.5, 'Experimental Quantile', ha='center', va='center', rotation='vertical')
    fig.suptitle(title)

    row_counter = 0
    for ind, thresh in enumerate(thresholds):
        col = ind - (row_counter * ncols)
        thresh_events = events.query(f'threshold == {thresh}')
        tmp_events = sorted(thresh_events['duration'].to_numpy())
        sample_quantiles = [i / len(tmp_events) for i in range(len(tmp_events))]
        sample_mean = sum(tmp_events) / len(tmp_events)
        sample_var = np.sqrt(np.sum(np.square(np.array(tmp_events) - sample_mean)) / (len(tmp_events) - 1))

        theoretical_quantiles = 1 - np.exp(-np.array(tmp_events) * (1 / sample_mean))

        event_month = thresh_events['start'].astype('datetime64[ns]').dt.month
        spring = ['darkorange' if m > 1 and m < 6 else 'slateblue' for m in event_month]
        spring = [x for _, x in sorted(zip(thresh_events['duration'].to_numpy(), spring))]

        mean_var_text = r'$\widehat{\mu} / \sqrt{S}$' + f' = {round(sample_mean / sample_var, 2)}'
        all_text = f'{mean_var_text}'

        axs[row_counter, col].set_facecolor("whitesmoke")

        axs[row_counter, col].text(0, 1, all_text, c='k', ha='left', va='top', size=6.5)
        axs[row_counter, col].text(1, 0, f'{int(thresh)}cfs', c='k', ha='right', va='bottom', size=15, alpha=0.8)

        axs[row_counter, col].scatter(theoretical_quantiles, sample_quantiles, fc=spring, s=4, alpha=0.8)
        axs[row_counter, col].plot([0, 1], [0, 1], ls='dashed', c='k', alpha=0.3)
        
        if col == (ncols - 1):
            row_counter += 1
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Non-Spring Events', markerfacecolor='slateblue', markersize=5),
                       Line2D([0], [0], marker='o', color='w', label='Spring Events', markerfacecolor='darkorange', markersize=5)]
    axs[-1,-1].legend(handles=legend_elements, ncols=2, bbox_to_anchor=(1, -0.2))
    
    fig.savefig(save_path, dpi=300)
    plt.close(fig)