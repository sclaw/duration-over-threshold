import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
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
    axs[1].set_ylim(10, 1.3 * (10 ** 5))
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

def plot_model_contour(a, b, xi, alpha, k, resolution=3000, plot_live=False, save_path=None):
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

    ax.set_ylabel('Flowrate (cfs)')
    ax.set_xlabel('Duration (days)')
    ax.set_xlim(10 / rescale, max_d / rescale)
    ax.set_xscale('log')
    ax.set_facecolor("whitesmoke")
    ax.grid(c='k')
    fig.tight_layout()
    
    fig.tight_layout()
    if plot_live:
        plt.show()
    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.close(fig)

