import pandas as pd
import numpy as np

def process_station(time_series, threshold_count=30, series_base='flowrate', space='scott'):
    """Create a set of flowrate thresholds and extract events above them.

    Args:
        time_series (pandas df): Flowrate timeseries
        threshold_count (int, optional): Number of thresholds to analyze
        between max and min thresholds. Defaults to 30.
        series_base (str, optional): Either 'flowrate' or 'percentile'.
        This determines whether to generate the threshold grid based on
        flowrate units or flowrate percentile units. Defaults to 'flowrate'.
        space (str, optional): Can be 'linear', 'log', or 'scott'.  'scott'
        is recommended.  'linear' equally spaces the thresholds from min
        to max recorded values.  'log' spaces thresholds equally in log
        space from min to max recorded values.  'scott' performs some
        preliminary grid searches and tries to find 1) the flowrate that 
        yields the maximum number of events and 2) the maximum flowrate 
        that yields three events.  The finally thresholds are linearly 
        spaced between this min and max. Defaults to 'scott'.

    Returns:
        Pandas df with all extracted events, metadata containing list of
        flowrates analyzed and their equivalent flow precentiles.
    
    Todo: 
        Add ability to insert custom list of flowrate thresholds.
    """
    # Generate thresholds to analyze
    metadata = dict()
    time_series = time_series.astype({'datetime': 'datetime64[ns]', 'flowrate': 'float32'})
    thresholds = generate_thresholds(time_series, threshold_count, series_base, space)
    metadata['flowrates'] = thresholds['flowrate'].to_list()
    metadata['percentiles'] = thresholds['percentile'].to_list()
    metadata = pd.DataFrame(metadata)

    # Extract events from thresholds
    print('Final threshold extraction')
    events = [extract_from_threshold(t, time_series) for t in thresholds['flowrate']]
    main_df = pd.concat(events)
    
    return main_df, metadata

def generate_thresholds(series, threshold_count, series_base='percentile', space='scott'):
    """Creates a list of thresholds to analyze.

    Args:
        time_series (pandas df): Flowrate timeseries
        threshold_count (int, optional): Number of thresholds to analyze
        between max and min thresholds.
        series_base (str, optional): Either 'flowrate' or 'percentile'.
        This determines whether to generate the threshold grid based on
        flowrate units or flowrate percentile units. Defaults to 'flowrate'.
        space (str, optional): Can be 'linear', 'log', or 'scott'.  'scott'
        is recommended.  'linear' equally spaces the thresholds from min
        to max recorded values.  'log' spaces thresholds equally in log
        space from min to max recorded values.  'scott' performs some
        preliminary grid searches and tries to find 1) the flowrate that 
        yields the maximum number of events and 2) the maximum flowrate 
        that yields three events.  The finally thresholds are linearly 
        spaced between this min and max. Defaults to 'scott'.

    Returns:
        numpy array of threshold flowrates and associated flow percentiles.
    
    Todo:
        code up combinations (percentile, linear), (flowrate, log)
    """
    if series_base == 'percentile' and space == 'log':
        series = series['flowrate']
        percentiles = [1 - i for i in np.logspace(-5, 0, num=threshold_count)]
        thresholds = series.quantile(percentiles).to_frame()
        thresholds.reset_index(inplace=True)
        thresholds = thresholds.rename(columns={'index': 'percentile'})
        thresholds['flowrate'] = thresholds['flowrate'].astype(int)
        thresholds = thresholds.drop_duplicates(subset='flowrate')
    elif series_base == 'flowrate' and space == 'linear':
        series = series['flowrate']
        min = series.where(series > 0).min()
        max = series.max()
        linspace =  np.linspace(min, max, num=threshold_count+2)[1:-1]
        percentiles = np.interp(linspace, series.sort_values(), np.arange(series.size) / series.size)
        thresholds = pd.DataFrame({'percentile': percentiles, 'flowrate': linspace})
    elif series_base == 'flowrate' and space == 'scott':
        linspace = scott_thresholds(series, threshold_count)
        percentiles = np.interp(linspace, series['flowrate'].sort_values(), np.arange(series['flowrate'].size) / series['flowrate'].size)
        thresholds = pd.DataFrame({'percentile': percentiles, 'flowrate': linspace})
    return thresholds

def extract_from_threshold(threshold, series):
    """Extracts attributed events from a time-series at a given threshold"""
    print(f'-Processing flow threshold {round(threshold, 1)}')
    series['event_type'] = series['flowrate'].ge(0).astype(int)
    series['event_type'] += series['flowrate'].ge(threshold).astype(int)
    series['runs'] = (series['event_type'] - series['event_type'].shift(1)).ne(0).astype(int).cumsum()
    
    events = series.groupby(series['runs']).agg(start=('datetime', 'min'),
                                                stop=('datetime', 'max'),
                                                time_to_peak=('flowrate', np.argmax),
                                                min_flow=('flowrate', 'min'),
                                                max_flow=('flowrate', 'max'),
                                                riemann=('flowrate', 'sum'))
    del_time = (series['datetime'].iloc[1] - series['datetime'].iloc[0]).seconds / (60)
    events['time_to_peak'] *= del_time

    events['event_type'] = events.merge(series[['datetime', 'event_type']], how='left', left_on='start', right_on='datetime')['event_type'].values
    events['run_length'] = ((events['stop'] - events['start']) / pd.Timedelta('1 min')) + 15
    events['threshold'] = threshold

    return events

def scott_thresholds(series, threshold_count):
    """Generate stable list of threshold flowrates.

    Performs some preliminary grid searches and tries to find 1) the 
    flowrate that yields the maximum number of events and 2) the maximum
    flowrate that yields three events.  The finally thresholds are 
    linearly spaced between this min and max.

    Args:
        series (pandas df): flowrate timeseries
        threshold_count (int): total number of threshold flowrates

    Returns:
        numpy array of threshold flowrates
    """
    # Make an initial grid for guessing
    min = series['flowrate'].where(series['flowrate'] > 0).min() * 1.1
    max = series['flowrate'].max()
    guess_1 =  np.linspace(min, max, 10)
    # Extract from guess grid
    print('Preliminary threshold screening')
    events_1 = [extract_from_threshold(t, series) for t in guess_1]
    events_1 = pd.concat(events_1)
    # Clean guess grid
    clean_events_1 = clean_events(events_1, t_interevent=(5*24*60))
    counts = clean_events_1.query('type == 1')['threshold'].value_counts(sort=False)
    # Find best starting point, then refine grid
    max = counts.argmax()
    if max == 0:
        guess_2 =  np.linspace(counts.index[0], counts.index[1], 10)  # Todo change this to something like counts[0, counts[1]]
    else:
        guess_2 =  np.linspace(counts.index[max - 1], counts.index[max + 1], 10)
    # Extract from refined grid
    print('Refined minimum threshold screening')
    events_2 = [extract_from_threshold(t, series) for t in guess_2]
    events_2 = pd.concat(events_2)
    # Clean refined grid
    clean_events_2 = clean_events(events_2, t_interevent=(5*24*60))
    counts_2 = clean_events_2.query('type == 1')['threshold'].value_counts(sort=False)
    start_pt = guess_2[counts_2.argmax()]

    # Find best ending point, then refine grid
    if counts.min() < 3:
        gr_3 = (counts < 3).argmax()
        guess_3 =  np.linspace(counts.index[gr_3 - 1], counts.index[-1], 10)
        # Extract from refined grid
        print('Refined maximum threshold screening')
        events_3 = [extract_from_threshold(t, series) for t in guess_3]
        events_3 = pd.concat(events_3)
        # Clean from refined grid
        clean_events_3 = clean_events(events_3, t_interevent=(5*24*60))
        counts_3 = clean_events_3.query('type == 1')['threshold'].value_counts(sort=False)
        end_pt = counts_3.index[(counts_3 < 3).argmax() - 1]
    else:
        end_pt = counts.index[-1]

    return  np.linspace(start_pt, end_pt, num=threshold_count)

def events_from_arrival_rate(series, record_length, arrival_rate, t_int, min_recession_pct):
    """Generate events from a threshold with set arrival rate.

    Args:
        series (pandas df): flowrate timeseries
        record_length (float): length of flowrate record in years.
        arrival_rate (float): desired number of events per year
        t_int (float): min interevent duration required to prevent merging
        min_recession_pct (float): min recession percent required to prevent merging

    Returns:
        pandas df with flood events from threshold with desired exceedence
        frequency, pandas df with events from all thresholds analyzed.
    """
    # Ensure types correct
    series = series.astype({'datetime': 'datetime64[ns]', 'flowrate': 'float32'})

    # make an initial grid for guessing
    print('Preliminary threshold screening')
    min = series['flowrate'].where(series['flowrate'] > 0).min() * 1.1
    max = series['flowrate'].max()
    guess_1 =  np.linspace(min, max, 10)
    
    events_1 = [extract_from_threshold(t, series) for t in guess_1]
    events_1 = pd.concat(events_1)

    clean_events_1 = clean_events(events_1, t_interevent=t_int, min_recession_pct=min_recession_pct)
    clean_events_1_tmp = clean_events_1.query('type == 1')
    rates = clean_events_1_tmp['threshold'].value_counts(sort=False) / record_length
    closest_arrival_rate = abs(rates - arrival_rate).argmin()
    print(f'threshold {round(rates.index[closest_arrival_rate], 1)} closest with arrival rate {rates.iloc[closest_arrival_rate]} events per year')

    # refine grid and re-search
    print('Refined grid threshold screening')
    if closest_arrival_rate == 0:
        guess_2 = np.linspace(rates.index[0], rates.index[1], 10)
    else:
        guess_2 = np.linspace(rates.index[closest_arrival_rate - 1], rates.index[closest_arrival_rate + 1], 10)
    events_2 = [extract_from_threshold(t, series) for t in guess_2]
    events_2 = pd.concat(events_2)
    clean_events_2 = clean_events(events_2, t_interevent=t_int, min_recession_pct=min_recession_pct)
    clean_events_2_tmp = clean_events_2.query('type == 1')
    rates_2 = clean_events_2_tmp['threshold'].value_counts(sort=False) / record_length
    closest_arrival_rate = abs(rates_2 - arrival_rate).argmin()
    final_threshold = rates_2.index[closest_arrival_rate]
    print(f'threshold {round(rates_2.index[closest_arrival_rate], 1)} closest with arrival rate {rates_2.iloc[closest_arrival_rate]} events per year')
    
    all_events = pd.concat([clean_events_1, clean_events_2])
    marginal_events = all_events.query(f'threshold == {final_threshold}')
    return marginal_events, all_events

def clean_events(df, t_interevent=0, min_recession_pct=0):
    # Get a list of all event thresholds, then process by threshold
    thresholds = df['threshold'].unique()
    
    working_df = pd.DataFrame({'duration': list(), 'volume': list(), 'type': list(), 'threshold': list()})
    for t in thresholds:
        # Subset data and sort by event time
        subset = df[df['threshold'] == t].sort_values('start')
        # Calculate recession.
        subset['peak_prev'] = subset['max_flow'].shift(1)
        subset['peak_next'] = subset['max_flow'].shift(-1)
        subset['adj_peak'] = subset[['peak_prev','peak_next']].min(axis=1)
        subset['recession_pct'] = (subset['adj_peak'] - subset['min_flow']) / subset['adj_peak']
        # Filter out interevents that are too short or don't recede enough.
        removal_mask = np.logical_or((subset['run_length'] < t_interevent), (subset['recession_pct'] <= min_recession_pct))
        removal_mask = np.logical_and(removal_mask, (subset['event_type'].shift(1) == 2))  # check if left event is positive
        removal_mask = np.logical_and(removal_mask, (subset['event_type'].shift(1) == 2))  # check if right event is positive
        removal_mask = np.logical_and(removal_mask, (subset['event_type'] == 1))  # check if event is negative
        
        subset['event_type'] = subset['event_type'].where(~removal_mask, 2)
        # Combine back-to-back events of the same type
        subset.loc[subset['event_type'] == 1, 'event_type'] = 0  # From this point on, nodata events and negative events are labeled 0

        subset['run_id'] = (subset['event_type'] - subset['event_type'].shift(1)).apply(lambda x: 1 if x != 0 else 0).cumsum()
        new_events = subset.groupby(subset['run_id']).agg(start=('start', 'min'),
                                                          duration=('run_length', 'sum'),
                                                          volume=('riemann', 'sum'),
                                                          type=('event_type', lambda x: 1 if (x == 2).all() else 0),
                                                          max_ind=('max_flow', np.argmax),
                                                          time_to_peak=('time_to_peak', 'max'), # placeholder for further calcs
                                                          max_flow=('max_flow', 'max'),
                                                          min_flow=('min_flow', 'min'))

        # Sadly need to get a little inefficient here.  Calculated time to peak
        new_events['time_to_peak'] = 0
        for row, rid in enumerate(new_events.index):
            tmp_subset = subset.query(f'run_id == {rid}')
            if len(tmp_subset) == 1:
                continue
            peak_ind = new_events['max_ind'].iloc[row].item()
            cum_duration = sum(tmp_subset['run_length'].iloc[0:peak_ind])
            cum_duration += tmp_subset['time_to_peak'].iloc[peak_ind]
            new_events.loc[rid, 'time_to_peak'] = cum_duration
        
        new_events['threshold'] = t
        working_df = pd.concat([working_df, new_events])
    return working_df
