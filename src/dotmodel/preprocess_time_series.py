# -*- coding: utf-8 -*-
"""Tools for filling missing data from NWIS.
"""
import pandas as pd
import numpy as np

 
def fill_timeseries(data):
    """Forward fill all data gaps shorter than one day.

    Args:
        data (pandas df): timeseries from nwis.py

    Returns:
        Forward filled data (pandas df), diagnostic report on missing days
    """
    data = data.drop_duplicates(['datetime'])
    data = data.astype({'datetime': 'datetime64[ns]', 'flowrate': 'float32'})
    data = data.set_index('datetime')

    # Resample and cache time labels
    data = data.resample('15min').mean(numeric_only=True)
    data['datetime'] = data.index

    # find na sections, mark continuous runs, and attribute
    data['na'] = data.flowrate.isna()
    data['runs'] = data['na'].diff().cumsum() * data['na']
    events = data[data['na'] == True].groupby(data['runs']).agg(start=('datetime', 'min'), stop=('datetime', 'max'))
    events = events.merge(data[['datetime', 'runs']], how='left', left_on='start', right_index=True)[['start', 'stop', 'runs']]
    events['length(min)'] = (events['stop'] - events['start']) / pd.Timedelta('1 min')

    # Impute averages.  Blind to whether these are short periods or long periods
    data['flowrate'] = data['flowrate'].ffill()

    # Go back and set long periods to -1
    events['long'] = events['length(min)'] > (24 * 60)
    subset = events.query('long == True')
    run_list = subset['runs'].unique()
    data.loc[data['runs'].isin(run_list), 'flowrate'] = -1

    # Generate diagnostic data
    diagnostic_meta = dict()

    subset = events.query('long == True')
    diagnostic_meta['long_years'] = subset['start'].dt.year.to_list()
    diagnostic_meta['long_months'] = subset['start'].dt.month.to_list()
    diagnostic_meta['long_durations'] = subset['length(min)'].to_list()

    subset = events.query('long == False')
    diagnostic_meta['short_years'] = subset['start'].dt.year.to_list()
    diagnostic_meta['short_months'] = subset['start'].dt.month.to_list()
    diagnostic_meta['short_durations'] = subset['length(min)'].to_list()

    # Clean for export.  updated 3/27/23 may be buggy
    data = data[['flowrate']]
    data = data.reset_index()
    return data, diagnostic_meta


def impute_daily_peak(iv_data, dv_data):
    """Estimate daily max flowrate using linear regression.

    Args:
        iv_data (pandas df): instantaneous value time-series data from nwis
        dv_data (pandas df): daily mean value time-series data from nwis

    Returns:
        Imputed dataframe, estimated daily peak series, r2 value on imputation
        regression.
    """
    # Load data and reformat
    iv_data = iv_data.astype({'datetime': 'datetime64[ns]', 'flowrate': 'float32'})
    dv_data = dv_data.astype({'datetime': 'datetime64[ns]', 'flowrate': 'float32'})
    iv_data['date'] = iv_data['datetime'].dt.date
    dv_data['date'] = dv_data['datetime'].dt.date

    # Match iv and dv to make training dataset
    iv_data_train = iv_data.query('flowrate != -1')
    iv_day_summary = iv_data_train.groupby(iv_data_train['date']).agg(day_max_flowrate=('flowrate', 'max'))
    iv_data_train = None
    iv_day_summary['date_col'] = pd.to_datetime(iv_day_summary.index).date
    iv_day_summary = iv_day_summary.merge(dv_data, how='left', left_on='date_col', right_on='date')

    x_data = iv_day_summary['flowrate'].to_numpy().reshape(-1, 1)
    y_data = iv_day_summary['day_max_flowrate']

    # Regression
    m = np.linalg.lstsq(x_data, y_data, rcond=None)[0][0]
    r2 = 1 - (np.sum(np.square(y_data - (x_data[:, 0] * m))) / np.sum(np.square(y_data - np.mean(y_data))))

    # Make prediction dataset
    iv_data_predict = iv_data.query('flowrate == -1').copy()

    iv_day_summary = iv_data_predict.groupby(iv_data_predict['date']).agg(nvals=('flowrate', 'count'), mean_iv=('flowrate', 'mean'))
    iv_day_summary['percent_missing_day'] = (iv_day_summary['nvals'] / ((24*60) / 15)) * 100
    iv_day_summary = iv_day_summary.merge(dv_data, how='left', left_on='date', right_on='date')

    # Prediction
    iv_day_summary['q_peak'] = iv_day_summary['flowrate'] * m

    # Impute prediction in iv data
    iv_data = iv_data.merge(iv_day_summary, how='left', left_on='date', right_on='date')

    iv_data.loc[iv_data["flowrate_x"] == -1, "flowrate_x"] = iv_data['flowrate_y']
    iv_data = iv_data.rename(columns={'flowrate_x': 'flowrate', 'datetime_x': 'datetime'})

    return iv_data[['datetime', 'flowrate']], {'imputed_dvs': iv_day_summary['flowrate'], 'r2': [r2]}

