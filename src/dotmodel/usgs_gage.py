# -*- coding: utf-8 -*-
"""Code to read/write from SQL database and control model-fitting functions

This class automates much of SQL database interaction and function calling
involved in fitting the Duration-Over-Threshold model to USGS gage data.

For examples, see simple_example.py at 
https://github.com/sclaw/duration-over-threshold/tree/main/samples/simple_example

"""
import sqlite3
import pandas as pd
import numpy as np
from dotmodel import nwis, preprocess_time_series, extract_events, model, plotting


class Gage:

    def __init__(self, path: str) -> None:
        """Establish connection with an SQL database.

        Args:
            path (str): Path to database.  Will create if it does not exist.
        """
        self.path = path
        self.connection = sqlite3.connect(self.path)

    def download_nwis(self, site_no: str, water_new_year: str=None, begin_date: str=None, end_date: str=None):
        """Downloads site metadata, instantaneous data, and mean daily values.

        Args:
            site_no (str): USGS site identification number
            water_new_year (str, optional): Date of water new year 
            Typically '10-01'.  This will trim the period of record to
            the longest available full years of record. Defaults to None.
            begin_date (str, optional): Optional hard code begin date.
            Format should be 'MM-DD-YYYY'. Defaults to None.
            end_date (atr, optional): Optional hard code end date.
            Format should be 'MM-DD-YYYY'. Defaults to None.
        """
        misc, dv, iv = nwis.download_gage_date(site_no, water_new_year, begin_date, end_date)
        misc.to_sql('miscellaneous', self.connection, index=False, if_exists='replace')
        dv.to_sql('mean_dv_timeseries', self.connection, index=False, if_exists='replace')
        iv.to_sql('raw_timeseries', self.connection, index=False, if_exists='replace')

    def clean_timeseries(self):
        """Fills and imputes missing data gaps from flowrate timeseries.

        See full academic journal article for details of how this is
        accomplished.  Improves model fitting stability.
        """
        iv_data = pd.read_sql_query('SELECT datetime, flowrate FROM raw_timeseries', self.connection)
        dv_data = pd.read_sql_query('SELECT datetime, flowrate FROM mean_dv_timeseries', self.connection)

        filled_timeseries, _ = preprocess_time_series.fill_timeseries(iv_data)
        imputed_timeseries, _ = preprocess_time_series.impute_daily_peak(filled_timeseries, dv_data)
        filled_timeseries.to_sql('filled_timeseries', self.connection, if_exists='replace', index=False)
        imputed_timeseries.to_sql('imputed_timeseries', self.connection, if_exists='replace', index=False)

    def flow_frequency_analysis(self, truncation_arrival_rate: float, t_int='da', min_recession: float=0.75):
        """Performs a Peaks-Over-Threshold flow frequency analysis

        Args:
            truncation_arrival_rate (float): Desired arrival rate for 
            truncation threshold in floods/year
            t_int (float): min interevent duration required to prevent 
            merging.  Default of 'da' will set the value to 5+log(DA)
            min_recession_pct (float): min recession percent required to
            prevent merging.  Defaults to 0.75
        """
        imputed_timeseries = pd.read_sql_query('SELECT datetime, flowrate FROM imputed_timeseries', self.connection)
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)

        record_length = len(pd.date_range(start=misc_data['begin_date'][0],end=misc_data['end_date'][0],freq='Y'))
        if t_int == 'da':
            da = float(misc_data['drain_area_va'][0])
            t_int = 5 + np.log(da)

        frequency_events, _ = extract_events.events_from_arrival_rate(imputed_timeseries, record_length, arrival_rate=truncation_arrival_rate, t_int=t_int, min_recession_pct=min_recession)
        frequency_events.to_sql('marginal_events', self.connection, if_exists='replace', index=False)

        xi_gev, alpha_gev, k_gev = model.fit_marginal(frequency_events, record_length)
        misc_data['xi'], misc_data['alpha'], misc_data['k'] = xi_gev, alpha_gev, k_gev
        misc_data.to_sql('miscellaneous', self.connection, if_exists='replace', index=False)


    def duration_analysis(self, threshold_count: int=30, t_int='da', min_recession: float=0.75):
        """Fits a power law to the flowrate vs mean event duration relationship.

        Args:
            threshold_count (int, optional): Number of thresholds to
            analyze between min and max thresholds. Defaults to 30.
            t_int (float): min interevent duration required to prevent 
            merging.  Default of 'da' will set the value to 5+log(DA)
            min_recession_pct (float): min recession percent required to
            prevent merging.  Defaults to 0.75
        """
        filled_timeseries = pd.read_sql_query('SELECT * FROM filled_timeseries', self.connection)
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)
        if t_int == 'da':
            da = float(misc_data['drain_area_va'][0])
            t_int = 5 + np.log(da)

        events, _ = extract_events.process_station(filled_timeseries, 
                                                        threshold_count=threshold_count, 
                                                        series_base='flowrate', 
                                                        space='scott')

        cleaned_events = extract_events.clean_events(events, t_interevent=t_int, min_recession_pct=min_recession)
        cleaned_events.to_sql('clean_events', self.connection, if_exists='replace', index=False)
        
        power_fit = model.fit_conditional(cleaned_events)
        misc_data['power_a'] = power_fit['fit'][1]
        misc_data['power_b'] = power_fit['fit'][0]
        misc_data['smearing'] = power_fit['smearing']
        misc_data.to_sql('miscellaneous', self.connection, if_exists='replace', index=False)

    def plot_model_summary(self, save_path, type='contour', units='imperial'):
        """Makes contour plot of flowrate, duration, and recurrence interval

        Args:
            save_path (str): Path to save output
        """
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)
        a = misc_data['power_a'].item()
        b = misc_data['power_b'].item()
        xi = misc_data['xi'].item()
        alpha = misc_data['alpha'].item()
        k = misc_data['k'].item()
        e = misc_data['smearing'].item()
        a *= e

        if type == 'contour':
            plotting.plot_model_contour(a, b, xi, alpha, k, save_path=save_path, units=units)
        elif type == '3d':
            plotting.plot_model_3d(a, b, xi, alpha, k, save_path=save_path, units=units)

    def plot_flow_frequency(self, save_path):
        """Makes plot of flowrate vs recurrence interval

        Args:
            save_path (str): Path to save output
        """
        peaks = pd.read_sql_query('SELECT * FROM marginal_events', self.connection).query('type == 1')['max_flow']
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)
        record_length = len(pd.date_range(start=misc_data['begin_date'][0],end=misc_data['end_date'][0],freq='Y'))
        xi = misc_data['xi'].item()
        alpha = misc_data['alpha'].item()
        k = misc_data['k'].item()

        plotting.plot_marginal_fit(peaks, record_length, xi, alpha, k, save_path=save_path)

    def plot_duration_power_law(self, save_path):
        """Makes plot relating flowrate and mean event duration

        Args:
            save_path (str): Path to save output
        """
        events = pd.read_sql_query('SELECT * FROM clean_events', self.connection)
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)

        plotting.plot_conditional_fit(events, misc_data, save_path)

    def plot_timeseries(self, series, save_path):
        """Plots entire timeseries

        Args:
            series (str): Which time series to plot. Can be raw_timeseries,
            mean_dv_timeseries, filled_timeseries, or imputed_timeseries
            save_path (str): Path to save output
        """
        ts = pd.read_sql_query(f'SELECT * FROM {series}', self.connection)
        ts = ts.astype({'datetime': 'datetime64[ns]', 'flowrate': 'float32'})
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)

        plotting.plot_timeseries(ts, misc_data, save_path)

    def plot_exponential_durations(self, save_path):
        """Makes QQ plots for event durations at each threshold.

        Args:
            save_path (str): Path to save output
        """
        clean_events = pd.read_sql_query('SELECT * FROM clean_events', self.connection)
        clean_events = clean_events.query(f'type == 1')
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)

        plotting.exponential_check(clean_events, misc_data, save_path)
