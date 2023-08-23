import sqlite3
import pandas as pd
import numpy as np
from dotmodel import nwis, preprocess_time_series, extract_events, model, plotting


class Gage:

    def __init__(self, path: str) -> None:
        self.path = path
        self.connection = sqlite3.connect(self.path)

    def download_nwis(self, site_no: str, water_new_year: str=None, begin_date: str=None, end_date: str=None):
        self.site_no = site_no
        misc, dv, iv = nwis.download_gage_date(site_no, water_new_year, begin_date, end_date)
        misc.to_sql('miscellaneous', self.connection, index=False, if_exists='replace')
        dv.to_sql('mean_dv_timeseries', self.connection, index=False, if_exists='replace')
        iv.to_sql('raw_timeseries', self.connection, index=False, if_exists='replace')

    def clean_timeseries(self):
        iv_data = pd.read_sql_query('SELECT datetime, flowrate FROM raw_timeseries', self.connection)
        dv_data = pd.read_sql_query('SELECT datetime, flowrate FROM mean_dv_timeseries', self.connection)

        filled_timeseries, _ = preprocess_time_series.fill_timeseries(iv_data)
        imputed_timeseries, _ = preprocess_time_series.impute_daily_peak(filled_timeseries, dv_data)
        filled_timeseries.to_sql('filled_timeseries', self.connection, if_exists='replace', index=False)
        imputed_timeseries.to_sql('imputed_timeseries', self.connection, if_exists='replace', index=False)

    def flow_frequency_analysis(self, truncation_arrival_rate: float, t_int='da', min_recession: float=0.75):
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

    def plot_model_summary(self, save_path):
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)
        a = misc_data['power_a'].item()
        b = misc_data['power_b'].item()
        xi = misc_data['xi'].item()
        alpha = misc_data['alpha'].item()
        k = misc_data['k'].item()
        e = misc_data['smearing']

        plotting.plot_model_contour(a, b, xi, alpha, k, save_path=save_path)

    def plot_flow_frequency(self, save_path):
        peaks = pd.read_sql_query('SELECT * FROM marginal_events', self.connection).query('type == 1')['max_flow']
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)
        record_length = len(pd.date_range(start=misc_data['begin_date'][0],end=misc_data['end_date'][0],freq='Y'))
        xi = misc_data['xi'].item()
        alpha = misc_data['alpha'].item()
        k = misc_data['k'].item()

        plotting.plot_marginal_fit(peaks, record_length, xi, alpha, k, save_path=save_path)

    def plot_duration_power_law(self, save_path):
        events = pd.read_sql_query('SELECT * FROM clean_events', self.connection)
        misc_data = pd.read_sql_query('SELECT * FROM miscellaneous', self.connection)

        plotting.plot_conditional_fit(events, misc_data, save_path)