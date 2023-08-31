"""Functions related to downloading data from USGS NWIS REST API.

Raises:
    RuntimeError: Non-200 API response code
"""
import os
import csv
from datetime import datetime, timedelta
import sqlite3
import requests
import pandas as pd


class SiteService:
    """REST session to access NWIS site services.
    """

    def __init__(self):
        self.session = requests.Session()

    def query_record_lengths(self, service_type, state=None, site_no=None, 
                             status='all', format='rdb', site_type='ST', 
                             param_code='00060'):
        """Search NWIS for length of flowrate records within a state.

        Args:
            service_type (str): 'iv' or 'dv'
            state (str): Two-letter state abbreviation.
            site_no (str): USGS site or site numbers.
            status: Site status.  See NWIS. Defaults to 'all'.
            format (str, optional): NWIS output format. Defaults to 'rdb'.
            site_type (str, optional): NWIS gage type. Defaults to 'ST'.
            param_code (str, optional): NWIS parameter code. Defaults to '00060'.

        Raises:
            RuntimeError: if API call returns non-200, raises runtime error

        Returns:
            Pandas dataframe with all data.
        """
        if site_no is None:
            filter = f'stateCd={state}'
        elif state is None:
            filter = f'sites={site_no}'
        query_url = f'https://waterservices.usgs.gov/nwis/site/?format={format}&{filter}&outputDataTypeCd={service_type}&parameterCd={param_code}&siteType={site_type}&siteStatus={status}&hasDataTypeCd={service_type}'
        r = self.session.get(query_url)
        if r.status_code != 200:
            raise RuntimeError(f'Response status code = {r.status_code}')
        header, data = self.parse_rdb(r.text)
        data = data[data['parm_cd'] == param_code]
        return data

    def query_site_data_expanded(self, state=None, site_no=None):
        """Get drainage area from NWIS (could be used for more in future).

        Args:
            state (str): Two-letter state abbreviation.
            site_no (str): USGS site or site numbers.

        Raises:
            RuntimeError: if API call returns non-200, raises runtime error.

        Returns:
            Pandas dataframe with all data.
        """
        if site_no is None:
            query_url = f'https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={state}&siteStatus=all&siteOutput=expanded'
        elif state is None:
            query_url = f'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={site_no}&siteStatus=all&siteOutput=expanded'
        r = self.session.get(query_url)
        if r.status_code != 200:
            raise RuntimeError(f'Response status code = {r.status_code}')
        header, data = self.parse_rdb(r.text)
        data = data[['site_no', 'drain_area_va']]
        return data

    def parse_rdb(self, in_text):
        """Convert USGS rdb to pandas dataframe."""
        split_text = in_text.split('\n')
        header = [r.replace('#', '') for r in split_text if '#' in r]
        data = [r for r in split_text if '#' not in r]
        parsed_data = list(csv.reader(data, delimiter='\t'))
        dataframe = pd.DataFrame(parsed_data[2:-1], columns=parsed_data[0])
        return header, dataframe


class ValueService:
    """REST session to access NWIS value services.
    """

    def __init__(self):
        self.session = requests.Session()

    def query_record(self, site_no, start_date, end_date, service_type=None, 
                     save_path=None, parameter_code='00060', format='rdb'):
        """Downloads flowrate data from NWIS for a certain site and timefreame.

        Args:
            site_no: USGS gage number
            start_date: First date to query. "MM/DD/YYYY"
            end_date: Last date to query. "MM/DD/YYYY"
            service_type: 'iv' or 'dv'. Defaults to None.
            save_path: Path for .csv file to save to. Defaults to None.
            parameter_code: USGS parameter code. Defaults to '00060'.
            format: Output format. Defaults to 'rdb'.

        Returns:
            A pandas dataframe with the requested flowrate data.
        """
        # break up request by year to improve performance
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
        date_breaks = list()
        tmp_start = start_date
        while (end_date - tmp_start).days > 365:
            tmp_end = tmp_start.replace(tmp_start.year + 1) - timedelta(days=1)
            date_breaks.append((datetime.strftime(tmp_start, '%Y-%m-%d'), 
                                datetime.strftime(tmp_end, '%Y-%m-%d')))
            tmp_start = tmp_end + timedelta(days=1)
        date_breaks.append((datetime.strftime(tmp_start, '%Y-%m-%d'), 
                            datetime.strftime(end_date, '%Y-%m-%d')))

        df_list = list()

        for date_range in date_breaks:
            print(f'-Querying date range {date_range[0]} to {date_range[1]}')
            query_url = f'https://waterservices.usgs.gov/nwis/{service_type}/?format={format}&sites={site_no}&startDT={date_range[0]}&endDT={date_range[1]}&parameterCd={parameter_code}&siteStatus=all'
            r = self.session.get(query_url)
            header, data = self.parse_rdb(r.text)
            if 'No sites found matching all criteria' in header[0]:
                continue
            df_list.append(data)

        full_df = pd.DataFrame(pd.concat(df_list))
        if service_type == 'iv':
            full_df.columns.values[4] = "flowrate"
            full_df.columns.values[5] = "approval"
        else:
            full_df.columns.values[3] = "flowrate"
            full_df.columns.values[4] = "approval"
        full_df = full_df[['datetime', 'flowrate', 'approval']]
        full_df = full_df.astype({'datetime': 'datetime64[ns]', 
                                  'flowrate': 'float32', 
                                  'approval': 'object'})

        if save_path:
            full_df.to_csv(save_path, index=False)
        return full_df

    def parse_rdb(self, in_text):
        """Convert USGS rdb to pandas dataframe."""
        split_text = in_text.split('\n')
        header = [r.replace('#', '') for r in split_text if '#' in r]
        data = [r for r in split_text if '#' not in r]
        parsed_data = list(csv.reader(data, delimiter='\t'))
        dataframe = pd.DataFrame(parsed_data[2:-1], columns=parsed_data[0])
        return header, dataframe


def generate_batch(state, out_path, min_record_length=None, water_new_year=None):
    """Query NWIS for a list of gages within a state.

    Args:
        state: Two letter abbreviation for state of interest.
        out_path: Path to csv where list should be saved.
        min_record_length (int, optional): Minimum length of record to 
        download in years. Defaults to None.
        water_new_year (str): Date of water NY.  Typically '10-01'
        specific date. Defaults to None.
    """
    print(f'Generating batch for {state}')
    # Download metadata
    service = SiteService()
    record_lengths = service.query_record_lengths('iv', state=state)
    site_data = service.query_site_data_expanded(state)
    combo = record_lengths.merge(site_data, on='site_no')

    # Subset records
    if water_new_year is not None:
        combo = trim_record_to_water_new_year(combo, water_new_year)
    if min_record_length is not None:
        combo = combo[combo['count_nu'] >= (365 * min_record_length)]

    combo.to_csv(out_path, index=False)


def batch_download(output_dir, site_service_path):
    """Download flowrate data for list generated by generate_batch.

    Args:
        output_dir: Location to save output .db file.
        site_service_path: Path to csv from generate_batch.
        service_type (optional): iv or dv. Defaults to 'iv'.
    """
    iv_access = ValueService()
    site_data = pd.read_csv(site_service_path, parse_dates=['begin_date', 
                                                            'end_date'])

    # Make requests
    for row in site_data.iterrows():
        station = str(row[1]['site_no']).rjust(8, '0')
        print(f'Station: {station}')
        begin_date = row[1]['begin_date'].strftime('%Y-%m-%d')
        end_date = row[1]['end_date'].strftime('%Y-%m-%d')

        connection = sqlite3.connect(os.path.join(output_dir, f'{station}.db'))

        out_df = row[1][['site_no', 'station_nm', 'begin_date', 'end_date', 
                         'count_nu', 'dec_lat_va', 'dec_long_va', 
                         'drain_area_va']].to_frame().T
        out_df.to_sql('miscellaneous', connection, index=False, 
                      if_exists='replace')
        
        print('Downloading dv data')
        out_df = iv_access.query_record(station, begin_date, end_date, 
                                        service_type='dv', save_path=None)
        out_df.to_sql('mean_dv_timeseries', connection, index=False, 
                      if_exists='replace')

        print('Downloading iv data')
        out_df = iv_access.query_record(station, begin_date, end_date, 
                                        service_type='iv', save_path=None)
        out_df.to_sql('raw_timeseries', connection, index=False, 
                      if_exists='replace')
        
def trim_record_to_water_new_year(site_data, water_new_year):
    """Trims start and end dates to the water NY and NYE.

    Args:
        site_data (pandas df): pandas dataframe from ValueService
        water_new_year (str): Date of water NY.  Typically '10-01'

    Returns:
        _type_: _description_
    """
    month_break, day_break = [int(i) for i in water_new_year.split('-')]

    date = site_data['begin_date'].str.split('-', expand=True).astype(int)
    date.columns = ['year', 'month', 'day']
    date['year'] = date['year'] + (((date['month'] > month_break) & 
                                    (date['day'] > day_break)) * 1)
    date['month'] = month_break
    date['day'] = day_break
    site_data['begin_date'] = pd.to_datetime(date)

    water_nye = (site_data['begin_date'][0] - pd.Timedelta(days=1))
    month_break = water_nye.month
    day_break = water_nye.day
    date = site_data['end_date'].str.split('-', expand=True).astype(int)
    date.columns = ['year', 'month', 'day']
    date['year'] = date['year'] - (((date['month'] < month_break) & 
                                    (date['day'] < day_break)) * 1)
    date['month'] = month_break
    date['day'] = day_break
    site_data['end_date'] = pd.to_datetime(date)

    site_data['count_nu'] = (site_data['end_date'] - 
                             site_data['begin_date']).dt.days
    site_data['end_date'] = site_data['end_date'].dt.strftime('%Y-%m-%d')
    site_data['begin_date'] = site_data['begin_date'].dt.strftime('%Y-%m-%d')
    return site_data


def download_gage_date(site_no, water_new_year=None, begin_date=None, end_date=None):
    """Download metadata and flowrate data for a single USGS gage.

    Args:
        site_no (str): USGS site number.
        water_new_year (str): Date of water NY.  Typically '10-01'
        begin_date (str, optional): Optional hard code begin date.
        Format should be 'MM-DD-YYYY'. Defaults to None.
        end_date (atr, optional): Optional hard code end date.
        Format should be 'MM-DD-YYYY'. Defaults to None.
    """
    # Query metadata
    print(f'Loading station metadata for {site_no}')
    service = SiteService()
    record_lengths = service.query_record_lengths(site_no=site_no, 
                                                  service_type='iv')
    site_data = service.query_site_data_expanded(site_no=site_no)
    combo = record_lengths.merge(site_data, on='site_no')
    if water_new_year:
        combo = trim_record_to_water_new_year(combo, water_new_year)
    if not begin_date:
        begin_date = combo.iloc[0]['begin_date']
    if not end_date:
        end_date = combo.iloc[0]['end_date']
    
    # Download data
    iv_access = ValueService()

    misc = combo.iloc[0][['site_no', 'station_nm', 'begin_date', 'end_date', 
                          'count_nu', 'dec_lat_va', 'dec_long_va', 
                          'drain_area_va']].to_frame().T
    
    print('Downloading dv data')
    dv_data = iv_access.query_record(site_no, begin_date, end_date, 
                                     service_type='dv', save_path=None)
    
    print('Downloading iv data')
    iv_data = iv_access.query_record(site_no, begin_date, end_date, 
                                     service_type='iv', save_path=None)
    
    return misc, dv_data, iv_data
    
