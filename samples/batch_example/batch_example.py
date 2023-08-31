import os
import pandas as pd
from dotmodel import nwis, Gage


# Download data from gages in Vermont with more than 30 years of record
working_dir = 'vermont'
os.makedirs(working_dir, exist_ok=True)
site_list_path = os.path.join(working_dir, 'sites.csv')
nwis.generate_batch('VT', site_list_path, 30, '10-01')
gages = pd.read_csv(site_list_path)
db_list = [str(f).rjust(8, '0') for f in gages['site_no']]

# Fit model at each station
for db in db_list:
    db_path = os.path.join(working_dir, f'{db}.db')
    tmp_gage = Gage(db_path)
    tmp_gage.download_nwis('01638500', '10-01')
    tmp_gage.clean_timeseries()
    tmp_gage.flow_frequency_analysis(truncation_arrival_rate=2)
    tmp_gage.duration_analysis()
