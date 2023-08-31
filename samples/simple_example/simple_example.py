from dotmodel import Gage

db_path = r'01638500.db'
potomac = Gage(db_path)
potomac.download_nwis('01638500', '10-01')
potomac.clean_timeseries()
potomac.flow_frequency_analysis(truncation_arrival_rate=2)
potomac.duration_analysis()

potomac.plot_flow_frequency('01638500_ffa.png')
potomac.plot_duration_power_law('01638500_duration.png')
potomac.plot_model_summary('01638500.png')

potomac.plot_timeseries('raw_timeseries', '01638500_raw_ts.png')
potomac.plot_exponential_durations('01638500_exp_durations.png')