import comet_ml
import pandas as pd
import pyarrow.feather as feather


comet_api = comet_ml.API()
exps = comet_api.query('amiao', 'earnings-call', 
                       # comet_ml.api.Parameter('note').startswith('STL-23'),
                       comet_ml.api.Metric('test_rmse')!=None,
                       archived=False)

if len(exps)!=32:
    print(f'Experient number != 32! Get {len(exps)} results.')

log_comet = []
for exp in exps:
    # get parameter
    log = {param['name']:param['valueCurrent'] for param in exp.get_parameters_summary()}
    
    # get metrics
    log['test_rmse'] = exp.get_metrics('test_rmse')[0]['metricValue']
    
    if len(exp.get_metrics('test_rmse_car'))>0:
        log['test_rmse_car'] = exp.get_metrics('test_rmse_car')[0]['metricValue']
           
    if len(exp.get_metrics('test_rmse_inflow'))>0:
        log['test_rmse_inflow'] = exp.get_metrics('test_rmse_inflow')[0]['metricValue']
    
    if len(exp.get_metrics('test_rmse_inflow'))>0:
        log['test_rmse_inflow'] = exp.get_metrics('test_rmse_car')[0]['metricValue']
           
    # get metadat
    log = {**log, **exp.get_metadata()}
    
    # delete useless params
    for key in ['checkpoint_path', 'f']:
        log.pop("key", None)
    log_comet.append(log)
    
log_comet = pd.DataFrame(log_comet)

# append new results to existing comet log
# old_log_comet = feather.read_feather('data/comet_log.feather')
# new_log_comet = pd.concat([old_log_comet, log_comet], ignore_index=True)
# pd.DataFrame.drop_duplicates(new_log_comet, ignore_index=True)
feather.write_feather(log_comet, 'data/comet_log.feather')