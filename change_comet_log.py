import comet_ml

query = ((comet_ml.api.Metric('test_rmse')!=None) &
         (comet_ml.api.Parameter('note')=='STL-23,(car~txt+fr 5y),txtfc=0,fc=0,txtdropout=no,fc_dropout=no,NormCAR=yes,bsz=28,seed=42,log(mcap)=yes,lr=0.0003'))

exps = comet_ml.api.API().query('amiao', 'earnings-call', query, archived=False)

for exp in exps:
    # roll_type = exp.get_parameters_summary('roll_type').valueCurrent
    exp.log_parameter('note', 'STL-28,(car~txt+fr 5y),txtfc=0,fc=0,txtdropout=no,fc_dropout=no,NormCAR=yes,bsz=28,seed=42,log(mcap)=yes,lr=3e-4')
    