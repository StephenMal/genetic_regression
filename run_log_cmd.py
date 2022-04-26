from genreg import geneticLogisticRegression
import numpy as np
import pandas as pd
from datasets import load_dataset_from_dir
from tensorflow import convert_to_tensor, float32
import sys, json, os, argparse
from statistics import mean, stdev
import random

def interpret_params():
    print('Interpretting arguments')
    argp = argparse.ArgumentParser()
    argp.add_argument('inp_fol', type=str)
    argp.add_argument('out_fol', type=str)
    argp.add_argument('rep', type=str, choices=['vga','bsga','pga','fga','mga'])
    argp.add_argument('--n_runs', type=int, default=50)
    argp.add_argument('--gene_size', type=int, default=16)
    argp.add_argument('--noncoding', type=int, default=3)
    argp.add_argument('--length', type=int, default=-1)
    argp.add_argument('--pga_map_fxn', type=int, choices=[0,1,2,3], default=2)
    argp.add_argument('--n_gens', type=int, default=200)
    argp.add_argument('--mut_rate', type=int, default=0.015)
    argp.add_argument('--xov_rate', type=int, default=0.9)
    argp.add_argument('--mut_op', type=str, default='')
    argp.add_argument('--xov_op', type=str, default='')
    argp.add_argument('--pop_size', default=100)
    argp.add_argument('--L1', type=int, default=0)
    argp.add_argument('--L2', type=int, default=0)
    argp.add_argument('--min_weight', type=float, default=-1)
    argp.add_argument('--max_weight', type=float, default=1)
    argp.add_argument('--min', type=float, default=-1)
    argp.add_argument('--max', type=float, default=1)
    argp.add_argument('--dtype', default='float')
    argp.add_argument('--togs', action='store_true')
    argp.add_argument('--exps', action='store_true')
    argp.add_argument('--exps_keep_sign', action='store_true')
    args = argp.parse_args(sys.argv[1:])

    if args.dtype == 'float':
        args.dtype = float
        args.min, args.max = float(args.min), float(args.max)
    elif args.dtype == 'int':
        args.dtype = int
        args.min, args.max = int(args.min), int(args.max)

    return args

args = interpret_params()
print('Loading Data')
data = load_dataset_from_dir(args.inp_fol)

train_feats = data['train']
train_lbls = train_feats.pop(train_feats.columns[-1])

test_feats = data['test']
test_lbls = test_feats.pop(test_feats.columns[-1])

'Setting up Params'
ga_params = {'n_runs':args.n_runs, 'n_gens':args.n_gens,\
             'L1':args.L1, 'L2':args.L2,\
             'min_weight':args.min_weight, 'max_weight':args.max_weight,\
             'encode_toggles':args.togs,\
             'encode_exponents':args.exps,\
             'exponents_keep_sign':args.exps_keep_sign,\
             'pop_size':args.pop_size,\
             'xov_rate':args.xov_rate, 'xov_op':args.xov_op,\
             'mut_rate':args.mut_rate, 'mut_op':args.mut_op,\
             'header':list(train_feats.columns),\
             'min':args.min,'max':args.max}

if args.rep == 'vga':
    ga_params.update({'representation':'vector',\
                      'dtype':args.dtype,\
                      'xov_op':args.xov_op \
                                if args.xov_op != '' else 'twopt',\
                      'mut_op':args.mut_op \
                                if args.mut_op != '' else 'uniform_mutation'})
elif args.rep == 'bsga':
    ga_params.update({'representation':'binary',\
                      'gene_size':args.gene_size,\
                      'dtype':args.dtype,\
                      'xov_op':args.xov_op \
                                if args.xov_op != '' else 'twopt',\
                      'mut_op':args.mut_op \
                                if args.mut_op != '' else 'flipbit'})
elif args.rep == 'pga':
    ga_params.update({'representation':'proportional',\
                      'n_noncoding_chars':args.noncoding,\
                      'map_fxn':args.pga_map_fxn,\
                      'xov_op':args.xov_op \
                                if args.xov_op != '' else 'twopt',\
                      'mut_op':args.mut_op \
                                if args.mut_op != '' else 'uniform_mutation'})
    ga_params['len'] = len(train_feats.columns)*abs(args.length) \
                                if args.length < 0 else args.length

print('Creating Model')
genetic_regressor = geneticLogisticRegression(preprocess=True,\
                                              preprocess_lbls=False,\
                                              normalize=True,\
                                              standardize=True,\
                                              opt_params=ga_params)
print('Fitting')
genetic_regressor.fit(feats=train_feats, lbls=train_lbls,\
                                test_feats=test_feats, test_lbls=test_lbls)

''' Evaluation '''
# Try out every possible of ensemble
print('Evaluation', end='')
accuracy_dct = {}
for type, feats, lbls in (('train',train_feats,train_lbls),\
                         ('test',test_feats,test_lbls)):
    # Pick the best model for training and test
    print('.',end='')
    accuracy_dct.update(\
        {f'{type}-best_train':\
                genetic_regressor.score(feats, lbls, best_train_model=True),\
         f'{type}-best_test':\
                genetic_regressor.score(feats, lbls, best_test_model=True)})
    # Pick v_method
    for v_meth in ('hard', 'soft'):
        print('.',end='')
        accuracy_dct.update(\
            {f'{type}-{v_meth}-unweighted':\
                genetic_regressor.score(feats, lbls, voting_method=v_meth,\
                                                        voting_ensemble=True)})
        # Look at different weights
        for w in ('train', 'test'):
            print('.',end='')
            accuracy_dct.update(\
                {f'{type}-{v_meth}-{w}':\
                    genetic_regressor.score(feats, lbls, \
                                                voting_method=v_meth,\
                                                voting_ensemble=True,\
                                                w_ensemble=w)})
print('')

''' Record data '''
# Get information on the different weights
wmat = genetic_regressor.get_wmat(incl_constant=False)
columns = {f'w_{cnum}_{cname}':[wmat[run][cnum] for run in range(len(wmat))] \
                        for cnum, cname in enumerate(train_feats.columns)}
w_dct = {f'{key}_mean':mean(col) for key, col in columns.items()}
w_dct.update({f'{key}_stdev':stdev(col) for key, col in columns.items()})
best_model = max(genetic_regressor.models, key=lambda item: item.train_acc)
w_dct.update({f'w_{cnum}_{cname}_best':w for cnum, (cname, w) in enumerate(zip(\
                                train_feats.columns, best_model.weights))})
constants = [m.constant for m in genetic_regressor.models]
w_dct['w_constant_mean'] = mean(constants)
w_dct['w_constant_stdev'] = stdev(constants)
w_dct['w_constant_best'] = best_model.constant

# Combine all the dictionaries
dct = ga_params.copy()
dct.update(accuracy_dct)
dct.update(w_dct)
dct.pop('header')


# Create output folder (if its not already made)
if not os.path.exists(f'{args.out_fol}/'):
    os.makedirs(f'{args.out_fol}/')
# Either create a new data frame or append to it
if not os.path.exists(f'{args.out_fol}/res.feather'):
    print('Creating res.feather')
    df = pd.DataFrame()
else:
    print('Found existing results')
    df = pd.read_feather(f'{args.out_fol}/res.feather')
    print(df)
# Generate random id, verify its not in the df
id_num = random.randint(0,999999999)
if df.shape[0] != 0:
    while(id_num in df['id_num']):
        id_num = random.randint(0,999999999)
dct['id_num'] = id_num

for key, item in dct.items():

    if not isinstance(item, float) and \
           not isinstance(item, int) and \
           not isinstance(item, bool):
       dct[key] = str(item)
dct = {key:[item if isinstance(item, (int,float,bool)) else str(item)] \
                        if not isinstance(item, list) else item \
                                                for key,item in dct.items()}
#df.append(dct, ignore_index=True)
df = pd.concat([df, pd.DataFrame(dct)], ignore_index=True)
# Reorder it to be alphabetical (kinda)
df = df.reindex(columns=['id_num']+\
                sorted(accuracy_dct.keys())+\
                sorted(ga_params.keys())+\
                sorted(w_dct.keys()))
try:
    df.to_feather(f'{args.out_fol}/res.feather')
except:
    print('Reseting index')
    df = df.reset_index()
    df.to_feather(f'{args.out_fol}/res.feather')

# Save run results
results = genetic_regressor.get_results()
result_dfs = results.to_df()
# Make results dir
if not os.path.exists(f'{args.out_fol}/{id_num}/'):
    os.makedirs(f'{args.out_fol}/{id_num}/')
# Save the weights plot
genetic_regressor.get_best_run_weights_plot().\
                    write_html(f'{args.out_fol}/{id_num}/weights.html')
# Save the results
result_dfs[0].to_feather(f'{args.out_fol}/{id_num}/indvs.feather')
result_dfs[1].to_feather(f'{args.out_fol}/{id_num}/popstats.feather')
# Save the preprocessor information
with open(f'{args.out_fol}/{id_num}/preproc.json', 'w') as F:
    json.dump(genetic_regressor.get_preproc_dict(), F)

print('Completed')
