from genreg import geneticLogisticRegression
import numpy as np
import pandas as pd
from datasets import load_dataset_from_dir
from tensorflow import convert_to_tensor, float32

#gen_regressor = geneticRegressor()

data = load_dataset_from_dir('datasets/medicare')

train_feats = data['train']
train_lbls = train_feats.pop('Above_Median?')

test_feats = data['test']
test_lbls = test_feats.pop('Above_Median?')

ga_params = {'n_runs':5, 'n_gens':20,\
             'L1':0.0, 'L2':0.0,\
             'gene_size':24, 'representation':'vector',\
             'min_weight':-1.0, 'max_weight':1.0,\
             'encode_toggles':True,\
             'encode_exponents':False, 'exponents_keep_sign':True,\
             'pop_size':200,\
             'xov_rate':0.9, 'xov_op':'twopt',\
             'mut_rate':0.2, 'mut_op':'uniform_mutation',\
             'header':list(train_feats.columns),\
             'min':-1.0,'max':1.0,\
             'dtype':float}

genetic_regressor = geneticLogisticRegression(preprocess=True,\
                                              preprocess_lbls=False,\
                                              normalize=True,\
                                              standardize=True,\
                                              opt_params=ga_params)

genetic_regressor.fit(feats=train_feats, lbls=train_lbls,\
                                test_feats=test_feats, test_lbls=test_lbls)

genetic_regressor.get_best_run_weights_plot().write_html(f'results/weights.html')


# Try out every possible of ensemble
accuracy_dct = {}
for type, feats, lbls in (('train',train_feats,train_lbls),\
                         ('test',test_feats,test_lbls)):
    # Pick the best model for training and test
    accuracy_dct.update(\
        {f'{type}-best_train':\
                genetic_regressor.score(feats, lbls, best_train_model=True),\
         f'{type}-best_test':\
                genetic_regressor.score(feats, lbls, best_test_model=True)})
    # Pick v_method
    for v_meth in ('hard', 'soft'):
        accuracy_dct.update(\
            {f'{type}-{v_meth}-unweighted':\
                genetic_regressor.score(feats, lbls, voting_method=v_meth,\
                                                        voting_ensemble=True)})
        # Look at different weights
        for w in ('train', 'test'):
            accuracy_dct.update(\
                {f'{type}-{v_meth}-{w}':\
                    genetic_regressor.score(feats, lbls, \
                                                voting_method=v_method,\
                                                voting_ensemble=True,\
                                                w_ensemble=w)})

results = genetic_regressor.get_results()
result_dfs = results.to_df()

result_dfs[0].to_feather('results/indvs.feather')
result_dfs[1].to_feather('results/popstats.feather')
