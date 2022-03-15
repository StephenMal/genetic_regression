from genreg import  geneticLogisticRegression, geneticLinearRegression
import numpy as np
import pandas as pd

#gen_regressor = geneticRegressor()


def load(inp, **kargs):
    df = pd.read_csv(inp, delimiter=kargs.get('sep',','))

    feat_cols = [df.columns[i] for i in kargs.get('feat_cols',\
                                                range(0,len(df.columns)-1))]
    lbl_col = df.columns[kargs.get('lbl_col',-1)]

    return df[feat_cols].to_numpy(), df[lbl_col].to_numpy(), feat_cols, lbl_col

train_feats, train_lbls, feat_cols, lbl_col = \
        load('datasets/medicare_training_50.csv', \
                feat_cols=range(0,25), lbl_col=-1)

test_feats, test_lbls, feat_cols, lbl_col = \
        load('datasets/medicare_testing_50.csv', \
                feat_cols=range(0,25), lbl_col=-1)

ga_params = {'n_runs':2, 'n_gens':20,\
             'L1':0.0, 'L2':0.0,\
             'gene_size':24, 'representation':'vector',\
             'min_weight':-1.0, 'max_weight':1.0,\
             'encode_toggles':True,\
             'encode_exponents':False, 'exponents_keep_sign':True,\
             'pop_size':100,\
             'xov_rate':0.9, 'xov_op':'twopt',\
             'mut_rate':0.2, 'mut_op':'uniform_mutation',\
             'header':feat_cols,\
             'min':-1.0,'max':1.0}

genetic_regressor = geneticLogisticRegression(standardize=True, \
                                              normalize=False, \
                                              encode_toggles=True,\
                                              encode_decision_boundary=False,\
                                              encode_exponents=False,\
                                              opt_params=ga_params)

results = genetic_regressor.fit(features=train_feats, labels=train_lbls,\
                                test_features=test_feats, test_labels=test_lbls)


genetic_regressor.get_best_run_weights_plot().write_html(f'results/weights.html')

result_dfs = results.to_df()

result_dfs[0].to_csv('results/indvs.csv')
result_dfs[1].to_csv('results/popstats.csv')
