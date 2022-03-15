
# Module imports
from simpgenalg import geneticAlgorithm
import numpy as np

# Local imports
from .regfit import regressionEvaluator
from .regressor import regressionModel
from .preprocessor import preprocessor

class geneticRegressor(regressionModel):

    __slots__ = ('ga_params')

    dflt_ga_params = {'len':None,\
                      'chr_max':1,\
                      'chr_min':0,\
                      'xov_op':'onept',\
                      'mut_op':'uniform_mutation',\
                      'mut_rate':0.2,\
                      'xov_rate':0.9,\
                      'maximize':False,\
                      'cmpr_map_dist':False}

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        # Create dictionary to store ga parameters
        self.ga_params = {}

    # Fit the
    def fit(self, feats=None, lbls=None, test_feats=None, test_lbls=None, \
                    ga_params={}, **kargs):

        # If provided any extra kargs, update the parameters
        if len(kargs) > 0:
            self.set_params(**kargs)

        # Verify both features and labels were provided
        if feats is None or lbls is None:
            raise Exception('Either feats or lbls were not provided')

        # Preprocess feats and labels
        self._build_preprocess(feats, lbls)
        train = self._preprocess(feats, lbls)

        feats = train['feats']
        if self.params['reg_type'] != 'log':
            lbls = train['lbls']

        # Verify if test values are provided that both
        if (test_feats is not None or test_lbls is not None):
            if not (test_feats is not None and test_lbls is not None):
                raise Exception('both test_feats and test_lbls need to be '+\
                                    'provided')
            test = self._preprocess(test_feats, test_lbls)
            test_feats = test['feats']
            if self.params['reg_type'] != 'log':
                test_lbls = test['lbls']

        # Add it into the ga params
        for key, item in self.dflt_ga_params.items():
            if key not in self.ga_params:
                self.ga_params[key] = item


        # Update the ga params
        self.ga_params.update(self.params)
        self.ga_params.update(ga_params)

        # Determine length
        n_feats = feats.shape[1]
        length = n_feats+1
        self.ga_params.update({})
        if self.ga_params.setdefault('encode_decision_boundary', False):
            length += 1
        else:
            self.ga_params.setdefault('decision_boundary', 0)
        if self.ga_params.setdefault('encode_exponents', False):
            length += n_feats
        if self.ga_params.setdefault('encode_toggles', False):
            length += n_feats
        self.ga_params['len'] = length


        # Set up the GA using the provided parameters
        self.ga_params['evaluator'] = regressionEvaluator
        self.ga_params['dtype'] = float
        self.ga_params['train_feats'] = feats
        self.ga_params['train_lbls'] = lbls
        self.ga_params['standardize'] = self.params.get('standardize', False)
        self.ga_params['normalize'] = self.params.get('normalize', False)
        self.ga_params['test_feats'] = test_feats
        self.ga_params['test_lbls'] = test_lbls
        self.ga_params['maximize'] = False
        self.ga_params['reg_type'] = self.params.get('reg_type', 'lin')
        if self.ga_params['reg_type'] == 'log':
            self.ga_params['tracking_vars'] = ('fit.mean', 'fit.stdev',\
                                               'train_acc.mean',\
                                               'train_acc.max','fit.min',\
                                               'fit.max')
        self.ga_params['L1'] = self.params.get('L1', 0)
        self.ga_params['L2'] = self.params.get('L2', 0)

        ga = geneticAlgorithm(**self.ga_params)

        # Run the GA
        results = ga.run()

        # Get the best weights
        self.weights = results.get_best()['mapped']

        return results
