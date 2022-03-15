from .regressor import regressionModel
from .regfit import logisticRegressionEvaluator, linearRegressionEvaluator
from simpgenalg import geneticAlgorithm


# Logistic Regression via Genetic Algorithm
class geneticLogisticRegression(regressionModel):

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def update_params(self, **kargs):
        if 'standardize' in kargs:
            self.standardize = kargs.get('standardize')
        if 'normalize' in kargs:
            self.noramlize = kargs.get('noramlize')
        if 'L1' in kargs:
            self.L1 = kargs.get('L1')
        if 'L2' in kargs:
            self.L2 = kargs.get('L2')
        if 'opt_params' in kargs:
            self.opt_params = kargs.get('opt_params')

    # Fits using a GA
    def fit(self, features, labels, test_features=None, test_labels=None):
        # Build a preprocessor
        self._build_preprocessor(train_feats=features)

        # Get parameters for optimizer
        params = self.opt_params.copy()
        params.update({'L1':self.L1, 'L2':self.L2, \
                       'evaluator':logisticRegressionEvaluator,\
                       'train_feats':self._preprocess(feats=features)['feats'],\
                       'train_lbls':labels,\
                       'maximize':False,\
                       'tracking_vars':('fit.max', 'fit.min', 'fit.mean','fit.stdev',\
                           'train_acc.genbest', 'train_acc.mean', 'test_acc.genbest')})
        # If given test features, load them
        if test_features is not None and test_labels is not None:
            params.update({\
                'test_feats':self._preprocess(feats=test_features)['feats'],\
                'test_lbls':test_labels})

        # Build and run the GA and get the results
        gen_alg = geneticAlgorithm(**params)
        results = gen_alg.run()

        # Get best result and extract constant / weights
        best = results.get_best()
        self.constant, self.weights = best['mapped'][0], best['mapped'][1:]

        self._save_results(results)

        return results

    # Returns score of model (accuracy 0 - 1.0)
    def score(self, features, labels):
        # Preprocess the features
        features = self._preprocess(feats=features)['feats']
        return super().score(features, labels)

# Linear Regression via Genetic Algorithm
class geneticLinearRegression(regressionModel):

    __slots__ = ()

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    # Fits using a GA
    def fit(self, features, labels, **kargs):
        # Build a preprocessor
        self._build_preprocessor(feats=features)
        # Preprocess the features
        features = self._preprocess(feats=features)
        # Any additional kargs should be sent to try to extact important values
        if len(kargs) > 0:
            self.update_params(**kargs)
        # Copy optimizer parameters and update with object's values
        params = self.opt_params.copy()
        params.update({'L1':self.L1, 'L2':self.L2, \
                       'evaluator':linearRegressionEvaluator,\
                       'dtype':float,\
                       'maximize':False,\
                       'tracking_vars':('fit.max', 'fit.min', 'fit.mean','fit.stdev',\
                           'train_acc.max', 'train_acc.mean')})

        # Run the GA and get the results
        gen_alg = geneticAlgorithm(**params)
        results = gen_alg.run()

        # Get best result and extract constant / weights
        best = results.get_best()
        self.constant, self.weights = best['mapped'][0], best['mapped'][1:]

        self._save_results(results)

        return results
    # Returns score of model (coefficient of determination 0 - 1.0)
    def score(self, features, labels):
        # Preprocess the features and labels
        dct = self._preprocess(feats=features, lbls=labels)
        return super().score(dct['feats'], dct['labels'])
