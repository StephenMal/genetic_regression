from .regressor import regressionModel
from tensorflow.kears.optimizers import SGD, GradientDescentOptimizer, \
    Adagrad, RMSprop, Adadelta, Adam, Adamax, Nadam, Ftrl


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
                       'evaluator':logisticRegressionEvaluator})

        # Run the GA and get the results
        gen_alg = geneticAlgorithm(**self.opt_params)
        results = gen_alg.run()

        # Get best result and extract constant / weights
        best = results.get_best()
        self.constant, self.weights = best['mapped'][0], best['mapped'][1:]

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
                       'evaluator':logisticRegressionEvaluator})

        # Run the GA and get the results
        gen_alg = geneticAlgorithm(**params)
        results = gen_alg.run()

        # Get best result and extract constant / weights
        best = results.get_best()
        self.constant, self.weights = best['mapped'][0], best['mapped'][1:]

        return results
    # Returns score of model (coefficient of determination 0 - 1.0)
    def score(self, features, labels):
        # Preprocess the features and labels
        dct = self._preprocess(feats=features, lbls=labels)
        return super().score(dct['feats'], dct['labels'])
