from .logistic_regfit import logisticRegressionEvaluator
from .regressor import regressionModel
from .preprocessor import preprocessor
import numpy as np

class geneticLogisticRegression(regressionModel):

    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.fit_fxn = logisticRegressionEvaluator

    def score(self, feats, lbls, **kargs):
        if self.preprocess_lbls:
            _, lbls = self._preprocess(feats=feats, lbls=lbls)
        preds = self.predict(feats, **kargs)
        return self.fit_fxn.score(preds=preds, lbls=lbls)

    def predict(self, feats, **kargs):
        feats = self._preprocess(feats=feats)
        # Get features into numpy array if not already
        if not isinstance(feats, np.ndarray):
            try:
                feats = np.array(feats)
            except Exception as e:
                self.log.exception(str(e))
                raise Exception('Feats needs to a numpy ndarray')

        if len(self.models) == 0:
            raise Exception('Cannot predict on unfitted model')
        elif len(self.models) == 1 or kargs.get('best_train_model', False) or \
                    kargs.get('best_test_model', False) or 'model_n' in kargs:
            if len(self.models) == 1:
                model = self.models[0]
            elif kargs.get('best_train_model', False):
                models = [m for m in self.models if m.train_acc is not None]
                if len(models) == 1:
                    model = models[0]
                elif len(models) == 0:
                    raise Exception('No models had training accuracy saved')
                else:
                    model = max(models, key=lambda item: item.train_acc)
            elif kargs.get('best_test_model', False):
                models = [m for m in self.models if m.test_acc is not None]
                if len(models) == 1:
                    model = models[0]
                elif len(models) == 0:
                    raise Exception('No models had testing accuracy saved')
                else:
                    model = max(models, key=lambda item: item.test_acc)
            elif 'model_n' in kargs:
                n = kargs.get('model_n')
                if not isinstance(n, int):
                    raise TypeError('Expected int for model_n')
                elif n < 0 or n > len(self.models):
                    raise ValueError('model_n must be between 0 and # of models')
                model = self.models[n]
            else:
                raise Exception('Failed to select a model to test')

            c,w = model.constant, model.weights
            if feats.shape[1] != len(w):
                raise ValueError(f'{len(w)} number of weights '+\
                                  f'for {feats.shape[1]} features')
            z = self.fit_fxn.predict(constant=model.constant,\
                                         weights=model.weights,\
                                         feats=feats)
            return self.fit_fxn.predict(constant=model.constant,\
                                         weights=model.weights,\
                                         feats=feats)
        else: # Otherwise its an ensemble
            # Determine voting method (hard or soft)
            hard_v = kargs.get('voting_method', 'hard')
            if not isinstance(hard_v, str):
                raise TypeError('Expected string for voting_method')
            elif hard_v.lower() != 'hard' and hard_v.lower() != 'soft':
                raise ValueError('Expected soft or hard for voting method')
                hard_v = hard_v.lower() == 'hard'
                hard_v = kargs.get('voting_method', 'hard') == 'hard'
            # Get fit_fxn in this scope (due to multiple calls)
            fit_fxn = self.fit_fxn
            # \/ Goes into voting ensemble
            if kargs.get('voting_ensemble', False):
                # Stores votes
                votes = [[0,0] for x in range(feats.shape[0])]
                # If no w_ensemble value given, assume unweighted
                if kargs.get('w_ensemble', None) is None or \
                                        kargs.get('w_ensemble', False) == False:
                    # Predict per model
                    predictions = [fit_fxn.predict(constant=m.constant,\
                                                   weights=m.weights, \
                                                   feats=feats, \
                                                   return_prob=(not hard_v))\
                                                   for m in self.models]
                    if hard_v: # If hard v, a vote is a vote
                        for preds in predictions:
                            for indx, vote in enumerate(preds):
                                votes[indx][int(vote)] += 1
                    else: # If soft v, we add the probabilities not the vote
                        for preds in predictions:
                            for indx, vote in enumerate(preds):
                                votes[indx][round(vote)] += vote
                    # \/ weighted ensembles will apply weighting on diff models
                else:
                    # Determine which kind of weighted ensembler we are building
                    w_ensemble = kargs.get('w_ensemble')
                    if not isinstance(w_ensemble, str):
                        raise TypeError('Expected w_ensemble to be a str')
                    w_ensemble = w_ensemble.lower()
                    if w_ensemble not in ('test', 'train', 'custom'):
                        raise ValueError('Expected test, train, or custom for'+\
                                         ' w_ensemble')
                    # Get the weights depending on w_ensemble type
                    if w_ensemble == 'train':
                        models = [m for m in self.models \
                                                    if m.train_acc is not None]
                        accs = [m.train_acc for m in models]
                    elif w_ensemble == 'test':
                        models = [m for m in self.models \
                                                    if m.test_acc is not None]
                        accs = [m.test_acc for m in models]
                    elif w_ensemble == 'custom':
                        models = self.models
                        accs = kargs.get('custom_w', None)
                        if accs is None:
                            raise ValueError('Expected a custom_w value if '+\
                                             'using custom val for w_ensemble')
                        if len(accs) != models:
                            raise ValueError('custom_w should be an iterable '+\
                                             'same length as number of models')
                    else:
                        raise Exception('w_ensemble not properly specified')

                    # Get predictions per model
                    predictions = [fit_fxn.predict(constant=m.constant,\
                                                   weights=m.weights, \
                                                   feats=feats, \
                                                   return_prob=(not hard_v))\
                                                            for m in models]
                    # Apply voting
                    if hard_v: # Hard vote means a vote is a vote
                        for acc, preds in zip(accs, predictions):
                            for indx, vote in enumerate(preds):
                                # Adds accuracy to account for weight
                                votes[indx][vote] += acc
                    else:
                        for acc, preds in zip(accs,predictions):
                            for indx, vote in enumerate(preds):
                                # Multiplies by accuracy to account for weight
                                votes[indx][round(vote)] += vote*acc
                # Whichever has more votes wins, returns list of predictions
                return [0 if v[0]>v[1] else 1 for v in votes]
