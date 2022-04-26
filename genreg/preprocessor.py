import numpy as np

try:
    import pandas as pd
except:
    pass

class preprocessor():

    __slots__ = ('standardize', 'normalize', 'stats')

    def __init__(self, **kargs):

        # Booleans of whether we are applying standardization / normalization
        self.normalize = kargs.get('normalize', False)
        self.standardize = kargs.get('standardize', False)
        # Contains stats of all the columns
        self.stats = kargs.get('stats')

    def _set_stats(self, feats=None, lbls=None):
        if self.stats is None:
            self.stats = {'standardize':self.standardize,\
                          'normalize':self.normalize}

        if isinstance(feats, pd.DataFrame):
            self.stats.update({'mean':{col_name:float(feats[col_name].mean()) \
                                      for col_name in feats.columns},\
                               'stdev':{col_name:float(feats[col_name].std()) \
                                      for col_name in feats.columns}, \
                               'max':{col_name:float(feats[col_name].max()) \
                                      for col_name in feats.columns}, \
                               'min':{col_name:float(feats[col_name].min()) \
                                      for col_name in feats.columns}})
        elif isinstance(feats, np.ndarray):
            if len(feats.shape) != 2:
                raise ValueError('Expected 2D shape numpy array')
            self.stats.update({'mean':{col_name:float(feats[col_name].mean()) \
                                      for col_name in range(len(feats))},\
                               'stdev':{col_name:float(feats[col_name].std()) \
                                      for col_name in range(len(feats))}, \
                               'max':{col_name:float(feats[col_name].max()) \
                                      for col_name in range(len(feats))}, \
                               'min':{col_name:float(feats[col_name].min()) \
                                      for col_name in range(len(feats))}})
        elif isinstance(feats, (list, tuple)):
            if not all([isinstance(item, (list,tuple)) for item in feats]):
                raise ValueError('Expected 2D list')
            elif not all([len(item)==len(feats[0]) for item in feats]):
                raise ValueError('Inconsistent lengths')
            self.stats.update({'mean':{col_name:float(mean(feats[col_name])) \
                                      for col_name in range(len(feats))},\
                               'stdev':{col_name:float(stdev(feats[col_name])) \
                                      for col_name in range(len(feats))}, \
                               'max':{col_name:float(max(feats[col_name])) \
                                      for col_name in range(len(feats))}, \
                               'min':{col_name:float(min(feats[col_name])) \
                                      for col_name in range(len(feats))}})
        else:
            raise TypeError(f'Expected DataFrame, Numpy ndarray, or list')

        if lbls is not None:

            if isinstance(lbls, (pd.DataFrame, pd.Series)):
                self.stats.update({'lbl_info':{'mean':float(lbls.mean()),\
                                               'stdev':float(lbls.std()),\
                                               'max':float(lbls.max()),\
                                               'min':float(lbls.min())}})
            elif isinstance(lbls, np.ndarray):
                if len(lbls.shape) != 1:
                    raise ValueError('Expected 1D shape numpy array')
                self.stats.update({'lbl_info':{'mean':float(lbls.mean()),\
                                               'stdev':float(lbls.std()),\
                                               'max':float(lbls.max()),\
                                               'min':float(lbls.min())}})
            elif isinstance(lbls, (list, tuple)):
                self.stats.update({'lbl_info':{'mean':float(mean(lbls)),\
                                               'stdev':float(stdev(lbls)),\
                                               'max':float(max(lbls)),\
                                               'min':float(min(lbls))}})
        return


    def fit(self, feats=None, lbls=None):
        # Set statistics dictionary
        self._set_stats(feats=feats, lbls=lbls)

    def transform(self, feats=None, lbls=None):
        new_feats = feats.copy()

        if isinstance(feats, (pd.DataFrame, pd.Series, np.ndarray)):
            if self.normalize:
                for col in feats.columns:
                    new_feats[col] = \
                            (new_feats[col] - new_feats[col].min()) / \
                                (new_feats[col].max() - new_feats[col].min())
            if self.standardize:
                for col in feats.columns:
                    new_feats[col] = (new_feats[col] - new_feats[col].mean()) /\
                                                            new_feats[col].std()
        elif isinstance(feats, (list, tuple)):
            if self.normalize:
                for i in range(len(feats)):
                    cmin, cmax = min(new_feats[i]), max(new_feats[i])
                    new_feats[i] = [(v - cmin)/(cmax-cmin) for v in new_feats[i]]
            if self.standardize:
                for i in range(len(feats)):
                    cmean, cstd = mean(new_feats[i]), stdev(new_feats[i])
                    new_feats[i] = [((v - cmean) / cstd) for v in new_feats[i]]

        if lbls is not None:
            new_lbls = new_lbls.copy()
            if isinstance(lbls, (pd.Series, np.ndarray)):
                if self.normalize:
                    new_lbls = (new_lbls - new_lbls.min()) / \
                                            (new_lbls.max() - new_lbls.min())
                if self.standardize:
                    new_lbls = (new_lbls - new_lbls.mean()) / new_lbls.std()
            elif isinstance(lbls, (list, tuple)):
                if self.normalize:
                    new_lbls = (new_lbls - new_lbls.min()) / \
                                            (new_lbls.max() - new_lbls.min())
                if self.standardize:
                    new_lbls = (new_lbls - new_lbls.mean()) / new_lbls.std()
            return new_feats, new_lbls

        return new_feats

    def fit_transform(self, feats=None, lbls=None):
        self.fit(feats=feats, lbls=lbls)
        return self.transform(feats=feats, lbls=lbls)

    def to_dict(self):
        return self.stats

    def from_dict(self):
        self.stats = stats_dct.copy()
        self.normalize = stats_dct.get('normalize', False)
        self.standardize = stats_dct.get('standardize', False)

    def __getstate__(self):
        return self.stats

    def __setstate__(self, stats_dct):
        self.stats = stats_dct.copy()
        self.normalize = stats_dct.get('normalize', False)
        self.standardize = stats_dct.get('standardize', False)
