from .preprocessor import preprocessor
from .regfit import logisticRegressionEvaluator, linearRegressionEvaluator
import sys

try:
    import pandas as pd
except:
    pass

try:
    import plotly.express as px
    import plotly.graph_objects as go
except:
    pass

class regressionModel():

    __slots__ = ('constant', 'weights',\
                 'preprocessor', 'standardize', 'normalize',\
                 'L1', 'L2', 'fit_fxn',
                 'opt_params')

    def __init__(self, *args, **kargs):

        self.constant = kargs.get('constant', None)
        self.weights = kargs.get('weights', None)
        self.preprocessor = kargs.get('preprocessor', None)
        self.standardize = kargs.get('standardize', False)
        self.normalize = kargs.get('normalize', False)
        self.L1, self.L2 = kargs.get('L1', 0.0), kargs.get('L2', 0.0)
        self.fit_fxn = kargs.get('fit_fxn', None)
        self.opt_params = kargs.get('opt_params', {})
        self.last_results = kargs.get('last_results', None)

    # Builds preprocessor based off training data / labels
    #   Called when fitting
    def _build_preprocessor(self, train_feats, train_lbls=None, **kargs):
        self.preprocessor = preprocessor(standardize=self.standardize,\
                                         normalize=self.normalize)
        self.preprocessor.fit(feats=train_feats, lbls=train_lbls)


    # Uses preprocessor
    def _preprocess(self, feats=None, lbls=None):
        return self.preprocessor.apply(feats=feats, lbls=lbls)

    # fits the model based off given training data
    def fit(self, *args, **kargs):
        raise NotImplementedError

    # Prediction
    def predict(self, features):
        return self.fit_fxn.predict(self.weights, \
                                    self._preprocess(feats=features))

    # Returns respective scoring
    def score(self, features, labels):
        # Make predictions
        predictions = self.fit_fxn.predict(self.weights, features)
        # Find accuracy
        return self.fit_fxn._score(predictions, labels)

    def _save_results(self, results):
        self.last_results = results

    def get_results(self):
        if self.last_results is None:
            raise Exception('Has no results')
        return self.last_results

    @staticmethod
    def read_csv(input, **kargs):
        if 'pandas' not in sys.modules:
            raise ModuleNotFoundError('Need pandas to use read_csv')
        # Read in csv
        df = pd.read_csv(inp, delimiter=kargs.get('sep', ','))

        # Get feature columns / label column
        feat_cols = [df.columns[i] for i in kargs.get('feat_cols',\
                                                    range(0,len(df.columns)-1))]
        lbl_col = df.columns[kargs.get('lbl_col',-1)]
        # Return the values
        return {'features':df[feat_cols].to_numpy(),\
                'feature_cols':feat_cols,\
                'labels':df[lbl_col].to_numpy(),\
                'lbl_col':lbl_col}

    ''' Optional Plots '''
    def get_best_run_weights_plot(self, stats_df=None):
        if stats_df is None:
            results = self.get_results()
            indvs, stats = results.to_df()
        else:
            stats = stats_df

        # Get weight columns
        weight_columns = [col for col in stats.columns \
                            if ('weight_' in col and '.runbest' in col)]
        df_cols = weight_columns + ['_run', 'fit.runbest']

        def sort_fxn(key):
            return int(key.split('_')[1])

        new_df = stats.melt(id_vars = ['_run', 'fit.runbest'], \
                            value_vars = sorted(weight_columns, key=sort_fxn), \
                            var_name = 'weight_name',\
                            value_name = 'weight_value')

        new_df = new_df.drop_duplicates()

        for weight_name in weight_columns:
            new_df = new_df.replace(weight_name, weight_name[7:-8])

        new_df['_run'] = new_df['_run'].astype(str)

        #new_df.to_csv('testAAA.csv')

        fig = px.bar(new_df, \
            x=new_df['weight_name'], \
            y="weight_value", \
            color="_run", \
            title="Best weights over runs",\
            barmode='group')

        return fig
