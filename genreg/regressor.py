from collections import namedtuple
from .preprocessor import preprocessor
import pandas as pd
from simpgenalg import geneticAlgorithm

try:
    import plotly.express as px
    import plotly.graph_objects as go
except:
    pass

class regressionModel():

    __slots__ = ('models', 'preprocess',\
                 'preprocessor','standardize', 'normalize', 'preprocess_lbls',\
                 'L1','L2', 'fit_fxn',\
                 'opt_params')

    dflt_ga_params = {'len':None,\
                      'chr_max':1,\
                      'chr_min':0,\
                      'xov_op':'onept',\
                      'mut_op':'uniform_mutation',\
                      'mut_rate':0.2,\
                      'xov_rate':0.9,\
                      'maximize':False,\
                      'cmpr_map_dist':False}

    model_tpl = namedtuple('model',['constant','weights','train_acc','test_acc'])

    def __init__(self, *args, **kargs):

        # Either load the models or create a list to store
        #   models post fit
        self.models = kargs.get('models')
        if self.models is None:
            self.models = []
        elif not isinstance(self.models, list):
            raise TypeError(f'models should be a list, not {type(self.models)}')
        elif any([not isinstance(m, namedtuple) for m in self.models]):
            raise TypeError('models should be a list of namedtuples')

        # Create the preprocessor
        self.preprocess = kargs.get('preprocess', True)
        self.preprocess_lbls = kargs.get('preprocess_lbls', False)
        if not isinstance(self.preprocess, bool):
            raise TypeError('preprocess should be a boolean')
        elif self.preprocess:
            self.preprocessor = kargs.get('preprocessor')
            if self.preprocessor is None:
                self.preprocessor = preprocessor(\
                            normalize=kargs.get('normalize', False),\
                            standardize=kargs.get('standardize', False),\
                            stats=kargs.get('stats', None))
                if self.preprocessor.normalize == False and \
                        self.preprocessor.standardize == False:
                    print('By default, normalize and standardize are False. '+\
                          'Both are currently False, so it is the same as not'+\
                          ' performing preprocessing')
            elif not isinstance(self.preprocessor, preprocessor):
                raise TypeError('Expected a preprocessor object from genreg lib'\
                        +'. If using another preprocessor, preprocess before'+\
                         'and set preprocess to False')


        # Load optimizer parameteres
        self.opt_params = kargs.get('opt_params')
        if self.opt_params is None:
            self.opt_params = self.dflt_ga_params.copy()
        elif not isinstance(self.opt_params, dict):
            raise TypeError('opt_params should be a dictionary of '+\
                            'optimizer parameters')

        self.L1, self.L2 = float(kargs.get('L1',0)), float(kargs.get('L2',0))
        if not isinstance(self.L1, (int, float)) or \
                   not isinstance(self.L2, (int, float)):
           raise TypeError('L1 and L2 should be float values')

        self.last_results = kargs.get('last_results', None)

    def fit(self, feats, lbls, test_feats=None, test_lbls=None):
        # Build the preprocessor & apply it to the features (optional for lbls)
        if self.preprocess:
            if self.preprocess_lbls:
                self._build_preprocessor(feats, lbls=lbls)
                feats, lbls = self._preprocess(feats=feats, lbls=lbls)
            else:
                self._build_preprocessor(feats)
                feats = self._preprocess(feats=feats)
        # Create the GA
        params = self.opt_params.copy()
        params.update({'L1':self.L1, 'L2':self.L2,\
                       'train_feats':feats, 'train_lbls':lbls,\
                       'evaluator':self.fit_fxn,\
                       'maximize':False, \
                       'tracking_vars':('fit.max', 'fit.min', 'fit.stdev',\
                                        'train_acc.genbest', 'train_acc.mean',\
                                        'train_acc.genbest')})
        if test_feats is not None and test_lbls is not None:
            if test_feats is None or test_lbls is None:
                raise ValueError('Must pass both test_feats and test_lbls or '+\
                                 'none at all')
            params.update({'test_feats':test_feats, 'test_lbls':test_lbls})

        gen_alg = geneticAlgorithm(**params)
        results = gen_alg.run(**params)
        self.save_results(results)

        # Create models from all the generations
        mtpl = self.model_tpl
        run_bests = [run.get_best() for run in results]
        # Save the best models from all the runs for potential ensembles
        self.models = [mtpl(rbst['constant'], rbst['w_lst'], \
                            rbst['train_acc'], rbst.get('test_acc',None)) \
                                for rbst in run_bests]

    def _build_preprocessor(self, feats, lbls=None):
        if self.preprocessor is None:
            self.preprocessor = preprocessor()
        if lbls is not None and self.preprocess_lbls:
            self.preprocessor.fit(feats=feats, lbls=lbls)
        else:
            self.preprocessor.fit(feats=feats)

    def _preprocess(self, feats=None, lbls=None):
        if self.preprocess == False:
            if lbls is None:
                return feats
            return feats,lbls
        if lbls is not None and self.preprocess_lbls:
            return self.preprocessor.transform(feats=feats, lbls=lbls)
        else:
            return self.preprocessor.transform(feats=feats)

    def clear_models(self):
        self.models = []

    # Adds a model to the list of models
    def add_model(self, constant, weights, train_acc=None, test_acc=None):
        try:
            if not isinstance(constant, float):
                constant = float(constant)
        except Exception as e:
            self.log.exception(str(e))
            raise Exception('Failed to convert constant to float')

        try:
            if not isinstance(weight, list) and not \
                                    all([isinstance(w,float) for w in weights]):
                weights = [float(w) for w in weights]
        except Exception as e:
            self.log.exception(str(e))
            raise Exception('Failed to convert weights to list of floats')

        if train_acc is not None:
            try:
                if not isinstance(train_acc, float):
                    train_acc = float(train_acc)
                if train_acc < 0 or train_acc > 1.0:
                    raise ValueError('train_acc needs to be btwn 0 and 1.0')
            except Exception as e:
                self.log.exception(str(e))
                raise Exception('Failed to get train_acc as float')

        if test_acc is not None:
            try:
                if not isinstance(test_acc, float):
                    test_acc = float(test_acc)
                if test_acc < 0 or test_acc > 1.0:
                    raise ValueError('test_acc needs to be btwn 0 and 1.0')
            except Exception as e:
                self.log.exception(str(e))
                raise Exception('Failed to get test_acc as float')

        if not isinstance(self.models, list):
            self.models = []

        self.models.append(\
                        self.model_tpl(constant, weights, train_acc, test_acc))

        # END OF ADD_MODEL

    def save_results(self, results):
        self.last_results = results

    def get_results(self):
        return self.last_results

    def get_wmat(self, incl_constant=True):
        if incl_constant:
            return [[m.constant]+m.weights for m in self.models]
        return [m.weights for m in self.models]

    def set_wmat(self, wmat):
        self.models = wmat

    def get_preproc_dict(self):
        if self.preprocessor is not None:
            return self.preprocessor.to_dict()
        return None

    def set_preproc_dict(self, dct):
        self.preprocessor = preprocessor()
        self.preprocessor.from_dict(dct)

    def score(self, feats, lbls, **kargs):
        raise NotImplementedError

    def predict(self, feats, **kargs):
        raise NotImplementedError

    ''' Optional Plots '''
    def get_best_run_weights_plot(self, stats_df=None):
        if stats_df is None:
            results = self.get_results()
            indvs, stats = results.to_df()
        else:
            stats = stats_df

        # Get weight columns
        weight_columns = ['constant.runbest'] + [col for col in stats.columns \
                                if ('weight_' in col and '.runbest' in col)]
        df_cols = weight_columns + ['_run', 'fit.runbest']

        def sort_fxn(key):
            if key == 'constant.runbest':
                return -1
            return int(key.split('_')[1])

        new_df = stats.melt(id_vars = ['_run', 'fit.runbest'], \
                            value_vars = sorted(weight_columns, key=sort_fxn), \
                            var_name = 'weight_name',\
                            value_name = 'weight_value')

        new_df = new_df.drop_duplicates()

        for weight_name in weight_columns:
            if weight_name == 'constant.runbest':
                new_df = new_df.replace(weight_name, 'constant')
            else:
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
