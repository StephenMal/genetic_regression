# Base Library Imports
import sys
from statistics import mean

# Other imports
import numpy as np

# Simpgenalg imports
from simpgenalg.evaluators.basics import basicEvaluator

# Tensorflow imports
from tensorflow import cast as tf_cast, \
                       constant as tf_constant, \
                       convert_to_tensor, is_tensor, float32, shape



try:
    from scipy.spatial.distance import squareform, pdist
except:
    pass

class regressionEvaluator(basicEvaluator):

    __slots__ = ('header', \
                 'train_feats', 'train_lbls', \
                 'test_feats', 'test_lbls', \
                 'loss_metric', 'L1', 'L2', \
                 'encode_toggles', \
                 'encode_exponents', 'exponents_keep_sign',\
                 'track_weight_diversity', 'calc_test_loss',\
                 'min_weight', 'max_weight')

    ''' Setup '''
    def __init__(self, *args, **kargs):

        # Initialize basic evaluator and basic component
        super().__init__(*args, **kargs)

        # Load in the data
        self.load_data(*args, **kargs)

        # LOADS PARAMETERS (Will check kargs before config)
        if 'min_weight' in kargs:
            self.min_weight = kargs.get('min_weight')
        else:
            self.min_weight = self.config.get('min_weight', dtype=(int, float))

        if 'max_weight' in kargs:
            self.max_weight = kargs.get('max_weight')
        else:
            self.max_weight = self.config.get('max_weight', dtype=(int, float))

        # Raise an error if the min_weight is greq the max weight
        if self.min_weight >= self.max_weight:
            raise ValueError('min_weight must be less than max_weight')

        if 'L1' in kargs:
            self.L1 = kargs.get('L1')
        else:
            self.L1 = self.config.get('L1', 0.0, mineq=0, dtype=(float,int))

        if 'L2' in kargs:
            self.L2 = kargs.get('L2')
        else:
            self.L2 = self.config.get('L2', 0.0, mineq=0, dtype=(float,int))

        if 'encode_toggles' in kargs:
            self.encode_toggles = kargs.get('encode_toggles')
        else:
            self.encode_toggles = self.config.get('encode_toggles', True, dtype=bool)

        if 'encode_exponents' in kargs:
            self.encode_exponents = kargs.get('encode_exponents')
        else:
            self.encode_exponents = \
                        self.config.get('encode_exponents', False, dtype=bool)

        if 'exponents_keep_sign' in kargs:
            self.exponents_keep_sign = kargs.get('exponents_keep_sign')
        else:
            self.exponents_keep_sign = \
                    self.config.get('exponents_keep_sign', True, dtype=bool)


        needed_genes = self._determine_n_needed_genes()
        n_genes = self.config.get('num_genes', needed_genes, dtype=int, min=0)
        if n_genes != needed_genes:
            raise ValueError(f'Neded at least {needed_genes} with given params,'\
                                    'not {n_genes}')

        if 'track_weight_diversity' in kargs:
            self.track_weight_diversity = kargs.get('track_weight_diversity')
        else:
            self.track_weight_diversity = \
                    self.config.get('track_weight_diversity', True, dtype=bool)

        if 'calc_test_loss' in kargs:
            self.calc_test_loss = kargs.get('calc_test_loss')
        else:
            self.calc_test_loss = \
                        self.config.get('calc_test_loss', False, dtype=bool)

    def load_data(self, *args, **kargs):
        # Converts passed data to tensor if not already a tensor
        def convert_to_tensor_if_not_tensor(data):
            if not is_tensor(data):
                return convert_to_tensor(data, float32)
            return data

        self.header = kargs.get('header')
        if self.header is None:
            self.header = self.config.get('header', None)

        # Load in train feats and train lbls
        self.train_feats = kargs.get('train_feats')
        if self.train_feats is None:
            self.train_feats = self.config.get('train_feats')

        self.train_lbls = kargs.get('train_lbls')
        if self.train_lbls is None:
            self.train_lbls = self.config.get('train_lbls')

        # Make sure it is a tensor
        self.train_feats = convert_to_tensor_if_not_tensor(self.train_feats)
        self.train_lbls = convert_to_tensor_if_not_tensor(self.train_lbls)

        # Load in the feats/lbls for test
        self.test_feats = kargs.get('test_feats')
        if self.test_feats is None:
            self.test_feats = self.config.get('test_feats')

        self.test_lbls = kargs.get('test_lbls')
        if self.test_lbls is None:
            self.test_lbls = self.config.get('test_lbls')

        # If provided test feats or test lbls but not the other raise an error
        if self.test_feats is None or self.test_lbls is None and \
            (self.test_feats is not None or self.test_lbls is not None):
                raise ValueError('Must provide both test_feats and test_lbls')
        elif self.test_feats is not None and self.test_lbls is not None:
            # Make sure it is a tensor
            self.test_feats = convert_to_tensor_if_not_tensor(self.test_feats)
            self.test_lbls = convert_to_tensor_if_not_tensor(self.test_lbls)

        return

    # _determine_n_needed_genes
    #   - Returns the number (int) of genes required for the fitfxn
    #   Outputs:
    #       - number (int) of genes required to run
    def _determine_n_needed_genes(self):
        # Get number of weights ( equal to # of feats ) + 1 for constant
        n_feats = int(shape(self.train_feats)[1])
        n_needed = n_feats + 1

        # Add an extra value per feature if toggles or exponents
        if self.encode_toggles:
            n_needed += n_feats
        if self.encode_exponents:
            n_needed += n_feats
        return n_needed

    # _decode_exponent
    #   - Given a value, returns respective exponent (1,2,3,4)
    #   Inputs:
    #       - Val = Value we are converting
    #       - vmin = Minimum value possible
    #       - vrange = Range of possible values
    #   Outputs:
    #       - exp (float) = 1.0,2.0,3.0,4.0
    @staticmethod
    def _decode_exponent(val, vmin, vrange):
        v = (val-vmin)/vrange
        if v < 0.50:
            return 1.0 if v < 0.25 else 2.0
        else:
            return 3.0 if v < 0.75 else 4.0

    # _decode_batch
    #   - Given a list of individuals, returns a list of decoded sets of weights
    #   Inputs:
    #       - btch = list object of individuals
    #   Outputs:
    #       - processed_indvs = list of decoded individuals' weights
    def _decode_batch(self, btch):

        # Get information about weights
        min_w, max_w = self.min_weight, self.max_weight
        range_w = max_w - min_w

        # Get information about features
        n_feats, header = self.train_feats.shape[1], self.header

        # See if we are using toggles or exponents
        encode_toggles, encode_exponents, keep_signs = \
            self.encode_toggles, self.encode_exponents, self.exponents_keep_sign

        def decode(indv):
            # Get minimum value encodable and range
            vmin, vrange = indv.get_valmin(), indv.get_valrange()

            # Map
            mapped = indv.get_mapped()

            # Create dictionary to store stats
            decode_stats = {}

            indx = 1 + n_feats
            constant = (mapped[0] - vmin / vrange)*range_w
            decode_stats['constant'] = constant
            weights = [(((v-vmin)/vrange)*range_w)+min_w for v in mapped[1:indx]]

            if encode_toggles:
                last_indx, indx = indx, indx+n_feats

                toggles = [v > 0 for v in mapped[last_indx:indx]]
                decode_stats['n_toggles'] = toggles.count(True)

                for i, toggle in enumerate(toggles):
                    if not toggle:
                        weights[i] = 0

                # Record weights
                if header is None:
                    decode_stats.update({f'toggle_{i}':tog \
                                           for i, tog in enumerate(toggles)})
                else:
                    decode_stats.update({f'toggle_{i}_{header[i]}':tog \
                                           for i, tog in enumerate(toggles)})

            if encode_exponents:
                # Determine indicies
                last_indx, indx = indx, indx+n_feats

                # Calculate exponents
                _decode_exponent = self._decode_exponent
                exps = [_decode_exponent(v, vmin, vrange) \
                                                for v in mapped[last_indx:indx]]

                if keep_signs:
                    for i, exp in enumerate(exps):
                        if weights[i] != 0 and exp != 1:
                            weights[i] = (weights[i] ** exp) \
                                            if weights[i] > 0 else \
                                            -(weights[i] ** exp)
                else:
                    for i, exp in enumerate(exps):
                        if weights[i] != 0 and exp != 1:
                            weights[i] = weights[i] ** exp

                if header is None:
                    decode_stats.update({f'exponent_{i}':exp \
                                           for i, exp in enumerate(exponents)})
                else:
                    decode_stats.update({f'exponent_{i}_{header[i]}':exp \
                                           for i, exp in enumerate(exponents)})

            if header is None:
                decode_stats.update({f'weight_{i}':w \
                                       for i, w in enumerate(weights)})
            else:
                decode_stats.update({f'weight_{i}_{header[i]}':w \
                                       for i, w in enumerate(weights)})

            decode_stats['w_lst'] = weights

            return (indv, constant, weights, decode_stats)

        return [decode(indv) for indv in btch]

    def _compare_weight_distance(self, processed):
        if 'scipy' not in sys.modules:
            self.log.exception('Scipy needed to find distance between '+\
                                'individuals.', err=ModuleNotFoundError)

        dist_mat = squareform(pdist([[x[1]]+x[2] for x in processed]))

        for i, (indv, constant, weights, decode_stats) in enumerate(processed):
            indv.set_attr('avg_w_dist',mean(dist_mat[i]))

    # Different cache functions
    def _get_none(self, constant, weights):
        return None
    def _get_bcache(self, constant, weights):
        key = (constant if isinstance(constant, float) else float(constant),\
               weights if isinstance(weights, tuple) else \
                    tuple([float(w) for w in weights]))
        return self.cache.get(key, self.sCache.get(key,None))
    def _get_scache(self, constant, weights):
        key = (constant if isinstance(constant, float) else float(constant),\
               weights if isinstance(weights, tuple) else \
                    tuple([float(w) for w in weights]))
        return self.sCache.get(key, None)
    def _get_ncache(self, constant, weights):
        key = (constant if isinstance(constant, float) else float(constant),\
               weights if isinstance(weights, tuple) else \
                    tuple([float(w) for w in weights]))
        return self.cache.get(key, None)
    def _set_none(self, constant, weights, fit, stats):
        return None
    def _set_bcache(self, constant, weights, fit, stats):
        key = (constant if isinstance(constant, float) else float(constant),\
               weights if isinstance(weights, tuple) else \
                    tuple([float(w) for w in weights]))
        self.cache[key] = (fit, stats)
        self.sCache[key] = (fit, stats)
    def _set_scache(self, constant, weights, fit, stats):
        key = (constant if isinstance(constant, float) else float(constant),\
               weights if isinstance(weights, tuple) else \
                    tuple([float(w) for w in weights]))
        self.sCache[key] = (fit, stats)
    def _set_ncache(self, constant, weights, fit, stats):
        key = (constant if isinstance(constant, float) else float(constant),\
               weights if isinstance(weights, tuple) else \
                    tuple([float(w) for w in weights]))
        self.cache[key] = (fit, stats)

    # Evaluates a singular individual
    def evaluate(self, indv):
        raise NotImplementedError

    # Evaluates a batch of indvs
    def evaluate_batch(self, btch, **kargs):
        # Decode the batch
        decoded = self._decode_batch(btch)
        # Get the evaluation fxn
        eval = self.evaluate
        # Iterate trhough
        for indv, c, w, stats in decoded:
            # Evaluate the individual, supply constant weights and stats
            eval(indv, constant=c, weights=w, stats=stats)
        # Compare weights
        if self.track_weight_diversity:
            try:
                dist_mat = squareform(pdist([[c]+w \
                                                for indv,c,w,stats in decoded]))
                for i, (indv, constant, weights, stats) in enumerate(decoded):
                    indv.set_attr('avg_w_dist',dist_mat[i].mean())
            except Exception as e:
                self.log.exception(str(e))
                raise Exception('Failed to perform weight distance')
        return
        # End of evaluate_batch

    @classmethod
    def predict(cls, c, w, feats):
        raise NotImplementedError

    # Returns a score depending on quality
    @classmethod
    def score(cls, preds=None, lbls=None, c=None, w=None, feats=None):
        raise NotImplementedError
