from simpgenalg.evaluators.basics import basicEvaluator

import sys
import numpy as np
from statistics import mean

try:
    from scipy.spatial.distance import squareform, pdist
except:
    pass

# Import necessary tensorflow functions
from tensorflow.math import abs as tf_abs
from tensorflow.math import square as tf_sqr
from tensorflow.math import reduce_sum as tf_reduce_sum
from tensorflow.math import greater_equal as tf_greq
from tensorflow.math import equal as tf_eq
from tensorflow import cast as tf_cast
from tensorflow import constant as tf_constant
from tensorflow import convert_to_tensor, is_tensor, float32, shape
from tensorflow.keras.losses import BinaryCrossentropy, \
                                    MeanAbsolutePercentageError, \
                                    MeanAbsoluteError, MeanSquaredError, \
                                    MeanSquaredLogarithmicError

# Base regression evaluator, does everything but the evaluation
class regressionEvaluator(basicEvaluator):

    __slots__ = ('header', \
                 'train_feats', 'train_lbls',\
                 'test_feats', 'test_lbls',\
                 'min_weight', 'max_weight',\
                 'loss_metric', 'L1', 'L2',\
                 'encode_toggles', \
                 'encode_exponents', 'exponents_keep_sign',\
                 'track_weight_diversity')

    def __init__(self, *args, **kargs):
        # Initialize basic evaluator and basic component
        super().__init__(*args, **kargs)

        # Load in the data
        self.load_data(*args, **kargs)

        # Get min/max weight
        self.min_weight = kargs.get('min_weight', \
                                self.config.get('min_weight', dtype=float))
        self.max_weight = kargs.get('max_weight', \
                                self.config.get('max_weight', dtype=float))
        if self.min_weight >= self.max_weight:
            raise ValueError('min_weight must be less than max_weight')

        self.L1 = kargs.get('L1', self.config.get('L1', 0.0, mineq=0, dtype=float))
        self.L2 = kargs.get('L2', self.config.get('L2', 0.0, mineq=0, dtype=float))

        # Whether we are encoding toggle values (zeros out weights)
        self.encode_toggles = kargs.get('encode_toggles', \
                    self.config.get('encode_toggles', True, dtype=bool))

        # Whether we are encoding exponents (applied to weights)
        self.encode_exponents = kargs.get('encode_exponents', \
                    self.config.get('encode_exponents', False, dtype=bool))

        if self.encode_exponents:
            # WHether we keep the signs if applied to the weights
            self.exponents_keep_sign = kargs.get('exponents_keep_sign', \
                        self.config.get('exponents_keep_sign', True, dtype=bool))
        else:
            self.exponents_keep_sign = False

        # Verify correct number of genes
        needed_genes = self._determine_n_needed_genes()
        n_genes = self.config.get('num_genes', needed_genes, dtype=int, min=0)
        if n_genes != needed_genes:
            raise ValueError(f'Needed at least {needed_genes}, not {n_genes}')

        self.track_weight_diversity = \
                self.config.get('track_weight_diversity', True, dtype=bool)

    def _determine_n_needed_genes(self):
        # Get number of features
        n_feats = int(shape(self.train_feats)[1])
        # Add 1 for the constant
        n_needed = n_feats + 1
        # Add 1 per feat for toggle or exponents
        if self.encode_toggles:
            n_needed += n_feats
        if self.encode_exponents:
            n_needed += n_feats
        return n_needed

    # Using a value range, turns a value into a respective integer exponent
    @staticmethod
    def _decode_exponent(val, vmin, vrange):
        v = (val-vmin)/vrange
        if v < 0.25:
            return 1.0
        elif v < 0.50:
            return 2.0
        elif v < 0.75:
            return 3.0
        elif v <= 1.0:
            return 4.0

    # Decodes a batch of individuals
    def _decode_batch(self, btch, **kargs):

        # Get the minimum and maximum weight values allowed
        min_w, max_w = kargs.get('min_weight',self.min_weight),\
                       kargs.get('max_weight', self.max_weight)
        # Find the range and half range
        range_w = max_w - min_w

        # Get training data info
        n_feats = kargs.get('n_feats', self.train_feats.shape[1])
        header = kargs.get('header', self.header)

        # Optional encodings
        encode_toggles, encode_exponents, keep_signs  = \
            kargs.get('encode_toggles', self.encode_toggles),\
            kargs.get('encode_exponents', self.encode_exponents),\
            kargs.get('exponents_keep_sign', self.exponents_keep_sign)

        # Iterate through each individual
        processed_indvs = []
        for indv in btch:
            # Get value min / range of individual
            vmin, vrange = indv.get_valmin(), indv.get_valrange()

            # Remap the range to fit within the given range
            mapped = indv.get_mapped()

            # Create dictionary to store stats
            decode_stats = {}

            indx = 1 + n_feats
            weights = [(((v-vmin)/vrange)*range_w)+min_w for v in mapped[:indx]]
            constant, weights = weights[0], weights[1:indx]
            decode_stats['constant'] = constant

            # Applies toggles
            if encode_toggles:
                # Determine index
                last_indx, indx = indx, indx+n_feats

                # Calculate toggles
                toggles = [mapped[i] > 0 for i in range(last_indx, indx)]
                decode_stats['n_toggled'] = \
                                len([None for toggle in toggles if toggle])

                # Apply toggles to the weight list
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

            # Applies exponents
            if encode_exponents:
                # Determine indicies
                last_indx, indx = indx, indx+n_feats

                # Calculate exponents
                exponents = [self._decode_exponent(mapped[i], vmin, vrange) \
                                for i in range(last_indx, indx)]

                if keep_signs: # If keeping signs, make sure to check signs
                    for i, exp in enumerate(exponents):
                        if exp != 1 and weights[i] != 0:
                            weights[i] = (weights[i] ** exp) \
                                                if weights[i] > 0 else \
                                                -1*(weights[i] ** exp)
                else: # Otherwise do however
                    for i, exp in enumerate(exponents):
                        if exp != 1 and weights[i] != 0:
                            weights[i] = weights[i] ** exp

                if header is None:
                    decode_stats.update({f'exponent_{i}':exp \
                                           for i, exp in enumerate(exponents)})
                else:
                    decode_stats.update({f'exponent_{i}_{header[i]}':exp \
                                           for i, exp in enumerate(exponents)})

            if header is None:
                decode_stats.update({f'weight_{i}':exp \
                                       for i, exp in enumerate(exponents)})
            else:
                decode_stats.update({f'weight_{i}_{header[i]}':exp \
                                       for i, exp in enumerate(weights)})

            # Add individual to list of processed individuals
            processed_indvs.append((indv, constant, weights, decode_stats))

        return processed_indvs

    # Calculate the weights given the weights as a tensor
    def calc_penalty(self, weights, **kargs):
        # Get the L1 and L2 weights from kargs or from the object
        L1, L2, = kargs.get('L1', self.L1), kargs.get('L2', self.L2)
        # Penalty by default should be 0
        penalty1, penalty2 = 0, 0
        # Skip if L1 is 0
        if L1 != 0:
            # Abs each element then sum
            penalty1 = float(L1 * tf_reduce_sum(tf_abs(weights)))
        # Skip if L2 is 0
        if L2 != 0:
            # Square each element, sum
            penalty2 = float(L2 * tf_reduce_sum(tf_sqr(weights)))
        # Return the two penalties
        return penalty1, penalty2, penalty1+penalty2

    # Load in the data
    def load_data(self, *args, **kargs):
        # Converts passed data to tensor if not already a tensor
        def convert_to_tensor_if_not_tensor(data):
            if not is_tensor(data):
                return convert_to_tensor(data, float32)
            return data

        self.header = kargs.get('header', self.config.get('header', None))

        # Load in training data
        self.train_feats = kargs.get('train_feats', \
                                            self.config.get('train_feats'))
        self.train_lbls = kargs.get('train_lbls', \
                                            self.config.get('train_lbls'))
        # Make sure it is a tensor
        self.train_feats = convert_to_tensor_if_not_tensor(self.train_feats)
        self.train_lbls = convert_to_tensor_if_not_tensor(self.train_lbls)

        # Load in test data
        self.test_feats = kargs.get('test_feats', \
                                            self.config.get('test_feats', None))
        self.test_lbls = kargs.get('test_lbls', \
                                            self.config.get('test_lbls', None))

        # If provided test feats or test lbls but not the other raise an error
        if self.test_feats is None or self.test_lbls is None and \
            (self.test_feats is not None or self.test_lbls is not None):
                raise ValueError('Must provide both test_feats and test_lbls')
        elif self.test_feats is not None and self.test_lbls is not None:
            # Make sure it is a tensor
            self.test_feats = convert_to_tensor_if_not_tensor(self.test_feats)
            self.test_lbls = convert_to_tensor_if_not_tensor(self.test_lbls)

        return

    # True evaluation process (unique to lin reg vs log reg vs multiclass log reg)
    def _evaluate(self, indv, **kargs):
        # SET CACHE INSIDE _EVALUATE
        raise NotImplementedError

    # Checks cache, applies evaluation if needed
    def evaluate(self, indv, **kargs):

        # Try reading the cache
        cached_val = self.get_cache(indv)

        # If fit is None, calculate loss
        if cached_val is None:
            fit, attrs = self._evaluate(indv, **kargs)
        else:
            fit, attrs = cached_val

        # Update individual's values
        indv.update_attrs(attrs)
        indv.set_fit(fit)

        # Replaces if the best
        self._replace_if_best(indv)

        return

    def evaluate_batch(self, btch, **kargs):

        # Check caches for each individual
        cache_checks = [(indv, self.get_cache(indv)) for indv in btch]

        # Decode individuals who need to be evaluated
        decoded = self._decode_batch([indv for (indv, cached_vals) \
                                        in cache_checks if cached_vals is None])
        # Evaluate those not in cache that we just decoded
        evaluated = [(indv, self._evaluate(indv,\
                                        constant=constant,\
                                        weights=weights,\
                                        stats = decode_stats))\
                                        for indv, constant, weights, decode_stats\
                                            in decoded]

        if self.track_weight_diversity:
            self.compare_weight_distance(decoded)

        # Update attributes, set fitness, etc
        for indv, (fit, attrs) in evaluated:
            indv.update_attrs(attrs)    # Update attributes
            indv.set_fit(fit)           # Set fitness
            self._replace_if_best(indv) # Replace if best

        return

    def compare_weight_distance(self, processed):
        if 'scipy' not in sys.modules:
            self.log.exception('Scipy needed to find distance between '+\
                                'individuals.', err=ModuleNotFoundError)

        dist_mat = squareform(pdist([[x[1]]+x[2] for x in processed]))

        for i, (indv, constant, weights, decode_stats) in enumerate(processed):
            indv.set_attr('avg_w_dist',mean(dist_mat[i]))


    # Makes a singular prediction
    @classmethod
    def predict(cls, constant, weights, features):
        raise NotImplementedError

    # Makes batch prediction
    @classmethod
    def predict_batch(cls, constant, weights, features):
        return [cls.predict(constant, weights, feats) for feats in features]

# Logistic regression evaluator
class logisticRegressionEvaluator(regressionEvaluator):

    __slots__ = ('bndry')

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        self.bndry = kargs.get('decision_boundary', \
                        self.config.get('decision_boundary', 0.5, dtype=float))

        if not is_tensor(self.bndry):
            self.bndry = convert_to_tensor(self.bndry, float32)

        self.loss_metric = BinaryCrossentropy(from_logits=True)

    def _score(self, true, predictions):
        return float(tf_reduce_sum(tf_cast(\
                                tf_eq(predictions, true),\
                                float32)))/len(predictions)

    # Apply logistic regression and evaluate the binary cross entropy of train
    #   and test along with the accuracy.
    def _evaluate(self, indv, *args, **kargs):
        # Get constant, weights, and stats dict
        constant, weights, stats = kargs.get('constant', None), \
                                   kargs.get('weights', None),\
                                   kargs.get('stats', None)

        # If not provided, run decode batch and just grab the singular item
        if constant is None or weights is None or stats is None:
            constant, weights, stats = self._decode_batch([indv])[0]

        # Convert tensor
        constant, weights = convert_to_tensor(constant, float32), \
                            convert_to_tensor(weights, float32)

        # Calculate penalty
        L1_penalty, L2_penalty, penalty = self.calc_penalty(weights)

        # Calculate weighted sums
        w_sums = tf_reduce_sum(self.train_feats*weights, axis=1)+constant

        # Calculate Binary-Cross Entropy
        train_bce = float(self.loss_metric(self.train_lbls, w_sums))

        # Calculate accuracy
        predictions = tf_cast(tf_greq(w_sums, self.bndry), float32)
        train_acc = float(tf_reduce_sum(tf_cast(\
                                tf_eq(predictions, self.train_lbls),\
                                float32)))/len(predictions)

        # Evaluate on test if test case, otherwise just add current stats
        if self.test_feats is not None or self.test_lbls is not None:
            # Calculate weighted sum\s
            w_sums = tf_reduce_sum(self.test_feats*weights, axis=1)+constant

            # Calculate Binary-Cross Entropy
            test_bce = float(self.loss_metric(self.test_lbls, w_sums))

            # Calculate accuracy
            predictions = tf_cast(tf_greq(w_sums, self.bndry), float32)
            test_acc = float(tf_reduce_sum(tf_cast(\
                                    tf_eq(predictions, self.test_lbls),\
                                    float32)))/len(predictions)

            # Update all stats before sending back
            stats.update({'L1':L1_penalty, 'L2':L2_penalty, 'penalty':penalty,\
                          'train_bce':train_bce, 'train_acc':train_acc,\
                          'test_bce':test_bce, 'test_acc':test_acc})
        else:
            # Update all stats before sending back
            stats.update({'L1':L1_penalty, 'L2':L2_penalty, 'penalty':penalty,\
                          'train_bce':train_bce, 'train_acc':train_acc})
        # Return training bce as fit and the stats
        return train_bce+penalty, stats

    # Makes a singular prediction
    @classmethod
    def predict(cls, weights, features):
        return tf_greq(tf_reduce_sum(features*weights, axis=1)+constant, \
                                                                self.bndry)[0]

# Linear regression evaluator
class linearRegressionEvaluator(regressionEvaluator):

    __slots__ = ()

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

        loss_metric_type = self.config.get('loss_metric','mape',dtype=str)

        if loss_metric_type == 'MAPE':
            self.loss_metric = MeanAbsolutePercentageError()
        elif loss_metric_type == 'MAE':
            self.loss_metric = MeanAbsoluteError()
        elif loss_metric_type == 'MSE':
            self.loss_metric = MeanSquaredError()
        elif loss_metric_type == 'MSLE':
            self.loss_metric = MeanSquaredLogarithmicError()

    def _score(self, true, predictions):
        return 1 - \
            (tf_sqr(true - predictions).sum() / tf_sqr(true-true.mean()).sum())

    # Apply logistic regression and evaluate the binary cross entropy of train
    #   and test along with the accuracy.
    def _evaluate(self, indv, *args, **kargs):
        # Get constant, weights, and stats dict
        constant, weights, stats = kargs.get('constant', None), \
                                   kargs.get('weights', None),\
                                   kargs.get('stats', None)

        # If not provided, run decode batch and just grab the singular item
        if constants is None or weights is None or stats is None:
            constant, weights, stats = self._decode_batch([indv])[0]

        # Convert tensor
        constant, weights = convert_to_tensor(constant, float32), \
                            convert_to_tensor(weights, float32)

        # Calculate penalty
        L1_penalty, L2_penalty, penalty = self.calc_penalty(weights)

        # Calculate weighted sums
        w_sums = tf_reduce_sum(self.train_feats*weights, axis=1)+constant

        # Calculate loss
        train_loss = float(self.loss_metric(self.train_lbls, w_sums))

        # Score
        train_coeff_determination = \
                self._score(self.train_lbls, w_sums)


        # Evaluate on test if test case, otherwise just add current stats
        if self.test_feats is not None or self.test_lbls is not None:
            # Calculate weighted sum\s
            w_sums = tf_reduce_sum(self.test_feats*weights, axis=1)+constant

            # Calculate Binary-Cross Entropy
            test_loss = float(self.loss_metric(self.test_lbls, w_sums))

            # Score
            test_coeff_determination = \
                    self._score(self.train_lbls, w_sums)


            # Update all stats before sending back
            stats.update({'L1':L1_penalty, 'L2':L2_penalty, 'penalty':penalty,\
                          'train_loss':train_loss, 'test_loss':train_loss, \
                          'train_score':train_coeff_determination,\
                          'test_score':test_coeff_determination})
        else:
            # Update all stats before sending back
            stats.update({'L1':L1_penalty, 'L2':L2_penalty, 'penalty':penalty,\
                          'train_loss':train_loss,\
                          'train_score':train_coeff_determination})

        # Return training bce as fit and the stats
        return train_loss+penalty, stats

    @classmethod
    def predict(cls, weights, features):
        return float(tf_reduce_sum(features*weights, axis=1)+constant)
