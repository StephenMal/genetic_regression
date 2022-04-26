# Other imports
import numpy as np

# Tensorflow imports
from tensorflow import cast as tf_cast, \
                       constant as tf_constant, \
                       convert_to_tensor, is_tensor, float32, int32, shape

from tensorflow.nn import sigmoid as tf_sig

from tensorflow.math import abs as tf_abs, \
                            square as tf_sqr, \
                            reduce_sum as tf_reduce_sum, \
                            greater_equal as tf_greq, \
                            equal as tf_eq, \
                            logical_and as tf_logical_and, \
                            logical_not as tf_logical_not

from tensorflow.compat.v1 import confusion_matrix as tf_conf_mat

from tensorflow.keras.losses import BinaryCrossentropy, \
                                    MeanAbsolutePercentageError, \
                                    MeanAbsoluteError, \
                                    MeanSquaredError, \
                                    MeanSquaredLogarithmicError

from .basic_regfit import regressionEvaluator

class logisticRegressionEvaluator(regressionEvaluator):

    def __init__(self, *args, **kargs):

        super().__init__(*args, **kargs)

        self.loss_metric = BinaryCrossentropy(from_logits=False)

    @classmethod
    def predict(cls, constant, weights, feats, return_prob=False):
        if not is_tensor(feats):
            feats = convert_to_tensor(feats, dtype=float32)
        if not is_tensor(weights):
            weights = convert_to_tensor(weights, dtype=float32)
        if not is_tensor(constant):
            constant = convert_to_tensor(constant, dtype=float32)
        if len(weights) != feats.shape[1]:
            raise ValueError('# of weights should equal # of features')
        if return_prob:
            return tf_sig(tf_reduce_sum(feats*weights, axis=1)+constant)
        return tf_cast(tf_greq(tf_sig(\
                    tf_reduce_sum(feats*weights, axis=1)+constant),0.5),int32)

    @classmethod
    def score(cls, preds=None, constant=None, weights=None, feats=None, lbls=None):
        if lbls is None:
            raise Exception('Must provide lbls')
        if preds is None:
            if constant is None or weights is None or feats is None:
                raise Exception('If not providing pred, must provide constant'+\
                                ', weights, and feats')
            # Predict the values
            preds = cls.predict(constant, weights, feats, return_prob=False)
        elif preds is not None:
            if not (constant is None and weights is None and feats is None):
                raise Exception('Must provide constant, weights, and feats if'+\
                                ' not providing predictions')

        # Unpack the confusion matrix
        (tp1, fp1), (fn1, tn1) = tf_conf_mat(lbls, preds, num_classes=2)
        del preds # Last time using prob, we can go ahead and delete it
        # Turn into integers
        (tp1, fp1), (fn1, tn1) = (int(tp1), int(fp1)), (int(fn1), int(tn1))
        # Get training accuracy
        return (tp1+tn1)/(fn1+fp1+tn1+tp1)


    def evaluate(self, indv, constant=None, weights=None, stats=None, check_cache=True):
        # If not provided, we need to calculate them
        if constant is None or weights is None or stats is None:
            constant, weights, stats = self._decode_batch([indv])[0]

        # If check cache, check it
        if check_cache:
            cval = self.get_cache(constant, weights)
            if cval is not None:
                fit, stats = cval
                indv.update_attrs(stats)
                indv.set_fit(fit)
                return

        # Get the constant and weights
        constant, weights = convert_to_tensor(constant, float32), \
                            convert_to_tensor(weights, float32)

        # Determine the penalty
        penalty1 = float(self.L1*tf_reduce_sum(tf_abs(weights))) \
                                                if self.L1 != 0 else 0
        penalty2 = float(self.L2*tf_reduce_sum(tf_abs(weights))) \
                                                if self.L2 != 0 else 0
        penalty = penalty1 + penalty2

        # Determine prediction of probabilities for training datas
        prob = tf_sig(tf_reduce_sum(self.train_feats*weights, axis=1)+constant)

        # Training Binary Cross Entropy
        train_bce = float(self.loss_metric(self.train_lbls, prob))

        # Unpack the confusion matrix
        (tp1, fp1), (fn1, tn1) = tf_conf_mat(self.train_lbls, \
                                     tf_cast(tf_greq(prob,0.5),int32),\
                                     num_classes=2)
        del prob # Last time using prob, we can go ahead and delete it

        # Turn into integers
        (tp1, fp1), (fn1, tn1) = (int(tp1), int(fp1)), (int(fn1), int(tn1))

        # Get training accuracy
        train_acc = (tp1+tn1)/(fn1+fp1+tn1+tp1)

        if self.test_feats is not None or self.test_lbls is not None:
            # Determine prediction of probabilities for training datas
            prob = tf_sig(tf_reduce_sum(self.test_feats*weights, axis=1)+constant)
            # Training Binary Cross Entropy
            if not self.calc_test_loss:
                stats['test_bce'] = float(self.loss_metric(self.test_lbls, prob))

            # Unpack the confusion matrix
            (tp2, fp2), (fn2, tn2) = tf_conf_mat(self.test_lbls, \
                                         tf_greq(prob,0.5),\
                                         num_classes=2)
            del prob # Last time using prob, we can go ahead and delete it

            # Turn into integers
            (tp2, fp2), (fn2, tn2) = (int(tp2), int(fp2)), (int(fn2), int(tn2))
            # Get test acc
            test_acc = (tp2+tn2) / (fn2+fp2+tn2+tp2)

            stats.update({'L1':penalty1, 'L2':penalty2, 'penalty':penalty,\
                          'train_bce':train_bce, 'train_acc':train_acc,\
                          'test_acc':test_acc,\
                          'train_tp':tp1, 'train_tn':tn1,\
                          'train_fp':fp1, 'train_fp':fn1,\
                          'test_tp':tp2, 'test_tn':tn2,\
                          'test_fp':fp2, 'test_fn':fn2})
        else:
            stats.update({'L1':penalty1, 'L2':penalty2, 'penalty':penalty,\
                          'train_bce':train_bce, 'train_acc':train_acc,\
                          'train_tp':tp1, 'train_tn':tn1,\
                          'train_fp':fp1, 'train_fp':fn1})

        indv.update_attrs(stats)
        indv.set_fit(train_bce)
        self._replace_if_best(indv)
        self.set_cache(constant, weights, train_bce, stats)
        return
