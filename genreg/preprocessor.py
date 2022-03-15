import numpy as np

class preprocessor():

    def __init__(self, **kargs):

        # Booleans of whether we are applying standardization / normalization
        self.standardize = kargs.get('standardize', False)
        self.normalize = kargs.get('normalize', False)

        # Functions to apply standardization / normalization
        self.apply_std = kargs.get('apply_std', None)
        self.apply_nrm = kargs.get('apply_nrm', None)

        # Functions to undo standardization / normalization
        self.undo_std = kargs.get('undo_std', None)
        self.undo_nrm = kargs.get('undo_nrm', None)


    # Fitting for normalization
    def _fit_normalize(self, feats=None, lbls=None):
        # Raise an error if not passed nparrays
        if feats is not None and not isinstance(feats, np.ndarray):
            raise TypeError(f'Expected numpy ndarray, not {type(feats)}')
        if lbls is not None and not isinstance(lbls, np.ndarray):
            raise TypeError(f'Expected numpy ndarray, not {type(lbls)}')

        # Function that returns the minimum, maximum, and range of values
        #   in each column of a matrix
        def get_nrm_params(values):
            # Determine max, min, range
            vmin, vmax = values.min(axis=0), values.max(axis=0)
            vrange = vmax-vmin
            return (vmin, vmax, vrange)

        # Get parameters if provided data (otherwise assume not wanted)
        f_params = None if feats is None else get_nrm_params(feats)
        l_params = None if lbls is None else get_nrm_params(lbls)

        # Performs the normalizations
        def apply_nrm(values, params):
            # Extract parameters
            vmin, vmax, vrange = params
            # Apply it to the values
            if len(values.shape) == 2:
                return np.array(\
                        [[((float(v) - float(min))/(float(range)))  \
                                    for v, min, range in zip(row, vmin, vrange)]\
                                                            for row in values])
            elif len(values.shape) == 1:
                return np.array([((float(v) - float(vmin))/(float(vrange)))  \
                                        for v in values])

        def undo_nrm(values, params):
            # Extract parameters
            vmin, vmax, vrange = params
            # Apply it to the values
            if len(values.shape) == 2:
                return np.array(\
                        [[((float(v)*float(range))+float(min)) \
                                    for v, min, range in zip(row, vmin, vrange)]\
                                                        for row in values])
            elif len(values.shape) == 1:
                return np.array([((float(v)*float(vrange))+float(vmin)) \
                                        for v in values])


        # Create a function that will apply these on command
        def nrmfxn(feats=None, lbls=None):
            # Raise an error if not passed nparrays
            if feats is not None and not isinstance(feats, np.ndarray):
                raise TypeError(f'Expected numpy ndarray, not {type(feats)}')
            if lbls is not None and not isinstance(lbls, np.ndarray):
                raise TypeError(f'Expected numpy ndarray, not {type(lbls)}')

            # Apply to features
            dct = {}
            if f_params is not None and feats is not None:
                dct['feats'] = apply_nrm(feats, f_params)
            if l_params is not None and lbls is not None:
                dct['lbls'] = apply_nrm(lbls, l_params)

            return dct

        def undo_nrmfxn(feats=None, lbls=None):
            # Raise an error if not passed nparrays
            if feats is not None and not isinstance(feats, np.ndarray):
                raise TypeError(f'Expected numpy ndarray, not {type(feats)}')
            if lbls is not None and not isinstance(lbls, np.ndarray):
                raise TypeError(f'Expected numpy ndarray, not {type(lbls)}')

            # Apply to features
            dct = {}
            if f_params is not None and feats is not None:
                dct['feats'] = undo_nrm(feats, f_params)
            if l_params is not None and lbls is not None:
                dct['lbls'] = undo_nrm(lbls, l_params)

            return dct

        return nrmfxn, undo_nrmfxn

    # Fitting for standardization
    def _fit_standardize(self, feats=None, lbls=None):
        # Raise an error if not passed nparrays
        if feats is not None and not isinstance(feats, np.ndarray):
            raise TypeError(f'Expected numpy ndarray, not {type(feats)}')
        if lbls is not None and not isinstance(lbls, np.ndarray):
            raise TypeError(f'Expected numpy ndarray, not {type(lbls)}')

        # Function that returns the mean and stdev in each column of a matrix
        def get_std_params(values):
            # Determine mean and stdev of each column
            vmean, vstdev = values.mean(axis=0), values.std(axis=0)
            return (vmean, vstdev)

        # Get parameters if provided data (otherwise assume not wanted)
        f_params = None if feats is None else get_std_params(feats)
        l_params = None if lbls is None else get_std_params(lbls)

        # Performs the normalizations
        def apply_std(values, params):
            # Extract parameters
            vmean, vstdev = params
            # Apply it to the values
            if len(values.shape) == 2:
                return np.array(\
                        [[((float(v) - float(mean))/(float(stdev)))  \
                                for v, mean, stdev in zip(row, vmean, vstdev)]\
                                                            for row in values])
            elif len(values.shape) == 1:
                return np.array([((float(v) - float(vmean))/(float(vstdev)))  \
                                    for v in values])
            else:
                raise ValueError('Expected shape of 1 or 2 dimensions')

        # Performs the normalizations
        def undo_std(values, params):
            # Extract parameters
            vmean, vstdev = params
            # Apply it to the values
            if len(values.shape) == 2:
                return np.array(\
                        [[((float(v)*float(stdev))+float(mean)) \
                                for v, mean, stdev in zip(row, vmean, vstdev)]\
                                                            for row in values])
            elif len(values.shape) == 1:
                return np.array([((float(v)*float(vstdev))+float(vmean))
                                    for v in values])
            else:
                raise ValueError('Expected shape of 1 or 2 dimensions')

        # Create a function that will apply these on command
        def stdfxn(feats=None, lbls=None):
            # Raise an error if not passed nparrays
            if feats is not None and not isinstance(feats, np.ndarray):
                raise TypeError(f'Expected numpy ndarray, not {type(feats)}')
            if lbls is not None and not isinstance(lbls, np.ndarray):
                raise TypeError(f'Expected numpy ndarray, not {type(lbls)}')

            # Apply to features
            dct = {}
            if f_params is not None and feats is not None:
                dct['feats'] = apply_std(feats, f_params)
            if l_params is not None and lbls is not None:
                dct['lbls'] = apply_std(lbls, l_params)

            return dct

        # Create a function that will apply these on command
        def undo_stdfxn(feats=None, lbls=None):
            # Raise an error if not passed nparrays
            if feats is not None and not isinstance(feats, np.ndarray):
                raise TypeError(f'Expected numpy ndarray, not {type(feats)}')
            if lbls is not None and not isinstance(lbls, np.ndarray):
                raise TypeError(f'Expected numpy ndarray, not {type(lbls)}')

            # Apply to features
            dct = {}
            if f_params is not None and feats is not None:
                dct['feats'] = undo_std(feats, f_params)
            if l_params is not None and lbls is not None:
                dct['lbls'] = undo_std(lbls, l_params)

            return dct

        return stdfxn, undo_stdfxn

    # Fit the preprocessor
    def fit(self, feats=None, lbls=None, **kargs):

        # If provided values, turn into numpy arrays (if not already)
        if feats is not None and not isinstance(feats, np.ndarray):
            feats = np.array(feats)
        if lbls is not None and not isinstance(lbls, np.ndarray):
            lbls = np.array(lbls)

        if 'standardize' in kargs:
            self.standardize = kargs.get('standardize')
        if 'normalize' in kargs:
            self.normalize = kargs.get('normalize')

        if self.standardize and self.normalize:
            raise ValueError('Cannot apply both standardization and normalization')

        # Builds normalization function and nulls any pre-existing standardization
        if self.normalize:
            self.apply_std, self.undo_std = None, None
            self.apply_nrm, self.undo_nrm = \
                                    self._fit_normalize(feats=feats, lbls=lbls)

        # Builds standardization function and nulls any pre-existing normalization
        if self.standardize:
            self.apply_nrm, self.undo_nrm = None, None
            self.apply_std, self.undo_std = \
                                self._fit_standardize(feats=feats, lbls=lbls)

        return

    # Applies the preprocessing
    def apply(self, feats=None, lbls=None):
        # Raises error if has fxn for both
        if self.apply_nrm is not None and self.apply_std is not None:
            raise Exception('Preprocessor has standardization and '+\
                            'normalization which should not be possible.')
        elif self.apply_nrm is not None:
            return self.apply_nrm(feats=feats, lbls=lbls)
        elif self.apply_std is not None:
            return self.apply_std(feats=feats, lbls=lbls)
        else:
            return {'feats':feats, 'lbls':lbls}

    # Undoes the preprocessing
    def undo(self, feats=None, lbls=None):
        # Raises error if has fxn for both
        if self.undo_nrm is not None and self.undo_std is not None:
            raise Exception('Preprocessor has standardization and '+\
                            'normalization which should not be possible.')
        elif self.undo_nrm is not None:
            return self.undo_nrm(feats=feats, lbls=lbls)
        elif self.undo_std is not None:
            return self.undo_std(feats=feats, lbls=lbls)
        else:
            return {'feats':feats, 'lbls':lbls}
