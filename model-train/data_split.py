import random
import numpy as np

class DataSplit:

    """
    Split a time-series dataset into training, test, and validation subsets
    using a sliding window sampling strategy.

    Each sample consists of a window of feature values of length `winlen`
    followed by a target value at the window endpoint. Samples are created
    from indices `winlen` to `n_samples - 1`.

    The dataset is randomly divided into training, test, and validation
    splits according to the specified proportions.

    Attributes
    ----------
    features : np.ndarray
        Feature array of shape (n_samples, n_features) or (n_samples,).
    target : np.ndarray
        Target array of shape (n_samples,) or (n_samples, n_targets).
    time : np.ndarray or None
        Optional time array of shape (n_samples,) associated with each sample.
    data_length : int
        Total number of samples in the dataset.
    winlen : int
        Length of the sliding window used to construct feature sequences.
    tr_ind : list[int]
        Indices corresponding to training samples.
    tst_ind : list[int]
        Indices corresponding to test samples.
    val_ind : list[int]
        Indices corresponding to validation samples.
    train : DataIterable
        Iterable view of the training samples.
    test : DataIterable
        Iterable view of the test samples.
    val : DataIterable
        Iterable view of the validation samples.
    """

    features: np.ndarray
    target: np.ndarray
    time: np.ndarray | None

    data_length: int

    tr_ind: list[int]
    tst_ind: list[int]
    val_ind: list[int]

    winlen: int

    def __init__(self, X: np.ndarray, y:np.ndarray, win_length: int, split: tuple[float, float, float] | list[float] | None, time: np.ndarray | None = None, seed: int | None = None):
        """
        Initialize the dataset split.

        Parameters
        ----------
        X : np.ndarray
            Feature array of shape (n_samples, n_features) or (n_samples,).
        y : np.ndarray
            Target array of shape (n_samples,) or (n_samples, n_targets).
        win_length : int
            Length of the sliding window used to construct input sequences.
        split : tuple[float, float, float] or None
            Tuple specifying the proportions of the dataset allocated to
            (train, test, validation). The values must sum to 1. If None,
            the default split (0.6, 0.2, 0.2) is used.
        time : np.ndarray or None, optional
            Optional time array associated with each sample.
        seed : int or None, optional
            Random seed for reproducible splitting.

        Raises
        ------
        ValueError
            If `win_length` exceeds the dataset length.
        ValueError
            If the split proportions do not sum to 1.
        ValueError
            If any split proportion is non-positive.
        ValueError
            If the split proportions are too small to allocate samples to
            the test or validation sets.
        """
        if win_length > X.shape[0]:
            raise ValueError("Length of window must be smaller than length of dataset")

        self.winlen = win_length
        self.features = X
        self.target = y
        
        self.time = time

        self.data_length = self.features.shape[0]

        if split is not None:
            if abs(sum(split) - 1.0) > 1e-6:
                raise ValueError("Split proportions must sum to 1")
            if split[0] <= 0 or split[1] <= 0 or split[2] <= 0:
                raise ValueError("Split proportions must be positive")
            if split[1] * (self.data_length - win_length) < 1 or split[2] * (self.data_length - win_length) < 1:
                raise ValueError("Split proportions too small, no data allocated to test or validation set")
        else:
            split  = (0.6, 0.2, 0.2)

        self.split(split, seed)

    def split(self, split: tuple[float, float, float] | list[float], seed: int | None = None):
        """
        Randomly split the dataset into training, test, and validation sets.

        Samples are defined by indices corresponding to the end of each
        sliding window.

        Parameters
        ----------
        split : tuple[float, float, float]
            Proportions for (train, test, validation).
        seed : int or None, optional
            Random seed used to control the sampling process.

        Notes
        -----
        The method creates three iterable dataset views:

        - `train`
        - `test`
        - `val`

        Each is a `DataIterable` instance referencing the parent dataset.
        """
        random.seed(seed)

        # data starts at end of first window
        datarange = list(range(self.winlen,self.data_length))

        # choose k random indices w/o replacement for training
        self.tr_ind = random.sample(datarange, k = int(len(datarange) * split[0]))
        # remaining indices are for testing & validation
        remaining = list(set(datarange) - set(self.tr_ind))

        # proportion of remaining data to be used for testing vs validation
        prop = split[1] / (split[1] + split[2]) 
        self.tst_ind = random.sample(remaining, k = int(len(remaining) * prop))
        self.val_ind = list(set(remaining) - set(self.tst_ind))

        self.train = DataIterable(self, self.tr_ind)
        self.test = DataIterable(self, self.tst_ind)
        self.val = DataIterable(self, self.val_ind)
            
    def shuffle_train_test(self, seed: int | None = None):
        """
        Shuffle the training and test samples while preserving their sizes.

        The combined set of training and test indices is reshuffled and
        repartitioned into training and test subsets with the same
        proportions as before. Validation samples remain unchanged.

        Parameters
        ----------
        seed : int or None, optional
            Random seed used for reproducible shuffling.
        """

        random.seed(seed)
        test_train = self.tr_ind + self.tst_ind

        self.tr_ind = random.sample(test_train, k = len(self.tr_ind))
        # remaining indices are for testing & validation
        self.tst_ind = list(set(test_train) - set(self.tr_ind))

        self.train = DataIterable(self, self.tr_ind)
        self.test = DataIterable(self, self.tst_ind)

class DataIterable:
    """
    Iterable view of a subset of samples from a `DataSplit` dataset.

    This class provides indexing and iteration over samples belonging
    to a particular split (training, test, or validation). Each sample
    consists of a feature window and its corresponding target value,
    and optionally a time value.

    Attributes
    ----------
    parent : DataSplit
        Reference to the parent dataset containing the full data arrays.
    indices : list[int]
        List of indices identifying which samples belong to this subset.
    """

    parent: DataSplit
    indices: list[int]

    def __init__(self, parent: "DataSplit", indices: list[int]):
        """
        Create an iterable dataset view.

        Parameters
        ----------
        parent : DataSplit
            Parent dataset object containing the full data arrays.
        indices : list[int]
            Indices corresponding to samples in this subset.
        """
        self.parent = parent
        self.indices = indices
    
    def __len__(self):
        """
        Return the number of samples in the subset.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.indices)
    
    def __getitem__(self, i):
        """
        Retrieve a single sample from the dataset.

        Parameters
        ----------
        i : int
            Index of the sample within the subset.

        Returns
        -------
        tuple
            If time data is available:
                (X_window, y_value, time_value)

            Otherwise:
                (X_window, y_value)

            where `X_window` is the feature sequence defined by the
            sliding window and `y_value` is the corresponding target.
        """
        ind = self.indices[i]

        X = self.parent.features
        y = self.parent.target
        win = self.parent.winlen
        
        if X.ndim == 1:
            X_i = X[ind-win:ind]
        else:
            X_i = X[ind-win:ind, :]
        
        if y.ndim == 1:
            y_i = y[ind]
        else:
            y_i = y[ind, :]
        
        if self.parent.time is not None:
            t_i = self.parent.time[ind]
            return X_i, y_i, t_i
        else:
            return X_i, y_i
    
    def __iter__ (self):
        """
        Iterate over all samples in the subset.

        Yields
        ------
        tuple
            Samples returned in the same format as `__getitem__`.
        """
        for i in range(len(self)):
            yield self[i]
    
    def generator(self):
        """
        Generator that yields dataset samples sequentially.

        This is primarily used for constructing TensorFlow datasets.

        Yields
        ------
        tuple
            Samples in the same format returned by `__getitem__`.
        """
        for i in range(len(self)):
            yield self[i]
    
    def as_dataset(self, batch_size=32, shuffle=False):
        """
        Convert the iterable dataset into a TensorFlow `tf.data.Dataset`.

        Parameters
        ----------
        batch_size : int, default=32
            Number of samples per batch.
        shuffle : bool, default=False
            Whether to shuffle the dataset before batching.

        Returns
        -------
        tf.data.Dataset
            TensorFlow dataset yielding batches of samples.

        Notes
        -----
        The dataset is created using `tf.data.Dataset.from_generator`
        and therefore streams samples from the Python generator.
        """

        import tensorflow as tf

        sample = self[0]

        output_signature = tuple(
            tf.TensorSpec(shape=s.shape, dtype=tf.as_dtype(s.dtype))
            for s in sample
        )

        ds = tf.data.Dataset.from_generator(
            self.generator,
            output_signature=output_signature
        )

        if shuffle:
            ds = ds.shuffle(len(self))

        return ds.batch(batch_size)
    
    @property
    def shape(self):
        return (len(self), self.parent.winlen)
    
    def to_numpy(self):
        """
        Convert the dataset subset into NumPy arrays. Only use if absolutely necessary, as this may consume a large amount of memory depending on the window size. Required for SKLearn compatibility.

        Returns
        -------
        tuple
            (X_array, y_array) where:
            - X_array is a NumPy array of shape (n_samples, winlen, n_features)
              containing the feature windows.
            - y_array is a NumPy array of shape (n_samples,) or (n_samples, n_targets)
              containing the target values.
        """
        X_list = []
        y_list = []

        for i in range(len(self)):
            X_i, y_i = self[i][:2]  # Ignore time if present
            X_list.append(X_i)
            y_list.append(y_i)

        X_array = np.array(X_list)
        y_array = np.array(y_list)

        return X_array, y_array