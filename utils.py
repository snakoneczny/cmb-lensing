import logging
from os import path

DATA_DIR_RELATIVE = '../../data/lensing_takahashi_17'
DATA_DIR_CIS = '/mnt/home/snakoneczny/data/lensing_takahashi_17'
DATA_DIR_MOUNT = '/mnt/cis/data/lensing_takahashi_17'

logformat = '%(asctime)s %(levelname)s: %(message)s'
datefmt = '%d/%m/%Y %H:%M:%S'
logging.basicConfig(format=logformat, datefmt=datefmt, level=logging.INFO)
logger = logging.getLogger(__name__)


def save_predictions(predictions_df, args, timestamp):
    file_name = timestamp
    if args.tag:
        file_name = '{}_{}'.format(args.tag, file_name)
    if args.is_test:
        file_name = 'TEST_' + file_name
    predictions_path = 'predictions/{file_name}.csv'.format(file_name=file_name)
    predictions_df.to_csv(predictions_path, index=False)
    print('Predictions saved to: {}'.format(predictions_path))


def get_tensorboard_dir(args, timestamp, learning_rate, batch_size):
    log_folder = 'lr={lr}, bs={bs}, {ts}'.format(ts=timestamp.replace('_', ' '), tag=args.tag, lr=learning_rate,
                                                 bs=batch_size)
    if args.tag:
        log_folder = '{}, {}'.format(args.tag, log_folder)
    if args.is_test:
        log_folder = 'TEST_' + log_folder
    log_dir = path.join('tensorboard', log_folder)
    return log_dir


def safe_indexing(X, indices):
    """Return items or rows from X using indices.
    Allows simple indexing of lists or arrays.
    Parameters
    ----------
    X : array-like, sparse-matrix, list, pandas.DataFrame, pandas.Series.
        Data from which to sample rows or items.
    indices : array-like of int
        Indices according to which X will be subsampled.
    Returns
    -------
    subset
        Subset of X on first axis
    Notes
    -----
    CSR, CSC, and LIL sparse matrices are supported. COO sparse matrices are
    not supported.
    """
    if hasattr(X, 'iloc'):
        # Work-around for indexing with read-only indices in pandas
        indices = indices if indices.flags.writeable else indices.copy()
        # Pandas Dataframes and Series
        try:
            return X.iloc[indices]
        except ValueError:
            # Cython typed memoryviews internally used in pandas do not support
            # readonly buffers.
            # TODO: that was commented
            # warnings.warn("Copying input dataframe for slicing.",
            #               DataConversionWarning)
            return X.copy().iloc[indices]
    elif hasattr(X, 'shape'):
        if hasattr(X, 'take') and (hasattr(indices, 'dtype') and
                                   indices.dtype.kind == 'i'):
            # This is often substantially faster than X[indices]
            return X.take(indices, axis=0)
        else:
            return X[indices]
    else:
        return [X[idx] for idx in indices]
