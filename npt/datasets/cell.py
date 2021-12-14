from pathlib import Path

import numpy as np
import pandas as pd
import anndata

from npt.datasets.base import BaseDataset
from npt.utils.data_loading_utils import download


class CellDataset(BaseDataset):
    def __init__(self, c):
        super(CellDataset, self).__init__(
            fixed_test_set_index=None,
        )
        self.c = c

    def load(self):
        (
            self.data_table, 
            self.N, 
            self.D, 
            self.cat_features, 
            self.num_features,
            self.missing_matrix,
        ) = load_and_preprocess_cell_dataset(self.c)

        # For cell dataset, target index is the last column
        self.num_target_cols = []
        self.cat_target_cols = [-1]

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['wdbc.data']

        # overwrite missing values with a placeholder value of `0`
        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            self.data_table[self.missing_matrix] = 0


def load_and_preprocess_cell_dataset(c):
    """Load and preprocess a single cell dataset
    
    Parameters
    ----------
    c : AttrDict
        object with attributes for key:value pairs, e.g. a wandb config or AttrDict.
        data_path : str - main data path
        data_set : str - data set subdirectory in data_path
        data_name : str - data set filename in data_path/data_set
        obs_feature_names : list - str names of features in `adata.obs` to include as
        categorical features in the data_table.

    Returns
    -------
    data_table : np.ndarray
        [N, D] array of features
    N : int
        number of observations
    D : int
        number of feature dimensions
    cat_features : list
        [int,] list of categorical feature indices
    num_features : list
        [int,] list of numerical features
    missing_matrix : np.ndarray
        [N, D] boolean matrix of missing values where `True` indicates the value
        is missing.

    Notes
    -----
    Here, loads a single cell dataset from an AnnData object, sets all values in `.X`
    as features in the datatable, then adds any categorical features from `obs` that
    are specified in `c.obs_feature_names`.
    """
    path = Path(c.data_path) / c.data_set
    data_name = c.data_name

    file = path / data_name

    if not file.is_file():
        # raise an error
        raise FileNotFoundError(f"{file} does not exist.")

    # Read AnnData dataset
    # data_table = pd.read_csv(file, header=None).to_numpy()
    adata = anndata.read_h5ad(str(file))
    data_table = adata.X if type(adata.X)==np.ndarray else adata.X.toarray()
    n_gene = data_table.shape[1]

    cat_features_to_concat = []
    for k in c.obs_feature_names:
        cat_features_to_concat.append(adata.obs_vector(k))
    cat_feature_matrix = np.hstack(cat_features_to_concat) # [N, len(obs_feature_names)]

    data_table = np.concatenate([data_table, cat_feature_matrix], axis=1)

    N = data_table.shape[0]
    D = data_table.shape[1]

    missing_matrix = np.zeros((N, D))
    missing_matrix = missing_matrix.astype(dtype=np.bool_)

    cat_features = list(range(n_gene, D))
    num_features = list(range(0, n_gene))
    return data_table, N, D, cat_features, num_features, missing_matrix

