from pathlib import Path
import logging

import numpy as np
import pandas as pd
import anndata
import scanpy as sc
from scipy import sparse

from npt.datasets.base import BaseDataset
from npt.utils.data_loading_utils import download


logger = logging.getLogger(__name__)

AD_PATHS = {
    "tabula-muris-facs": "/group/singlecell/mouse/tabula_muris/anndata/tm_facs_log1p_cpm_counts_layer.h5ad",
    "tabula-muris-facs-small": "/group/singlecell/mouse/tabula_muris/anndata/tm_facs_log1p_cpm_counts_layer_small.h5ad",
    "tabula-muris-drop": "/group/singlecell/mouse/tabula_muris/anndata/tm_drop_log1p_cpm_counts_layer.h5ad",
    "tabula-muris-joint": "/group/singlecell/mouse/tabula_muris/anndata/tm_joint_log1p_cpm_counts_layer.h5ad",
    "tabula-muris-joint-small": "/group/singlecell/mouse/tabula_muris/anndata/tm_joint_log1p_cpm_counts_layer_small.h5ad",
}


class CellDataset(BaseDataset):
    def __init__(self, c):
        super(CellDataset, self).__init__(
            fixed_test_set_index=None,
        )
        self.c = c
        self.cell_dict = {}
        self.cell_dict["obs_feature_names"] = []

    def load(self):

        (
            self.data_table, 
            self.N, 
            self.D, 
            self.cat_features, 
            self.num_features,
            self.missing_matrix,
        ) = load_and_preprocess_cell_dataset(self.c, self.cell_dict)

        # For cell dataset, target index is the last column
        self.num_target_cols = []
        self.cat_target_cols = [self.D]

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ['wdbc.data']

        # overwrite missing values with a placeholder value of `0`
        if (p := self.c.exp_artificial_missing) > 0:
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            # TODO: maybe replace with np.nan
            self.data_table[self.missing_matrix] = 0


def load_and_preprocess_cell_dataset(c, cell_dict):
    """Load and preprocess a single cell dataset
    
    Parameters
    ----------
    c : AttrDict
        object with attributes for key:value pairs, e.g. a wandb config or AttrDict.
        data_path : str - main data path
        data_set : str - data set subdirectory in data_path
        data_name : str - data set filename in data_path/data_set
    other_dict : dict
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
    file = Path(cell_dict.get("h5ad_path", "adata.h5ad"))
    print(f"Attempting to load anndata from file:\n\t{file}")

    # Read AnnData dataset
    if not file.is_file() and (cell_dict.get("ebi_key", None) is not None):
        ebi_key = cell_dict.get("ebi_key")
        sc._settings.ScanpyConfig.datasetdir = path
        adata = sc.datasets.ebi_expression_atlas(ebi_key)
    elif not file.is_file():
        # raise an error
        raise FileNotFoundError(f"{file} does not exist.")
    else:
        adata = anndata.read_h5ad(str(file))
    print("\tanndata loaded.")

    if cell_dict.get("rename_dict", False):
        # rename some of the columns for easy access
        adata.obs.rename(columns=cell_dict.get("rename_dict", {}), inplace=True)

    if cell_dict.get("sparse", False):
        data_table = (
            adata.X if type(adata.X)==sparse.csr_matrix else sparse.csr_matrix(adata.X)
        )
    else:
        # data_table = adata.X if type(adata.X)==np.ndarray else adata.X.toarray()
        data_table = np.array(adata.obsm["X_pca"][:, :50])
    n_gene = data_table.shape[1]

    num_features_to_concat = []
    for k in cell_dict.get("obs_num_feature_names"):
        v = np.array(adata.obs_vector(k)).reshape(-1, 1)
        num_features_to_concat.append(v)
    num_feature_matrix = np.concatenate(num_features_to_concat, axis=1)

    cat_features_to_concat = []
    for k in cell_dict.get("obs_cat_feature_names"):
        raw_obs = adata.obs_vector(k)
        # convert categorical features to integer codes, lexographically sorted
        # this saves memory and allows us to sparsify the representation downstream
        v = pd.Categorical(raw_obs, categories=np.unique(raw_obs)).codes
        v = np.array(v).reshape(-1, 1)
        cat_features_to_concat.append(v)
    # [N, len(obs_feature_names)]
    cat_feature_matrix = np.concatenate(cat_features_to_concat, axis=1)

    if cell_dict.get("sparse", False):
        data_table = sparse.hstack(
            [
                data_table, 
                sparse.csr_matrix(num_feature_matrix), 
                sparse.csr_matrix(cat_feature_matrix),
            ],
        )
    else:
        data_table = np.concatenate(
            [data_table, num_feature_matrix, cat_feature_matrix], 
            axis=1,
        )

    N = data_table.shape[0]
    D = data_table.shape[1]

    missing_matrix = np.zeros((N, D))
    missing_matrix = missing_matrix.astype(dtype=np.bool_)

    cat_features = list(range(n_gene+num_feature_matrix.shape[1], D))
    num_features = list(range(0, n_gene+num_feature_matrix.shape[1]))
    return data_table, N, D, cat_features, num_features, missing_matrix


""" Specific Cell Datasets """

class TMCell(CellDataset):

    def __init__(self, c, **kwargs):
        """Load a Tabula Muris dataset
        
        Notes
        -----
        EBI Expression Atlas key: E-ENAD-15
        """
        super(TMCell, self).__init__(c=c, **kwargs)
        self.c = c
        self.ebi_key = "E-ENAD-15"
        # renaming dictionary for columns
        self.rename_dict = {"mouse.sex": "sex"}
        self.cell_dict = {}

        # categorical features to concatenate onto the data_table
        # added in the order listed, so the column order is preserved
        self.cell_dict["obs_cat_feature_names"] = [
            "tissue",
            "mouse.sex",
            "cell_ontology_class",
        ]
        self.cell_dict["obs_num_feature_names"] = [
            "n_counts",
        ]
        self.cell_dict["sparse"] = False
        self.cell_dict["data_name"] = f"{self.ebi_key}/{self.ebi_key}.h5ad"
        # self.cell_dict["ebi_key"] = self.ebi_key
        self.cell_dict["h5ad_path"] = AD_PATHS["tabula-muris-facs-small"]
        return

    def load(self,) -> None:
        """Load data_table and dimension indices into attributes"""
        (
            self.data_table, 
            self.N, 
            self.D, 
            self.cat_features, 
            self.num_features,
            self.missing_matrix,
        ) = load_and_preprocess_cell_dataset(self.c, self.cell_dict)
        logger.info(f"Extracted N {self.N}, D {self.D} data_table")
        # For TM dataset, target index is the last column for `cell_ontology_class`
        self.num_target_cols = []
        self.cat_target_cols = [-1]

        self.is_data_loaded = True
        self.tmp_file_or_dir_names = ["adata.h5ad"]

        # overwrite missing values with a placeholder value of `0`
        # c.exp_artifical_missing is a fraction of artifical missing data to inject
        # into the dataset. e.g. this induces dropout on the inputs.
        if (p := self.c.exp_artificial_missing) > 0:
            # make missing randomly masks values in all the non-target numeric columns
            self.missing_matrix = self.make_missing(p)
            # this is not strictly necessary with our code, but safeguards
            # against bugs
            self.data_table[self.missing_matrix] = 0
        return