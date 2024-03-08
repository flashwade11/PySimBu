"""
Author: Wade Huang
Date: 2024-03-05 16:28:30
LastEditors: Wade Huang
LastEditTime: 2024-03-05 16:29:55
FilePath: /PySimBu/src/dataset.py
Description: 

"""
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from .utils import console


def filter_matrix(
    dataset: ad.AnnData,
    filter_genes: bool = True,
    variance_cutoff: float = 0.0,
) -> ad.AnnData:
    genes = dataset.var.index.values
    count_matrix = dataset.to_df()
    if "tpm_matrix" not in dataset.uns.keys() or dataset.uns["tpm_matrix"] is None:
        tpm_matrix = None
    else:
        tpm_matrix = dataset.uns["tpm_matrix"]
        
    if filter_genes:
        console.print("Filtering genes...", style="bold cyan")
        count_sum = count_matrix.sum(axis=0)
        count_var = count_matrix.var(axis=0)
        low_expressed_genes_1 = count_sum[count_sum == 0.0].index.values
        low_variance_genes_1 = count_var[count_var < variance_cutoff].index.values
        if tpm_matrix:
            tpm_sum = tpm_matrix.sum(axis=0)
            tpm_var = tpm_matrix.var(axis=0)
            low_expressed_genes_2 = tpm_sum[tpm_sum == 0.0].index.values
            low_variance_genes_2 = tpm_var[tpm_var < variance_cutoff].index.values
        else:
            low_expressed_genes_2 = genes
            low_variance_genes_2 = genes 
        low_expressed_genes = np.intersect1d(low_expressed_genes_1, low_expressed_genes_2)
        low_variance_genes = np.intersect1d(low_variance_genes_1, low_variance_genes_2)
        
        genes_to_be_filtered = np.intersect1d(low_expressed_genes, low_variance_genes) 
        genes_to_keep = genes[~np.isin(genes, genes_to_be_filtered)]
        console.print(f"{len(genes_to_be_filtered):,} genes were filterd out", style="bold red")

    dataset = dataset[:, genes_to_keep]
    return dataset
    
# def compare_matrix_with_annotation(
#     # matrix: pd.DataFrame,
#     # annotation: pd.DataFrame
#     dataset: ad.AnnData
# ) -> pd.DataFrame:
#     cells_matrix = dataset.obs.index.values
#     cells_annotation = annotation["cellname"].values
    
#     if not all(np.isin(cells_annotation, cells_matrix)):
#         raise ValueError("Cells in annotation and matrix do not match.")
    
#     if len(cells_matrix) != len(cells_annotation):
#         print("Cells in matrix and annotation do not match. Intersection of both will be used.")
#         cells_intersection = np.intersect1d(cells_matrix, cells_annotation)
#         matrix = matrix.loc[cells_intersection, :]
#         annotation = annotation.loc[annotation["ID"].isin(cells_intersection), :]
#         print(f"Remaining number of cells: {len(cells_intersection)}")
    
#     return matrix

def preprocess_dataset(
    dataset: ad.AnnData,
    name: str = "dataset",
    filter_genes: bool = True,
    variance_cutoff: float = 0.0,
) -> ad.AnnData:
    n_cells = dataset.shape[0]
    cells_old = dataset.obs["cellname"].values
    new_ids = [f"{name}_{i}" for i in range(n_cells)]
    
    dataset.obs = pd.DataFrame(
        dict(
            cell_ID=new_ids,
            cell_ID_old=cells_old,
            cell_type=dataset.obs["celltype"].values
        ),
    )
    dataset.obs.set_index("cell_ID", inplace=True)
    
    # matrices = filter_matrix(count_matrix, None, filter_genes=filter_genes, variance_cutoff=variance_cutoff)
    # count_matrix = matrices["count_matrix"]
    # tpm_matrix = matrices["tpm_matrix"]
    # count_matrix = compare_matrix_with_annotation(count_matrix, annotation)
    # count_matrix.index = new_ids
    # adata = ad.AnnData(
    #     count_matrix, 
    #     obs=obs, 
    #     uns=dict(
    #         tpm_matrix=tpm_matrix       
    #     )
    # )
    dataset = filter_matrix(dataset, filter_genes=filter_genes, variance_cutoff=variance_cutoff)
    
    return dataset

def prepare_dataset(
    annotation_file: str | Path,
    count_file: str | Path,
    tpm_file: str | Path | None = None,
    transpose: bool = True,
) -> ad.AnnData:
    annotation = pd.read_csv(annotation_file, index_col=0)
    adata = ad.read_text(count_file, delimiter="\t").T if transpose \
        else ad.read_text(count_file, delimiter="\t")
    adata.obs = annotation
    adata.uns["tpm_matrix"] = None
    if tpm_file:
        tpm_matrix = ad.read_text(tpm_file, delimiter="\t").T.to_df() if transpose \
            else ad.read_text(tpm_file, delimiter="\t").to_df()
        adata.uns["tpm_matrix"] = tpm_matrix
    return adata
    
