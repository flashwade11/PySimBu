from pathlib import Path

import anndata as ad
import hdf5plugin  # noqa: F401
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
            cell_type=dataset.obs["celltype"].values,
        ),
    )
    dataset.obs.set_index("cell_ID", inplace=True)
    dataset = filter_matrix(dataset, filter_genes=filter_genes, variance_cutoff=variance_cutoff)

    return dataset


def prepare_dataset(
    annotation_file: str | Path,
    count_file: str | Path,
    tpm_file: str | Path | None = None,
    transpose: bool = True,
) -> ad.AnnData:
    annotation = pd.read_csv(annotation_file, index_col=0)
    adata = ad.read_text(count_file, delimiter="\t").T if transpose else ad.read_text(count_file, delimiter="\t")
    adata.obs = annotation
    adata.uns["tpm_matrix"] = None
    if tpm_file:
        tpm_matrix = (
            ad.read_text(tpm_file, delimiter="\t").T.to_df()
            if transpose
            else ad.read_text(tpm_file, delimiter="\t").to_df()
        )
        adata.uns["tpm_matrix"] = tpm_matrix
    return adata


class SimBuDataset:
    def __init__(
        self,
        adata: ad.AnnData,
        name: str = "dataset",
        filter_genes: bool = True,
        variance_cutoff: float = 0.0,
        **kwargs,
    ):
        self.adata = adata
        self.name = name
        self.filter_genes = filter_genes
        self.variance_cutoff = variance_cutoff

        self.preprocess_dataset()

        self.cell_type = self.adata.obs["cell_type"].unique().tolist()

    def filter_matrix(
        self,
    ) -> ad.AnnData:
        genes = self.adata.var.index.values
        count_matrix = self.adata.to_df()
        if "tpm_matrix" not in self.adata.uns.keys() or self.adata.uns["tpm_matrix"] is None:
            tpm_matrix = None
        else:
            tpm_matrix = self.adata.uns["tpm_matrix"]

        if self.filter_genes:
            console.print("Filtering genes...", style="bold cyan")
            count_sum = count_matrix.sum(axis=0)
            count_var = count_matrix.var(axis=0)
            low_expressed_genes_1 = count_sum[count_sum == 0.0].index.values
            low_variance_genes_1 = count_var[count_var < self.variance_cutoff].index.values
            if tpm_matrix:
                tpm_sum = tpm_matrix.sum(axis=0)
                tpm_var = tpm_matrix.var(axis=0)
                low_expressed_genes_2 = tpm_sum[tpm_sum == 0.0].index.values
                low_variance_genes_2 = tpm_var[tpm_var < self.variance_cutoff].index.values
            else:
                low_expressed_genes_2 = genes
                low_variance_genes_2 = genes
            low_expressed_genes = np.intersect1d(low_expressed_genes_1, low_expressed_genes_2)
            low_variance_genes = np.intersect1d(low_variance_genes_1, low_variance_genes_2)

            genes_to_be_filtered = np.intersect1d(low_expressed_genes, low_variance_genes)
            genes_to_keep = genes[~np.isin(genes, genes_to_be_filtered)]
            console.print(
                f"{len(genes_to_be_filtered):,} genes were filterd out",
                style="bold red",
            )
        else:
            genes_to_keep = genes
        self.adata = self.adata[:, genes_to_keep]

    def preprocess_dataset(
        self,
    ):
        console.print("Preprocessing dataset...", style="bold cyan")
        n_cells = self.adata.shape[0]
        cells_old = self.adata.obs["cellname"].values
        new_ids = [f"{self.name}_{i}" for i in range(n_cells)]

        self.adata.obs = pd.DataFrame(
            dict(
                cell_ID=new_ids,
                cell_ID_old=cells_old,
                cell_type=self.adata.obs["celltype"].values,
            ),
        )
        self.adata.obs.set_index("cell_ID", inplace=True)
        dataset = filter_matrix(self.adata)

        return dataset

    @classmethod
    def load_from_h5ad(
        cls,
        adata_file: str | Path,
        name: str = "dataset",
        filter_genes: bool = True,
        variance_cutoff: float = 0.0,
    ):
        adata = ad.read_h5ad(adata_file)
        return cls(adata, name=name, filter_genes=filter_genes, variance_cutoff=variance_cutoff)

    @classmethod
    def load_from_csv(
        cls,
        annotation_file: str | Path,
        count_matrix_file: str | Path,
        tpm_matrix_file: str | Path | None = None,
        sep: str = "\t",
        transpose: bool = True,
        name: str = "dataset",
        filter_genes: bool = True,
        variance_cutoff: float = 0.0,
    ):
        anno = pd.read_csv(annotation_file, sep=sep, index_col=0)
        adata = (
            ad.read_text(count_matrix_file, delimiter=sep).T
            if transpose
            else ad.read_text(count_matrix_file, delimiter=sep)
        )
        adata.obs = anno
        adata.uns["tpm_matrix"] = None
        if tpm_matrix_file:
            tpm_matrix = (
                ad.read_text(tpm_matrix_file, delimiter=sep).T.to_df()
                if transpose
                else ad.read_text(tpm_matrix_file, delimiter=sep).to_df()
            )
            adata.uns["tpm_matrix"] = tpm_matrix

        return cls(adata, name=name, filter_genes=filter_genes, variance_cutoff=variance_cutoff)
