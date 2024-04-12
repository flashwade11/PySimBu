from pathlib import Path

import anndata as ad
import hdf5plugin  # noqa: F401
import numpy as np
import pandas as pd

from .utils import console


class SimBuDataset:
    def __init__(
        self,
        adata: ad.AnnData,
        cellname_prefix: str = "cell",
        filter_genes: bool = True,
        variance_cutoff: float = 0.0,
        *args,
        **kwargs,
    ):
        self.adata = adata
        self.preprocess(cellname_prefix, filter_genes, variance_cutoff)
        self.cell_type = self.adata.obs["cell_type"].unique().tolist()

    def process_genes(
        self,
        variance_cutoff: float = 0.0,
    ) -> ad.AnnData:
        console.print("Filtering genes...", style="bold cyan")

        genes = self.adata.var.index.values
        count_matrix = self.adata.to_df()
        if "tpm_matrix" not in self.adata.uns.keys() or self.adata.uns["tpm_matrix"] is None:
            tpm_matrix = None
        else:
            tpm_matrix = self.adata.uns["tpm_matrix"]

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
        console.print(
            f"{len(genes_to_be_filtered):,} genes were filterd out",
            style="bold red",
        )
        self.adata = self.adata[:, genes_to_keep]

    def preprocess(
        self,
        cellname_prefix: str = "dataset",
        filter_genes: bool = True,
        variance_cutoff: float = 0.0,
    ):
        self.process_obs(cellname_prefix=cellname_prefix)
        if filter_genes:
            self.process_genes(variance_cutoff=variance_cutoff)

    def process_obs(self, cellname_prefix: str = "cell"):
        console.print("processing obs...", style="bold cyan")
        n_cells = self.adata.shape[0]
        cells_old = self.adata.obs["cellname"].values
        new_ids = [f"{cellname_prefix}_{i}" for i in range(n_cells)]

        self.adata.obs = pd.DataFrame(
            dict(
                cell_ID=new_ids,
                cell_ID_old=cells_old,
                cell_type=self.adata.obs["celltype"].values,
            ),
        )
        self.adata.obs.set_index("cell_ID", inplace=True)

    @classmethod
    def load_from_h5ad(
        cls,
        adata_file: str | Path,
        cellname_prefix: str = "cell",
        filter_genes: bool = True,
        variance_cutoff: float = 0.0,
    ):
        console.print("loading adata from h5ad file...", style="bold cyan")
        adata = ad.read_h5ad(adata_file)
        return cls(adata, cellname_prefix, filter_genes, variance_cutoff)

    @classmethod
    def load_from_csv(
        cls,
        annotation_file: str | Path,
        count_matrix_file: str | Path,
        tpm_matrix_file: str | Path | None = None,
        sep: str = "\t",
        transpose: bool = True,
        cellname_prefix: str = "cell",
        filter_genes: bool = True,
        variance_cutoff: float = 0.0,
    ):
        console.print("loading adata from csv file...", style="bold cyan")
        console.print("loading annotation...", style="bold cyan")
        anno = pd.read_csv(annotation_file, sep=sep, index_col=0)
        console.print("loading count matrix...", style="bold cyan")
        adata = (
            ad.read_text(count_matrix_file, delimiter=sep).T
            if transpose
            else ad.read_text(count_matrix_file, delimiter=sep)
        )
        adata.obs = anno
        adata.uns["tpm_matrix"] = None
        if tpm_matrix_file:
            console.print("loading tpm matrix...", style="bold cyan")
            tpm_matrix = (
                ad.read_text(tpm_matrix_file, delimiter=sep).T.to_df()
                if transpose
                else ad.read_text(tpm_matrix_file, delimiter=sep).to_df()
            )
            adata.uns["tpm_matrix"] = tpm_matrix

        return cls(adata, cellname_prefix, filter_genes, variance_cutoff)
