from itertools import chain
from pathlib import Path
from typing import Literal

import anndata as ad
import hdf5plugin
import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def simulate_sample(
    dataset: ad.AnnData,
    # scaling_vector,
    sample_vector: pd.Series,
    total_cells: int, 
    # total_read_counts: int,
    # remove_bias_in_counts: bool,
    # remove_bias_in_counts_method: str,
    # norm_counts,
    # seed: int = 42
) -> dict[str, pd.DataFrame]: 
    if not np.round(sample_vector.sum(), 5) == 1:
        raise ValueError(f"simulation_vector must sum up to 1 in each sample, but got {sample_vector.sum()}.")

    sampled_cells_ID = []
    for cell_type, proportion in sample_vector.items():
        cell_type_adata = dataset[dataset.obs.cell_type == cell_type]
        if proportion == 0.0:
            sampled_cell_type_ID = np.array([])
        else:
            cell_type_num = np.round(proportion * total_cells, decimals=0).astype(int)
            if cell_type_num == 0:
                cell_type_num = 1
            sampled_cell_type_ID = np.random.choice(
                cell_type_adata.obs.index,
                size=cell_type_num,
                replace=True
            )
        sampled_cells_ID.append(sampled_cell_type_ID)
    sampled_cells_ID = np.concatenate(sampled_cells_ID)
    sampled_cells_adata = dataset[sampled_cells_ID]
    
    sampled_cells_df = sampled_cells_adata.to_df()
    simulated_count_vector = sampled_cells_df.sum(axis=0).to_frame().T
    
    sampled_cell_type = dataset.obs.loc[sampled_cells_ID, "cell_type"].to_dict()
    simulated_cell_type_expression = (
        sampled_cells_df
        .reset_index()
        .replace({"cell_ID": sampled_cell_type})
        .rename(columns={"cell_ID": "cell_type"})
        .groupby("cell_type")
        .sum()
    )
    
    return dict(
        bulk=simulated_count_vector,
        cell_type_expression=simulated_cell_type_expression
    )


def simulate_prop(
    dataset: ad.AnnData,
    scenario: Literal["even", "random", "weighted", "custom"] = "random",
    n_samples: int = 100,
    weighted_cell_type: str | None = None,
    weighted_amount: float | None = None,
    custom_scenario_dataframe: pd.DataFrame | None = None,
    balance_even_mirror_scenario: float = 0.01,
) -> list[pd.DataFrame | str]:
    all_types = np.unique(dataset.obs.cell_type.values).tolist()
    n_cell_types = len(all_types)
    simulation_vector_list = []
    if scenario == "even":
        for _ in range(n_samples):
            vector = np.round(np.random.normal(loc=1.0 / n_cell_types, scale=balance_even_mirror_scenario, size=n_cell_types), 3)
            vector = vector / np.sum(vector)
            simulation_vector_list.append(vector)
        simulation_vector = pd.DataFrame(simulation_vector_list, columns=all_types)
        label = "even"
    elif scenario == "random":
        for _ in range(n_samples):
            vector = np.round(np.random.uniform(0, 1, size=n_cell_types), 3)
            vector = vector / np.sum(vector)
            simulation_vector_list.append(vector)
        simulation_vector = pd.DataFrame(simulation_vector_list, columns=all_types)
        label = "random"
    elif scenario == "weighted":
        if weighted_cell_type is None or weighted_amount is None:
            raise ValueError("weighted_cell_type and weighted_amount must be provided for weighted scenario")
        if weighted_amount > 0.99 or weighted_amount < 0:
            raise ValueError("weighted_amount must be between 0 and 0.99")
        if weighted_cell_type not in all_types:
            raise ValueError(f"weighted_cell_type must be one of {all_types}")
        random_cell_types = all_types.copy()
        random_cell_types.remove(weighted_cell_type)
        random_cell_types.insert(0, weighted_cell_type)
        for _ in range(n_samples):
            noise =  np.random.uniform(-0.01, 0.01, size=1)
            vector = np.round(np.random.uniform(0, 1, size=n_cell_types-1), 3)
            vector = (1 - weighted_amount - noise) * vector / np.sum(vector)
            vector = np.insert(vector, 0, weighted_amount + noise) 
            simulation_vector_list.append(vector)
        simulation_vector = pd.DataFrame(simulation_vector_list, columns=random_cell_types)
        label = f"weighted_{weighted_cell_type}_{weighted_amount}"
    elif scenario == "custom":
        if custom_scenario_dataframe is None:
            raise ValueError("custom_scenario_dataframe must be provided for custom scenario")
        if custom_scenario_dataframe.shape[0] != n_samples:
            raise ValueError("custom_scenario_dataframe must have the same number of rows as n_samples")
        if not all(custom_scenario_dataframe.columns.isin(all_types)):
            raise ValueError("Could not find all cell-types from scenario data in annotation.")
        simulation_vector = custom_scenario_dataframe
        # simulation_vector.index = range(n_samples)
        label = "custom"
    else:
        raise ValueError("Scenario must be either 'even', 'random', 'weighted', or 'custom'")  

    simulation_vector.index = simulation_vector.index.astype(str)
    return simulation_vector, label
    
def simulate_bulk(
    dataset: ad.AnnData,
    scenario: Literal["even", "random", "weighted", "custom"] = "random",
    n_samples: int = 100,
    n_cells: int = 1000,
    seed: int | None = None,
    weighted_cell_type: str | None = None,
    weighted_amount: float | None = None,
    custom_scenario_dataframe: pd.DataFrame | None = None,
    balance_even_mirror_scenario: float = 0.01,
    n_jobs: int = 16,
    verbose: int = 1
) -> ad.AnnData:
    simulation_vector, label = simulate_prop(
        dataset, scenario, n_samples, 
        weighted_cell_type, weighted_amount, 
        custom_scenario_dataframe, 
        balance_even_mirror_scenario
    )
        
    simulation_results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(simulate_sample)(
            dataset, sample_vector, total_cells=n_cells
        ) for _, sample_vector in simulation_vector.iterrows()
    )
    
    bulk_counts = pd.concat([res["bulk"] for res in simulation_results], ignore_index=True)
    bulk_counts.index = bulk_counts.index.astype(str)
    
    cell_type_expression = [res["cell_type_expression"] for res in simulation_results]
    
    simulation = ad.AnnData(
        X=bulk_counts, 
        obsm=dict(prop=simulation_vector), 
        uns=dict(expr=cell_type_expression),
    )
    
    simulation.obs = pd.DataFrame(
        dict(
            n_cells=n_cells,
            label=label
        ),
        index=simulation.obs.index
    )
    
    return simulation
    
def merge_simulation(
    simulation_list: ad.AnnData
) -> ad.AnnData:
    simulation = ad.concat(simulation_list)
    simulation.obs.index = pd.Index(range(simulation.shape[0]), dtype=str)
    simulation.uns["expr"] = list(chain.from_iterable([
        s.uns["expr"] for s in simulation_list
    ]))
    
    return simulation
    
def save_simulation(
    simulation: ad.AnnData,
    output_file: str | Path
):
    simulation.write_h5ad(
        filename=output_file,
        compression=hdf5plugin.FILTERS["zstd"]
    )