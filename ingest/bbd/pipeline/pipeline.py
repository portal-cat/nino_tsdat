import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from utils import IngestPipeline, format_time_xticks
from tsdat import DSUtil
from typing import Dict


class Pipeline(IngestPipeline):
    """--------------------------------------------------------------------------------
    EXAMPLE INGEST INGESTION PIPELINE

    This is an example ingest meant to demonstrate how one might set up an ingestion
    pipeline using this template repository. It should be deleted before this
    repository or any of its ingests are used in a production environment.

    --------------------------------------------------------------------------------"""

    def hook_customize_raw_datasets(
        self, raw_dataset_mapping: Dict[str, xr.Dataset]
    ) -> Dict[str, xr.Dataset]:
        return raw_dataset_mapping

    def hook_customize_dataset(
        self, dataset: xr.Dataset, raw_mapping: Dict[str, xr.Dataset]
    ) -> xr.Dataset:
        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        return dataset

    def hook_generate_and_persist_plots(self, dataset: xr.Dataset) -> None:
        """-------------------------------------------------------------------
        Hook to allow users to create plots from the xarray dataset after
        processing and QC have been applied and just before the dataset is
        saved to disk.
        To save on filesystem space (which is limited when running on the
        cloud via a lambda function), this method should only
        write one plot to local storage at a time. An example of how this
        could be done is below:
        ```
        filename = DSUtil.get_plot_filename(dataset, "sea_level", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(dataset["time"].data, dataset["sea_level"].data)
            fig.save(tmp_path)
            storage.save(tmp_path)
        filename = DSUtil.get_plot_filename(dataset, "qc_sea_level", "png")
        with self.storage._tmp.get_temp_filepath(filename) as tmp_path:
            fig, ax = plt.subplots(figsize=(10,5))
            DSUtil.plot_qc(dataset, "sea_level", tmp_path)
            storage.save(tmp_path)
        ```
        Args:
        ---
            dataset (xr.Dataset):   The xarray dataset with customizations and
                                    QC applied.
        -------------------------------------------------------------------"""

        def format_time_xticks(ax, start=4, stop=21, step=4, date_format="%H-%M"):
            # ax.xaxis.set_major_locator(mpl.dates.HourLocator(byhour=range(start, stop, step)))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter(date_format))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

        ds = dataset

        for var in ds.data_vars:
            if "qc" not in var:
                nm = var.replace("\\", "/").rsplit("/")[-1]
                filename = DSUtil.get_plot_filename(ds, nm, "png")
                with self.storage._tmp.get_temp_filepath(filename) as tmp_path:

                    # Create the figure and axes objects
                    fig, ax = plt.subplots(
                        nrows=1, ncols=1, figsize=(14, 8), constrained_layout=True
                    )
                    fig.suptitle(nm)

                    ds[var][0].plot(label="Raw data")
                    ds[var][0].where(ds["qc_" + var][0]).plot(label="QC inserted")
                    plt.legend()
                    # Set the labels and ticks
                    # format_time_xticks(ax)
                    ax.set_title("")  # Remove title created by xarray
                    ax.set_xlabel("Time (UTC)")
                    # ax.set_ylabel(r"Wind Speed (ms$^{-1}$)")

                    # Save the figure
                    fig.savefig(tmp_path, dpi=100)
                    self.storage.save(tmp_path)
                    plt.close()

        return
