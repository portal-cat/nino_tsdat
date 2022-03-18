import tsdat
import xarray as xr
import pandas as pd

# TODO – Developer: Write your FileHandler and add documentation
class CustomFileHandler(tsdat.AbstractFileHandler):
    """--------------------------------------------------------------------------------
    Custom file handler for reading <some data type> files from a <instrument name>.

    See https://tsdat.readthedocs.io/en/latest/autoapi/tsdat/io/index.html for more
    examples of FileHandler implementations.
    --------------------------------------------------------------------------------"""

    def read(self, filename: str, **kwargs) -> xr.Dataset:
        """----------------------------------------------------------------------------
        Method to read data in a custom format and convert it into an xarray Dataset.

        Args:
            filename (str): The path to the file to read in.

        Returns:
            xr.Dataset: An xr.Dataset object
        ----------------------------------------------------------------------------"""

        df = pd.read_csv(filename)
        ds = df.to_xarray()

        ds = ds.assign_coords({'time': ds.DateTime})
        # ds = ds.assign_coords({'id': ds.Identifier})
        ds = ds.drop_vars('index')
        ds = ds.rename({'index': 'time'})
        id_index = ds.groupby('Identifier').groups

        new_xrarry = {}
        for key in id_index:
            new_xrarry[key] = ds.isel(time=id_index[key])
            new_xrarry[key] = new_xrarry[key].assign_coords({'id': key})
            new_xrarry[key] = new_xrarry[key].drop_vars(['DateTime', 'Identifier'])

        ds_concat_data = new_xrarry[list(new_xrarry.keys())[0]]

        for key in list(new_xrarry.keys())[1:]:
                ds_concat_data = xr.concat((ds_concat_data, new_xrarry[key]), dim='id')

        return ds_concat_data