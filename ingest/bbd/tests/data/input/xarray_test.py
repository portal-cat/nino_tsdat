# -*- coding: utf-8 -*- 
# Author: Nino

import pandas as pd
import xarray as xr

df = pd.read_csv('data_Jan_three_day.csv')

# print(df.columns)

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
