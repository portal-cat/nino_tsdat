import numpy as np
import pandas as pd
import xarray as xr

from sklearn.impute import KNNImputer
from sklearn.decomposition import NMF
from typing import Optional

from tsdat import DSUtil, QualityChecker, QualityHandler


class RemoveFailedValues(QualityHandler):
    """-------------------------------------------------------------------
    Replace all the failed values with _FillValue
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):

        # Add j loop in order to read each Identifier's data
        for j in range(results_array.shape[0]):

            if results_array[j].any():
                fill_value = DSUtil.get_fill_value(self.ds, variable_name)

                self.ds[variable_name][j] = self.ds[variable_name][j].where(
                    ~results_array[j], fill_value
                )


class ReplaceFailedValuesWithPrevious(QualityHandler):
    """-------------------------------------------------------------------
    Fill all the failed values with previous values
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):

        # Add j loop in order to read each Identifier's data
        for j in range(results_array.shape[0]):
            if results_array[j].any():
                keep_array = ~results_array[j]
                failed_indices = np.where(results_array[j])

                var_values = self.ds[variable_name][j].data
                num_indices_to_search = self.params.get("num_indices_to_search", 0)

                for index in failed_indices[0]:
                    for i in range(1, num_indices_to_search + 1):
                        if index - i >= 0 and keep_array[index - i]:
                            var_values[index] = var_values[index - i]
                            break


class ReplaceFailedValuesWithForwardFill(QualityHandler):
    """-------------------------------------------------------------------
    Forward Fill all the failed values
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):

        # Add j loop in order to read each Identifier's data
        for j in range(results_array.shape[0]):
            if results_array[j].any():
                da = self.ds[variable_name][j]
                da = da.ffill("time", limit=None)

                self.ds[variable_name][j][np.where(results_array[j])] = da[
                    np.where(results_array[j])
                ]


class ReplaceFailedValuesWithLinear(QualityHandler):
    """-------------------------------------------------------------------
    Linear Fill all the failed values
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):

        # Add j loop in order to read each Identifier's data
        for j in range(results_array.shape[0]):

            if results_array[j].any():
                da = self.ds[variable_name][j]
                da = da.interpolate_na(
                    dim="time",
                    method="linear",
                    fill_value=da.median(),
                    limit=None,
                    keep_attrs=True,
                )
                self.ds[variable_name][j][np.where(results_array[j])] = da[
                    np.where(results_array[j])
                ]


class ReplaceFailedValuesWithPolynomial(QualityHandler):
    """-------------------------------------------------------------------
    Polynomial Fill all the failed values with Order Type 2
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):

        # Add j loop in order to read each Identifier's data
        for j in range(results_array.shape[0]):
            if results_array[j].any():
                da = self.ds[variable_name][j]
                da = da.interpolate_na(
                    dim="time", method="polynomial", order=2, limit=None, keep_attrs=True
                )
                self.ds[variable_name][j][np.where(results_array[j])] = da[
                    np.where(results_array[j])
                ]


class ReplaceFailedValuesWithKNN(QualityHandler):
    """-------------------------------------------------------------------
    Sk-learn's K-Nearest Neighbors (KNN) Fill all dataset
    -------------------------------------------------------------------"""

    def run(self, variable_name: str, results_array: np.ndarray):

        # Create correlation matrix for each identifier dataset
        if not hasattr(self, "correlation_df"):
            self.correlation_df = {}
            for j in range(results_array.shape[0]):
                df = self.ds.isel(id=j).to_dataframe()
                self.correlation_df[j] = df.corr(method="spearman")

        # Add j loop in order to read each Identifier's data
        for j in range(results_array.shape[0]):

            if results_array[j].any():
                # Run KNN using correlated "features" (column names) that meet a correlation threshold
                df = self.ds.isel(id=j).to_dataframe().drop(columns='id')

                # Get columns correlated to the current variable
                idp = np.where(
                    self.correlation_df[j][variable_name] > self.params["correlation_thresh"]
                )[0]

                # Run correlated columns through KNN imputation
                if len(idp) > 1:
                    out = KNNImputer(n_neighbors=3).fit_transform(df.iloc[:, idp])

                    # Get index of current variable in correlation matrix
                    var_index = np.where(variable_name == self.correlation_df[j].index.values)[0][0]

                    # Get index of current variable in KNN output
                    out_index = np.where(var_index == idp)[0][0]
                    # Add output directly into dataset
                    # Only replace certain values
                    self.ds[variable_name][j].values[np.where(results_array[j])] = out[
                        :, out_index
                    ].squeeze()[np.where(results_array[j])]

                else:
                    # Current variable isn't correlated with anything
                    # Fills in all nans with median value
                    self.ds[variable_name][j] = self.ds[variable_name][j].fillna(
                        self.ds[variable_name][j].median()
                    )


# class ReplaceFailedValuesWithNMF(QualityHandler):
#     """-------------------------------------------------------------------
#     Sk-learn's Non-Negative Matrix Factorization (NMF) Fill all dataset
#     -------------------------------------------------------------------"""
#
#     def run(self, variable_name: str, results_array: np.ndarray):
#         if results_array.any():
#             # !!! Scikit learn version can't handle missing values
#             var_names = [i.name for i in self.ds.data_vars]
#             nmf_model = NMF(n_components=len(var_names), random_state=0, shuffle=False)
#
#             # NMF if the gap is larger than one day
#             W = nmf_model.fit_transform(self.ds.to_pandas())
#             H = nmf_model.components_
#             out = W.dot(H)
#
#             idx = np.where(variable_name in var_names)[0]
#             self.ds[variable_name].values[np.where(results_array)] = out[:, idx][
#                 np.where(results_array)
#             ]


class CheckGap(QualityChecker):
    def check_missing(self, variable_name: str) -> Optional[np.ndarray]:
        # If this is a time variable, we check for 'NaT'
        if self.ds[variable_name].data.dtype.type == np.datetime64:
            results_array = np.isnat(self.ds[variable_name].data)

        else:
            fill_value = DSUtil.get_fill_value(self.ds, variable_name)

            # If the variable has no _FillValue attribute, then
            # we select a default value to use

            # if fill_value is None:
            #     fill_value = -9999

            # Make sure fill value has same data type as the variable
            fill_value = np.array(
                fill_value, dtype=self.ds[variable_name].data.dtype.type
            )

            # First check if any values are assigned to _FillValue
            results_array = np.equal(self.ds[variable_name].data, fill_value)

            # Then, if the value is numeric, we should also check if any values are assigned to NaN
            if self.ds[variable_name].data.dtype.type in (
                type(0.0),
                np.float16,
                np.float32,
                np.float64,
            ):
                results_array |= np.isnan(self.ds[variable_name].data)

        return results_array

    def run(self, variable_name: str) -> Optional[np.ndarray]:
        """-------------------------------------------------------------------
        Check the rows with minimum time gap
        -------------------------------------------------------------------"""
        # First return boolean array of missing data
        # nan values are true, non-nan values are false

        missing = self.check_missing(variable_name)

        fs = 1  # 1/min, sampling time interval

        min_time_gap = self.params.get("min_time_gap", 0)
        max_time_gap = self.params.get("max_time_gap", 0)

        self.ds[variable_name] = self.ds[variable_name].where(~missing)

        results_array_concat = np.zeros(missing.shape).astype(bool)

        # Identify consecutive nans and sum them together
        # Non-nan elements are set to zero
        # The size of each group of nans is stored at the first nan's index
        # https://stackoverflow.com/questions/29007830/identifying-consecutive-nans-with-pandas

        if not missing.any():
            return missing

        # Add j loop in order to read each Identifier's data
        for j in range(missing.shape[0]):

            results_array = np.zeros(self.ds[variable_name][j].size).astype(bool)

            if sum(missing[j]):
                gap_size = (
                    self.ds[variable_name][j]
                    .isnull()
                    .astype(int)
                    .groupby(self.ds[variable_name][j].notnull().astype(int).cumsum())
                    .sum()
                    .values
                )
                gap_size = np.pad(
                    gap_size, [(1, 0)], mode="constant"
                )  # pad 1 zero in front b/c above algorithm skips 1 index place
            else:
                results_array_concat[j] = results_array
                continue

            # Get the number of gaps and store their size in total_gaps
            gap_index = np.nonzero(gap_size)[0]
            total_gaps = gap_size[gap_index]

            # Check to see if there are any gaps that lie between min_time_gap and max_time_gap
            # These "if"s only work properly if run in cumulative (0 -> 60 -> 1440 -> inf) order
            if total_gaps.size == 0:
                # if no missing data, return all False
                results_array_concat[j] = results_array
                continue

            if total_gaps[0] == self.ds[variable_name][j].size:
                # if all data is missing, return all False
                results_array_concat[j] = results_array
                continue

            # if any(total_gaps > min_time_gap * fs) and not any(
            #     total_gaps <= max_time_gap * fs
            # ):
            #     # if gaps of missing data aren't between min and max, return all False
            #     results_array_concat[j] = results_array
            #     continue

            # Otherwise, if there are gaps, find what index they are in
            # Find indices of each gap (nans) and store in dictionary
            # (dictionary key is the gap size (g))
            # (dictionary items are the indices of ds[variable] containing nans)
            gap_indices = {}
            for i, g in enumerate(total_gaps):
                gap_indices[i] = np.arange(gap_index[i], gap_index[i] + g) + np.sum(
                    total_gaps[:i]
                )

            # Check to see how large each gap is
            gap_idx_to_fix = []
            gap_to_fix = []
            for i, x in enumerate(total_gaps):
                if x > min_time_gap and x <= max_time_gap:
                    gap_idx_to_fix.append(i)
                    gap_to_fix.append(x)

            if gap_idx_to_fix:
                print(
                    f"Biggest gap time --> {max(gap_to_fix)*fs} minutes, [min: {min_time_gap}, max: {max_time_gap}], Number of missing gaps: {len(gap_to_fix)} --> {variable_name} {self.ds[variable_name][j].id.values}"
                )
            # Create results_array
            for i in gap_idx_to_fix:
                results_array[gap_indices[i]] = True

            results_array_concat[j] = results_array
        return results_array_concat
