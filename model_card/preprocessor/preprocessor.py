import copy

import dateparser
import pandas as pd


class Preprocessor:
    def __init__(self, column_types, n_bins=5):

        self.num = [
            i for i in column_types.keys() if column_types[i] == "number"
        ]  # x.columns[x.dtypes != "object"]
        self.cat_cols = None
        self.nan_cols = None
        self.date_cols = [
            i for i in column_types.keys() if column_types[i] == "string/date"
        ]
        self.date_cols_ts = [
            i for i in column_types.keys() if column_types[i] == "number/timestamp"
        ]
        self.date_cols_ts_ms = [
            i for i in column_types.keys() if column_types[i] == "number/timestamp_ms"
        ]

        self.bins = n_bins
        self.cat = [
            i for i in column_types.keys() if column_types[i] in ["number", "string"]
        ]

    def transform(self, x_in):
        x = copy.deepcopy(x_in)
        cat_cols = {}
        xproc = pd.DataFrame()

        num = [i for i in x_in.columns if i in self.num]
        for i in num:
            x[i] = pd.cut(
                x[i],
                bins=min(self.bins, x[i].nunique()),
                right=False,
            ).astype(str)

        cat = [i for i in x_in.columns if i in self.cat]
        for i in cat:
            xc = pd.get_dummies(x[i], prefix=i)
            if i + "_nan" in list(xc.columns):
                xc = xc.drop([i + "_nan"], axis=1)

            if i + "_Grouped_labels" in list(xc.columns):
                xc = xc.drop([i + "_Grouped_labels"], axis=1)

            cat_cols[i] = list(xc.columns)

            xproc = pd.concat([xproc, xc], axis=1)

        self.cat_cols = cat_cols

        return xproc


