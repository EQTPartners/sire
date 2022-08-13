"""
Copyright (C) eqtgroup.com Ltd 2022
https://github.com/EQTPartners/sire
License: MIT, https://github.com/EQTPartners/sire/LICENSE.md
"""


from datetime import datetime
from typing import Tuple

import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


class Dataset:
    """The abstraction Class of a Dataset."""

    def __init__(
        self,
        metric_name: str = "revenue",
        growth_metric_name: str = "yoy_growth",
        dataset_name: str = "./data/arr129.json",
    ) -> None:
        """Initialize a Dataset.

        Args:
            metric_name (str, optional): the name of the metric. Defaults to "revenue".
            growth_metric_name (str, optional): the name of the growth metric. Defaults to "yoy_growth".
            dataset_name (str, optional): the dataset location. Defaults to "./data/arr129.json".
        """
        self.metric_name = metric_name
        self.growth_metric_name = growth_metric_name
        self.next_growth_metric_name = "next_{}".format(growth_metric_name)
        self.data = pd.read_json(dataset_name, orient="records")
        self.data["sectors"] = self.data["sectors"].apply(set)
        self.data["customer_focus"] = self.data["customer_focus"].apply(set)
        self.data = self.data.reset_index(drop=True)

    def get_unique_org_ids(self) -> list:
        """Get the unique IDs for companies.

        Returns:
            list: a list of unique company IDs.
        """
        return list(self.data.id.unique())

    def check_org_exists(self, org_id: str) -> bool:
        """Check if a particular company ID exists in the dataset.

        Args:
            org_id (str): the ID of a company.

        Returns:
            bool: ID exists (True) or not (False).
        """
        return org_id in self.get_unique_org_ids()

    def cut_org_timeline(
        self, latest_known_dt: datetime, org_id: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split the time-series of a company into two parts using a latest_known_dt.

        Args:
            latest_known_dt (datetime): the latest date when the metric is known.
            org_id (str): the ID of the company to be splitted.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: the know and unknown time-series.
        """
        unknown = self.data[
            (self.data.id == org_id) & (self.data.date > latest_known_dt)
        ].sort_values(by="date", ascending=True)
        unknown = unknown[["date", self.metric_name]].reset_index(drop=True)
        known = self.data[
            (self.data.id == org_id) & (self.data.date <= latest_known_dt)
        ].sort_values(by="date", ascending=True)
        known = known[["date", self.metric_name]].reset_index(drop=True)
        return known, unknown

    def get_subset_by_sector_b2x(
        self, latest_known_dt: datetime, org_id: str, use_logical_and: bool = True
    ) -> pd.DataFrame:
        """Obtain a sub dataset (within the known period) that is compatible
            with a company from sector and/or customer focus aspects.

        Args:
            latest_known_dt (datetime): the latest calendar date when the metric is known.
            org_id (str): the ID of the company in scope.
            use_logical_and (bool, optional): the relation between customer focus and sector
                (logical AND/OR). Defaults to True.

        Returns:
            pd.DataFrame: sub dataset (in the known period) that is compatible with org_id.
        """
        result = self.data[self.data.date <= latest_known_dt]
        tmp_df = self.data[self.data.id == org_id]
        customer_focus = tmp_df.iloc[0]["customer_focus"]
        sectors = tmp_df.iloc[0]["sectors"]

        def filter_sector_cf(row: dict) -> bool:
            """Check if the data point is compatible from sector and/or customer focus aspects.

            Args:
                row (dict): the data point with sector and customer focus info.

            Returns:
                bool: compatible (True) or not (False).
            """
            has_common_b2x = len(row["customer_focus"].intersection(customer_focus))
            has_common_sector = len(row["sectors"].intersection(sectors)) > 0
            if use_logical_and:
                combined_condition = has_common_b2x and has_common_sector
            else:
                combined_condition = has_common_b2x or has_common_sector
            if combined_condition:
                return True
            else:
                return False

        result["is_selected"] = result.apply(filter_sector_cf, axis=1)
        result = (
            result[result.is_selected]
            .drop(["is_selected"], axis=1)
            .reset_index(drop=True)
        )
        return result
