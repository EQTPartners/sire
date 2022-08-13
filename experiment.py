"""
Copyright (C) eqtgroup.com Ltd 2022
https://github.com/EQTPartners/sire
License: MIT, https://github.com/EQTPartners/sire/LICENSE.md
"""


from datetime import datetime
import warnings

import altair as alt
import pandas as pd
from dateutil.relativedelta import *

from extrapolations import Extrapolations
from dataset import Dataset
from typing import Tuple, Union

# warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None  # default='warn'


class Experiment:
    """The abstraction class of an experiment for a company:
    i.e. extrapolations from all possible dates,
    from which metrics are calculated and visualizations are made."""

    def __init__(
        self,
        dataset: Dataset,
        org_id: str,
        extrapolate_len: int,
        n_trials: int,
        method: str,
        yoy_step: int = 12,
    ) -> None:
        """Initialize Experiment class.

        Args:
            dataset (Dataset): a Dataset instance.
            org_id (str): the ID of the company to be extrapolated.
            extrapolate_len (int): the number of data points to be extrapolated.
            n_trials (int): the number of trials to carry out.
            method (str): the method to sample metric multiply for the next period;
                the recommended value is "probability_matching".
            yoy_step (int, optional): the time steps (months) used for calculating growth metric.
                Defaults to 12.
        """
        self.dataset = dataset
        self.method = method
        self.org_id = org_id
        self.extrapolate_len = extrapolate_len
        self.n_trials = n_trials
        self.yoy_step = yoy_step
        self.reset()

    def reset(self) -> None:
        """Reset the whole experiment object."""
        self.run_dates = self.get_run_dates()
        self.extrapolations = []
        self.org_metric_df = None
        self.neg_likelihood_matrix = None

    def get_run_dates(self) -> list:
        """Obtain the dates that may be possible to run extrapolation.

        Returns:
            list: the list of dates.
        """
        _df = self.dataset.data
        return list(
            _df[_df.id == self.org_id].sort_values(by="date", ascending=True)["date"]
        )

    def has_run(self) -> bool:
        """Check if the experiment has been run.

        Returns:
            bool: has been run (True) or not (False).
        """
        return len(self.extrapolations) == len(self.run_dates)

    def is_empty(self) -> bool:
        """Check if all extrapolations are empty.

        Returns:
            bool: all empty (True) or not (False).
        """
        n_empty_extrapolations = 0
        for extrapolations in self.extrapolations:
            if extrapolations.is_empty():
                n_empty_extrapolations += 1
        return n_empty_extrapolations == len(self.extrapolations)

    def get_extrapolations_starts_fr(
        self, start_date: datetime
    ) -> Union[Extrapolations, None]:
        """Get an extrapolation that starts from a particular date.

        Args:
            start_date (datetime): a calendar date when the extrapolation shall starts from.

        Returns:
            Extrapolations: an extrapolation starting from start_date.
        """
        for entry in self.extrapolations:
            if entry.trials[0].first_dt == start_date:
                return entry
        return None

    def run(self, params: dict = {}) -> None:
        """Run this experiment.

        Args:
            params (dict, optional): additional parameters such as {"filter_type": "smooth"};
                Defaults to {}.
        """
        self.run_params = params
        if self.has_run():
            print("Reset Extrapolations for {} !".format(self.org_id))
            self.extrapolations = []
        for dt in self.run_dates:
            extrapolations = Extrapolations(
                self.dataset,
                self.org_id,
                dt,
                self.extrapolate_len,
                self.n_trials,
                self.method,
                yoy_step=self.yoy_step,
            )
            print(str(dt)[:7], end=" ", flush=True)
            extrapolations.run(params=params)
            self.extrapolations.append(extrapolations)
        print()
        if self.is_empty():
            warnings.warn(
                "No valid extrapolation was generated for company {}, please check.".format(
                    self.org_id
                )
            )

    def get_extrapolations_metric_mean_confidence_interval_df_fr(
        self, start_date: datetime, confidence: float = 0.95
    ) -> Union[pd.DataFrame, None]:
        """Obtain the stats (e.g. mean and confidence interval bounds) for an extrapolation
            starting from a particular date.

        Args:
            start_date (datetime): a calendar date when the extrapolation shall starts from.
            confidence (int, optional): the confidence interval. Defaults to 0.95.

        Returns:
            pd.DataFrame: the result DataFrame with mean, min, max, upper&lower CI bounds;
                if no valid extrapolation is available, return None.
        """
        extrapolation = self.get_extrapolations_starts_fr(start_date)
        if extrapolation is None:
            return None
        else:
            return extrapolation.get_extrapolations_metric_mean_confidence_interval(
                confidence
            )

    def get_org_metric_df(self) -> pd.DataFrame:
        """Get all time-series data for the company in scope.

        Returns:
            pd.DataFrame: the time-series data for the company in scope.
        """
        if self.org_metric_df is not None:
            return self.org_metric_df
        tmp_df = self.dataset.data
        metric_name = self.dataset.metric_name
        self.org_metric_df = tmp_df[(tmp_df.id == self.org_id)][["date", metric_name]]
        return self.org_metric_df

    def get_smooth_metric_mean_confidence_interval_df_fr(
        self, start_date: datetime, confidence: float = 0.95
    ) -> Union[pd.DataFrame, None]:
        """Obtain the smoothed know data points (e.g. mean and confidence interval bounds)
            from an extrapolation starting from a particular date.

        Args:
            start_date (datetime): a calendar date when the extrapolation shall starts from.
            confidence (int, optional): the confidence interval. Defaults to 0.95.

        Returns:
            pd.DataFrame: the result DataFrame with mean, min, max, upper&lower CI bounds;
                if no valid extrapolation is available, return None.
        """
        extrapolation = self.get_extrapolations_starts_fr(start_date)
        if extrapolation is None:
            return None
        else:
            return extrapolation.get_smooth_metric_mean_confidence_interval(confidence)

    def plot_extrapolations_start_from(
        self,
        start_date: datetime,
        confidence: float = 0.95,
        width: int = 750,
        height: int = 500,
    ) -> alt.LayerChart:
        """Plot the extrapolation from a particular date.

        Args:
            start_date (datetime): a calendar date when the extrapolation shall starts from.
            confidence (int, optional): the confidence interval. Defaults to 0.95.
            width (int, optional): the width of the chart. Defaults to 750.
            height (int, optional): the height of the chart. Defaults to 500.

        Returns:
            alt.LayerChart: the chart to be rendered.
        """
        gt_df = self.get_org_metric_df()
        mean_conf_band = self.get_extrapolations_metric_mean_confidence_interval_df_fr(
            start_date, confidence
        )
        if mean_conf_band is None or len(mean_conf_band) < 1:
            return None
        metric_name = self.dataset.metric_name
        # add a dummy row for know metric
        known_metric_dt = start_date + relativedelta(months=-1)
        known_metric_df = gt_df[gt_df["date"] == known_metric_dt]
        if len(known_metric_df) > 0:
            known_metric = known_metric_df.iloc[0][metric_name]
            mean_conf_band = mean_conf_band.append(
                {
                    "prediction_date": known_metric_dt,
                    "{}_mean".format(metric_name): known_metric,
                    "{}_{}_lower_bound".format(
                        metric_name, str(int(confidence * 100))
                    ): known_metric,
                    "{}_{}_upper_bound".format(
                        metric_name, str(int(confidence * 100))
                    ): known_metric,
                    "{}_min".format(metric_name): known_metric,
                    "{}_max".format(metric_name): known_metric,
                },
                ignore_index=True,
            )
        gt_line = (
            alt.Chart(gt_df)
            .mark_line()
            .encode(x="date", y="{}:Q".format(metric_name), color=alt.value("#00AAFF"))
        )
        mean_line = (
            alt.Chart(mean_conf_band)
            .mark_line(strokeDash=[5, 2])
            .encode(
                x="prediction_date",
                y="{}_mean:Q".format(metric_name),
                color=alt.value("#4b0082"),
            )
        )
        conf_band_area = (
            alt.Chart(mean_conf_band)
            .mark_area(opacity=0.3, color="green")
            .encode(
                x="prediction_date",
                y="{}_{}_lower_bound:Q".format(metric_name, str(int(confidence * 100))),
                y2="{}_{}_upper_bound:Q".format(
                    metric_name, str(int(confidence * 100))
                ),
            )
        )
        band_area = (
            alt.Chart(mean_conf_band)
            .mark_area(opacity=0.3, color="green")
            .encode(
                x="prediction_date",
                y="{}_min:Q".format(metric_name),
                y2="{}_max:Q".format(metric_name),
            )
        )
        smooth_mean_conf_band = self.get_smooth_metric_mean_confidence_interval_df_fr(
            start_date, confidence
        )
        smooth_band_area = (
            alt.Chart(smooth_mean_conf_band)
            .mark_area(opacity=0.3, color="black")
            .encode(
                x="date",
                y="{}_min:Q".format(metric_name),
                y2="{}_max:Q".format(metric_name),
            )
        )
        smooth_mean_line = (
            alt.Chart(smooth_mean_conf_band)
            .mark_line(strokeDash=[5, 2])
            .encode(
                x="date",
                y="{}_mean:Q".format(metric_name),
                color=alt.value("#373737"),
            )
        )
        return alt.layer(
            conf_band_area,
            band_area,
            smooth_band_area,
            smooth_mean_line,
            mean_line,
            gt_line,
        ).properties(
            width=width,
            height=height,
            title="Extrapolations for {} starting from {}".format(
                self.org_id, start_date
            ),
        )
