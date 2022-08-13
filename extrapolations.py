"""
Copyright (C) eqtgroup.com Ltd 2022
https://github.com/EQTPartners/sire
License: MIT, https://github.com/EQTPartners/sire/LICENSE.md
"""


import math
import random
import warnings
from datetime import datetime
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy
from dateutil.relativedelta import *
from pykalman import KalmanFilter
from sklearn.neighbors import KernelDensity

from dataset import Dataset
from trial import Trial

# warnings.filterwarnings("ignore")

pd.options.mode.chained_assignment = None  # default='warn'


class Extrapolations:
    """The Class representing extrapolations.

    Raises:
        NotImplementedError: Illegal sample_x_method.
        NotImplementedError: Illegal filter type.
    """

    quantiles = [0.25, 0.5, 0.75, 0.9]
    em_vars = [
        # "transition_matrices",
        # "observation_matrices",
        "transition_offsets",
        "observation_offsets",
        "transition_covariance",
        "observation_covariance",
        "initial_state_mean",
        "initial_state_covariance",
    ]
    sample_x_methods = (
        "probability_matching",
        "bootstrapping",
    )

    def __init__(
        self,
        dataset: Dataset,
        org_id: str,
        latest_known_dt: datetime,
        extrapolate_len: int,
        n_trials: int,
        method: str,
        full_trajectory: bool = True,
        max_attempts: int = 500,
        yoy_step: int = 12,
    ) -> None:
        """Initialize Extrapolations class.

        Args:
            dataset (Dataset): a Dataset instance.
            org_id (str): the ID of the company to be extrapolated.
            latest_known_dt (datetime): the latest calendar date when the metrics are known.
            extrapolate_len (int): the number of data points to be extrapolated.
            n_trials (int): the number of trials to carry out.
            method (str): the method to sample metric multiply for the next period;
                the recommended value is "probability_matching".
            full_trajectory (bool, optional): require the extrapolated trajectories to be
                exactly extrapolate_len?. Defaults to True.
            max_attempts (int, optional): maximum number of retry to obtain full extrapolations.
                Defaults to 500.
            yoy_step (int, optional): the time steps (months) used for calculating growth metric.
                Defaults to 12.
        """
        self.dataset = dataset
        self.method = method
        self.org_id = org_id
        self.extrapolate_len = extrapolate_len
        self.yoy_step = yoy_step
        self.data = dataset.get_subset_by_sector_b2x(latest_known_dt, org_id)
        if len(self.data[self.data.id != org_id]) < 10:
            warnings.warn("use logical OR when filtering on sector and customer focus!")
            self.data = dataset.get_subset_by_sector_b2x(
                latest_known_dt, org_id, use_logical_and=False
            )
        self.org_known, self.org_unknown = dataset.cut_org_timeline(
            latest_known_dt, org_id
        )
        self.org_unknown_dates = list(self.org_unknown["date"])
        if len(self.org_known) < 1:
            warnings.warn("latest_known_date={} is too early!".format(latest_known_dt))
            self.latest_known_dt = latest_known_dt
        else:
            self.latest_known_dt = self.org_known.iloc[-1].date
        self.trials = [Trial(i, self.latest_known_dt, method) for i in range(n_trials)]
        self.pred_metric_df = None
        self.smooth_metric_df = None
        self.neg_likelihood = None
        self.full_trajectory = full_trajectory
        self.n_full_trial = 0
        self.n_total_trial = 0
        self.max_attempts = max_attempts

    @staticmethod
    def mean_confidence_interval(
        data: list, confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """Calculate mean and confidence interval (CI) over a number list.

        Args:
            data (list): the input list of numbers.
            confidence (float, optional): the confidence. Defaults to 0.95.

        Returns:
            Tuple[float, float, float]: mean, lower CI, upper bound CI.
        """
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return m, m - h, m + h

    @staticmethod
    def validate_signal(signal: list, consecutive: int = 3) -> int:
        """Validate if a time-series can be extrapolated.

        Args:
            signal (list): the time-series.
            consecutive (int, optional): minimum number of consecutive data points required.
                Defaults to 3.

        Returns:
            int: the earliest starting index if passing, otherwise -1.
        """
        n_consecutive_exist = 0
        for i in range(len(signal)):
            if not np.ma.is_masked(signal[i]):
                n_consecutive_exist += 1
            else:
                n_consecutive_exist = 0
            if n_consecutive_exist >= 3:
                break
        if n_consecutive_exist >= 3:
            return i - consecutive + 1
        else:
            return -1

    def get_pred_metric_df(self) -> pd.DataFrame:
        """Obtain the the entire prediction result.

        Returns:
            pd.DataFrame: the result DataFrame.
        """
        if self.pred_metric_df is not None:
            return self.pred_metric_df
        metric_name = self.dataset.metric_name
        result = []
        for trial in self.trials:
            for i in range(len(trial.pred_metric)):
                val = trial.pred_metric[i]
                dt = trial.first_dt + relativedelta(months=+i)
                trial_id = trial.id
                result.append(
                    {
                        "method": trial.method,
                        "trial_id": trial_id,
                        "date": dt,
                        metric_name: val,
                    }
                )
        self.pred_metric_df = pd.DataFrame(
            result, columns=["method", "trial_id", "date", metric_name]
        )
        return self.pred_metric_df

    def get_smooth_metric_df(self) -> pd.DataFrame:
        """Obtain the smooth version of known historical data points.

        Returns:
            pd.DataFrame: the result DataFrame.
        """
        if self.smooth_metric_df is not None:
            return self.smooth_metric_df
        metric_name = self.dataset.metric_name
        result = []
        for trial in self.trials:
            n_smooth_metrics = len(trial.smooth_metric)
            for i in range(n_smooth_metrics):
                val = trial.smooth_metric[n_smooth_metrics - i - 1]
                dt = trial.first_dt - relativedelta(months=+(i + 1))
                trial_id = trial.id
                result.append(
                    {
                        "method": trial.method,
                        "trial_id": trial_id,
                        "date": dt,
                        metric_name: val,
                    }
                )
        self.smooth_metric_df = pd.DataFrame(
            result, columns=["method", "trial_id", "date", metric_name]
        )
        return self.smooth_metric_df

    def calculate_squared_error(self) -> pd.DataFrame:
        """Calculate squared error for each predicted point.

        Returns:
            pd.DataFrame: the DataFrame that has a new squared_error column.
        """
        metric_name = self.dataset.metric_name
        result = self.get_extrapolations_metric_mean_confidence_interval()

        def calc_squared_error(row: dict) -> float:
            """Calculate squared error for one predicted point.

            Args:
                row (dict): the predicted point contains many stats such as mean.

            Returns:
                float: the calculated squared error.
            """
            base_dt = self.latest_known_dt
            dt = row["prediction_date"]
            months_diff = ((dt.year - base_dt.year) * 12) + dt.month - base_dt.month
            gt_df = self.org_unknown[self.org_unknown.date == dt]
            if len(gt_df) < 1:
                return None
            gt = gt_df.iloc[0][metric_name]
            pred_mean = row["{}_mean".format(metric_name)]
            return ((pred_mean - gt) / months_diff) ** 2

        if len(result) > 0:
            result["{}_se".format(metric_name)] = result.apply(
                calc_squared_error, axis=1
            )
        return result

    def calculate_rmse(self) -> float:
        """Calculate the RMSE evaluation metric for this extrapolation.

        Returns:
            float: the calculated RMSE value.
        """
        _df = self.calculate_squared_error()
        _df = _df[_df.revenue_se > 0]
        return np.sqrt(np.mean(_df.revenue_se.to_list()))

    def calculate_mape(self) -> float:
        """Calculate the MAPE evaluation metric for this extrapolation.

        Returns:
            float: the calculated MAPE value.
        """
        metric_name = self.dataset.metric_name
        ape_col_name = "{}_ape".format(metric_name)
        result = self.get_extrapolations_metric_mean_confidence_interval()

        def calc_absolute_percentage_error(row: dict) -> float:
            """Calculate absolute percentage error (APE) for one predicted point.

            Args:
                row (dict): the predicted point contains many stats such as mean.

            Returns:
                float: the calculated APE.
            """
            base_dt = self.latest_known_dt
            dt = row["prediction_date"]
            months_diff = ((dt.year - base_dt.year) * 12) + dt.month - base_dt.month
            gt_df = self.org_unknown[self.org_unknown.date == dt]
            if len(gt_df) < 1:
                return None
            gt = gt_df.iloc[0][metric_name]
            pred_mean = row["{}_mean".format(metric_name)]
            return (abs(pred_mean - gt) / gt) / months_diff

        if len(result) > 0:
            result[ape_col_name] = result.apply(calc_absolute_percentage_error, axis=1)

        _df = result[result[ape_col_name] > 0]
        return np.mean(_df[ape_col_name].to_list())

    def calculate_pcc(self) -> float:
        """Calculate the PCC evaluation metric for this extrapolation.

        Returns:
            float: the calculated PCC value.
        """
        metric_name = self.dataset.metric_name
        gt_col_name = "{}_gt".format(metric_name)
        result = self.get_extrapolations_metric_mean_confidence_interval()

        def get_gt(row: dict) -> float:
            """Get the ground truth data point.

            Args:
                row (dict): the predicted point contains many stats such as mean.

            Returns:
                float: the ground truth value.
            """
            dt = row["prediction_date"]
            gt_df = self.org_unknown[self.org_unknown.date == dt]
            if len(gt_df) < 1:
                return None
            gt = gt_df.iloc[0][metric_name]
            return gt

        if len(result) > 0:
            result[gt_col_name] = result.apply(get_gt, axis=1)

        _df = result[result[gt_col_name] >= 0]
        gt_ts = _df[gt_col_name].to_list()
        pred_ts = _df["{}_mean".format(metric_name)].to_list()
        return scipy.stats.pearsonr(gt_ts, pred_ts)[0]

    def calculate_binary_hit(self, confidence: float = 0.95) -> pd.DataFrame:
        """Calculate binary hit for each predicted point.

        Args:
            confidence (int, optional): the confidence interval. Defaults to 0.95.

        Returns:
            pd.DataFrame: the result DataFrame containing the binary hit column.
        """
        metric_name = self.dataset.metric_name
        lb_name = "{}_{}_lower_bound".format(metric_name, str(int(confidence * 100)))
        ub_name = "{}_{}_upper_bound".format(metric_name, str(int(confidence * 100)))
        result = self.get_extrapolations_metric_mean_confidence_interval(confidence)

        def calc_hit(row: dict) -> Union[int, None]:
            """Calculate hit for a particular predicted point.

            Args:
                row (dict): the predicted point contains many stats such as mean.

            Returns:
                int: hit (0), over-estimate (1), under-estimate (-1), not measurable (None).
            """
            dt = row["prediction_date"]
            gt_df = self.org_unknown[self.org_unknown.date == dt]
            if len(gt_df) < 1:
                return None
            gt = gt_df.iloc[0][metric_name]
            lb, ub = row[lb_name], row[ub_name]
            if gt < lb:
                return 1
            elif gt > ub:
                return -1
            else:
                return 0

        if len(result) > 0:
            result[
                "{}_{}_hit".format(metric_name, str(int(confidence * 100)))
            ] = result.apply(calc_hit, axis=1)
        return result

    def calculate_acc(self) -> float:
        """Calculate the accuracy of the entire extrapolation.

        Returns:
            float: the calculated accuracy.
        """
        _df = self.calculate_binary_hit()
        _df = _df[_df.revenue_95_hit >= 0]
        if len(_df) == 0:
            return 0
        else:
            return len(_df[_df.revenue_95_hit == 0]) / len(_df)

    def calculate_neg_likelihood(self, bandwidth: int = 1000) -> pd.DataFrame:
        """Calculate negative log likelihood (NLL) for each predicted point.

        Args:
            bandwidth (int, optional): the bandwidth parameter for KDE. Defaults to 1000.

        Returns:
            pd.DataFrame: the result DataFrame that contains the NLL column.
        """
        if self.neg_likelihood is not None:
            return self.neg_likelihood
        metric_name = self.dataset.metric_name
        dates2test = self.org_unknown_dates
        result = []
        for dt in dates2test:
            gt = self.org_unknown[self.org_unknown.date == dt].iloc[0][metric_name]
            xs = []
            for trial in self.trials:
                val = trial.get_metric_at(dt)
                if val is not None:
                    xs.append(val)
            if len(xs) < 3:
                score = [None]
            else:
                kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
                kde.fit(np.asarray(xs)[:, None])
                score = -kde.score_samples([[gt]])
            result.append(
                {
                    "start_date": trial.first_dt,
                    "prediction_date": dt,
                    "negative_likelihood": score[0],
                }
            )
        self.neg_likelihood = pd.DataFrame(result)
        return self.neg_likelihood

    def calculate_nll(self) -> float:
        """Calculate the negative log likelihood (NLL) measure for this extrapolation.

        Returns:
            float: the calculated NLL value.
        """
        return np.mean(self.calculate_neg_likelihood()["negative_likelihood"].to_list())

    def get_extrapolations_metric_mean_confidence_interval(
        self, confidence: float = 0.95
    ) -> pd.DataFrame:
        """Obtain the stats (e.g. mean and confidence interval bounds) for this extrapolation.

        Args:
            confidence (int, optional): the confidence interval. Defaults to 0.95.

        Returns:
            pd.DataFrame: the result DataFrame with mean, min, max, upper&lower CI bounds.
        """
        metric_name = self.dataset.metric_name
        metric_df = self.get_pred_metric_df()
        result = []
        for dt in metric_df.date.unique():
            tmp_df = metric_df[metric_df.date == dt]
            if len(tmp_df) < 3:
                continue
            batch = list(tmp_df[metric_name])
            min_batch, max_batch = min(batch), max(batch)
            min_batch = np.clip(min_batch, a_min=0, a_max=None)
            max_batch = np.clip(max_batch, a_min=0, a_max=None)
            avg, low, high = Extrapolations.mean_confidence_interval(batch, confidence)
            # avoid revenue prediction becomes negative
            avg = np.clip(avg, a_min=0, a_max=None)
            low = np.clip(low, a_min=0, a_max=None)
            high = np.clip(high, a_min=0, a_max=None)
            # avoid confidence band violates min/max
            if min_batch > low:
                min_batch = low
            if max_batch < high:
                max_batch = high
            # the previous signal is also appended
            result.append(
                {
                    "prediction_date": dt,
                    "{}_mean".format(metric_name): avg,
                    "{}_{}_lower_bound".format(
                        metric_name, str(int(confidence * 100))
                    ): low,
                    "{}_{}_upper_bound".format(
                        metric_name, str(int(confidence * 100))
                    ): high,
                    "{}_min".format(metric_name): min_batch,
                    "{}_max".format(metric_name): max_batch,
                }
            )
        return pd.DataFrame(result)

    def get_smooth_metric_mean_confidence_interval(
        self, confidence: float = 0.95
    ) -> pd.DataFrame:
        """Obtain the stats (e.g. mean and CI bounds) for historical smooth points.

        Args:
            confidence (int, optional): the confidence interval. Defaults to 0.95.

        Returns:
            pd.DataFrame: the result DataFrame with mean, min, max, upper&lower CI bounds.
        """
        metric_name = self.dataset.metric_name
        metric_df = self.get_smooth_metric_df()
        result = []
        for dt in metric_df.date.unique():
            tmp_df = metric_df[metric_df.date == dt]
            if len(tmp_df) < 3:
                continue
            batch = list(tmp_df[metric_name])
            avg, low, high = Extrapolations.mean_confidence_interval(batch, confidence)
            # the previous signal is also appended
            result.append(
                {
                    "date": dt,
                    "{}_mean".format(metric_name): avg,
                    "{}_{}_lower_bound".format(
                        metric_name, str(int(confidence * 100))
                    ): low,
                    "{}_{}_upper_bound".format(
                        metric_name, str(int(confidence * 100))
                    ): high,
                    "{}_min".format(metric_name): min(batch),
                    "{}_max".format(metric_name): max(batch),
                }
            )
        return pd.DataFrame(result)

    def get_quantile_min_max(
        self, metric_value: float, percentiles: list
    ) -> Union[Tuple[float, float], Tuple[None, None]]:
        """Identify the percentile boundaries (upper and lower) for a metric value.

        Args:
            metric_value (float): the value of the metric (e.g. revenue).
            percentiles (list): the percentile boundaries.

        Returns:
            Union[Tuple[float, float], Tuple[None, None]]: the identified lower and upper boundaries.
        """
        if metric_value >= percentiles[-1]:
            x_min = percentiles[-1]
            x_max = math.inf
        elif metric_value < percentiles[0]:
            x_max = percentiles[0]
            x_min = -math.inf
        else:
            for i in range(len(percentiles) - 1):
                if metric_value < percentiles[i + 1]:
                    x_min = percentiles[i]
                    x_max = percentiles[i + 1]
                    break
        if "x_min" in locals() and "x_max" in locals():
            return x_min, x_max
        else:
            warnings.warn("get_quantile_min_max can not calculate x_min or x_max.")
            return None, None

    def coarse_filter_on_metric(
        self,
        metric_name: str,
        metric_value: float,
        high_low_percentage: float = 0.5,
        cutoff_dt: Union[datetime, None] = None,
    ) -> pd.DataFrame:
        """Perform date and revenue filtering to obtain a coarse benchmarking set.

        Args:
            metric_name (str): the name of the metric (e.g. revenue).
            metric_value (float): the value of the current metric.
            high_low_percentage (float, optional): the revenue filter tolerance. Defaults to 0.5.
            cutoff_dt (Union[datetime, None], optional): the filtering date. Defaults to None.

        Returns:
            pd.DataFrame: the coarsely filtered benchmarking dataset.
        """
        upper_val = metric_value * (1 + high_low_percentage)
        lower_val = metric_value * (1 - high_low_percentage)
        if cutoff_dt is None:
            return self.data[
                (self.data[metric_name] < upper_val)
                & (self.data[metric_name] > lower_val)
            ]
        else:
            return self.data[
                (self.data.date <= cutoff_dt)
                & (self.data[metric_name] < upper_val)
                & (self.data[metric_name] > lower_val)
            ]

    def run(self, params: dict = {}) -> None:
        """Run this extrapolation.

        Args:
            params (dict, optional): additional parameters such as {"filter_type": "smooth"};
                Defaults to {}.
        """
        self.run_params = params
        i = 0
        while i < len(self.trials):
            # log total number of trials
            self.n_total_trial += 1
            if self.n_total_trial >= self.max_attempts:
                break
            # run a trial
            run_result = self.run_filter(i, **params, sample_x_method=self.method)
            # check completeness of the trial
            is_full = run_result == self.extrapolate_len - 1
            if is_full:
                self.n_full_trial += 1
            # handle differently of two cases:
            # - does not require full trajectory, i.e. self.full_trajectory == False
            # - require full trajectory, i.e. self.full_trajectory == True
            if (not self.full_trajectory) or (self.full_trajectory and is_full):
                i += 1

    def is_empty(self) -> bool:
        """Check if this extrapolation is still not run yet.

        Returns:
            bool: the check result of either empty (True) or not (False).
        """
        n_empty_trials = 0
        for trial in self.trials:
            if trial.is_empty():
                n_empty_trials += 1
        return n_empty_trials == len(self.trials)

    def one_step_simulation(
        self,
        _last_signal: float,
        _last_growth: float,
        _last_dt: datetime,
        sample_x_method: str,
    ) -> Tuple[float, float, list, pd.DataFrame]:
        """Perform one-step simulation into the future based on the current known/predicted step.

        Args:
            _last_signal (float): the latest known metric (e.g. revenue).
            _last_growth (float): the latest known growth (e.g. YoY growth/multiply).
            _last_dt (datetime): the date of the latest known metric.
            sample_x_method (str): the method to perform multiply (growth) sampling.

        Raises:
            NotImplementedError: Illegal sample_x_method.

        Returns:
            Tuple[float, float, list, pd.DataFrame]: the measured next metric,
                the sampled next growth/multiply, percentiles, and the benchmarking group.
        """
        metric_name = self.dataset.metric_name
        growth_metric_name = self.dataset.growth_metric_name
        next_growth_metric_name = self.dataset.next_growth_metric_name
        benchmark_group = self.coarse_filter_on_metric(
            metric_name, metric_value=_last_signal, cutoff_dt=_last_dt
        )
        percentiles = benchmark_group[growth_metric_name].quantile(self.quantiles)
        percentiles = sorted(percentiles.to_dict().values())
        x_min, x_max = self.get_quantile_min_max(_last_growth, percentiles)
        if x_min is None or x_max is None:
            return np.ma.masked, np.ma.masked, percentiles, None
        benchmark_percentile_group = benchmark_group[
            (benchmark_group[growth_metric_name] < x_max)
            & (benchmark_group[growth_metric_name] > x_min)
        ]
        bpg = benchmark_percentile_group[["id", "date", next_growth_metric_name]]
        # sample x
        xs = np.asarray(benchmark_percentile_group[next_growth_metric_name])
        if len(xs) < 3:
            return np.ma.masked, np.ma.masked, percentiles, bpg
        if sample_x_method == "probability_matching":
            kde = KernelDensity(bandwidth=0.1, kernel="gaussian")
            kde.fit(xs[:, None])
            all_xs = np.asarray(benchmark_group[next_growth_metric_name])
            min_lim, max_lim = all_xs.min(), all_xs.max()
            sampled_x = kde.sample(100).flatten()
            sampled_x = sampled_x[(sampled_x > min_lim) & (sampled_x < max_lim)]
            if len(sampled_x) < 1:
                return np.ma.masked, np.ma.masked, percentiles, bpg
            selected_x = random.choice(sampled_x)
        elif sample_x_method == "bootstrapping":
            sampled_x = random.choices(xs, k=len(xs))
            if len(sampled_x) < 1:
                return np.ma.masked, np.ma.masked, percentiles, bpg
            selected_x = np.mean(sampled_x)
        else:
            raise NotImplementedError(
                "Illegal sample_x_method: {}".format(sample_x_method)
            )
        # calc metric approximation
        return (
            (selected_x ** (1 / self.yoy_step)) * _last_signal,
            selected_x,
            percentiles,
            bpg,
        )

    def run_filter(
        self,
        trial_id: int,
        sample_x_method: str = None,
        filter_type: str = None,
    ) -> int:
        """Perform extrapolation for one trial.

        Args:
            trial_id (int): the ID of the current trial.
            sample_x_method (str, optional): the method to perform multiply (growth) sampling.
                Defaults to None.
            filter_type (str, optional): the type of filtering (smooth/filter). Defaults to None.

        Raises:
            NotImplementedError: Illegal filter_type.

        Returns:
            int: the step index where it can not extrapolate any more, or the end of the max_len.
        """
        metric_name = self.dataset.metric_name
        growth_metric_name = self.dataset.growth_metric_name
        if (
            sample_x_method is None
            or sample_x_method not in Extrapolations.sample_x_methods
        ):
            if random.uniform(0, 1) > 0.3:
                sample_x_method = Extrapolations.sample_x_methods[0]
            else:
                sample_x_method = Extrapolations.sample_x_methods[1]
        ## Fit a smooth Kalman Filter on known data
        # At least 3 data points should be available
        org_df = self.data[self.data.id == self.org_id]
        if len(org_df) < 3:
            return 0
        # Obtain true signal (w. dates imputed)
        raw_dt_list = sorted(org_df.date.to_list())  # sorted() is redundant
        earliest_dt, latest_dt = raw_dt_list[0], raw_dt_list[-1]
        months_diff = (
            ((latest_dt.year - earliest_dt.year) * 12)
            + latest_dt.month
            - earliest_dt.month
        )
        assert months_diff >= 2
        org_dict = (
            org_df[["date", metric_name, growth_metric_name]]
            .set_index("date")
            .T.to_dict("list")
        )
        i = 1
        signal = [org_dict[earliest_dt][0]]
        signal_dts = [earliest_dt]
        signal_growth = [org_dict[earliest_dt][1]]
        while True:  # loop through the dates between earliest and latest
            _dt = earliest_dt + relativedelta(months=+i)
            if _dt > latest_dt:
                break
            if _dt in org_dict:
                signal.append(org_dict[_dt][0])
                signal_growth.append(org_dict[_dt][1])
            else:
                signal.append(np.ma.masked)
                signal_growth.append(None)
            signal_dts.append(_dt)
            i += 1
        assert len(signal) == months_diff + 1
        assert type(signal[0]) == float
        assert len(signal_dts) == len(signal)
        assert len(signal) == len(signal_growth)
        # requirement: the first 3 signals must exist
        s_idx = Extrapolations.validate_signal(signal)
        if s_idx < 0:
            warnings.warn("no consecutive 3 signals found!")
            return 0
        else:
            signal = signal[s_idx:]
            signal_dts = signal_dts[s_idx:]
            signal_growth = signal_growth[s_idx:]
        # Obtain measurement of signal
        measurement, measurement_dts, measurement_x = [], [], []
        for i in range(len(signal) + 1):
            if i >= len(signal):
                _dt = signal_dts[i - 1] + relativedelta(months=+1)
            else:
                _dt = signal_dts[i]
            measurement_dts.append(_dt)
            # the first measure is not calculable
            if i == 0:
                measurement.append(np.ma.masked)
                measurement_x.append(np.ma.masked)
                continue
            _last_signal, _last_dt = signal[i - 1], signal_dts[i - 1]
            _last_growth = signal_growth[i - 1]
            if _last_growth is None:
                measurement.append(np.ma.masked)
                continue
            # perform one-step simulation
            this_measurement, selected_x, _, _ = self.one_step_simulation(
                _last_signal, _last_growth, _last_dt, sample_x_method
            )
            measurement.append(this_measurement)
            measurement_x.append(selected_x)
        assert np.ma.is_masked(measurement[0])
        assert len(measurement) == len(measurement_dts)
        assert len(measurement) == len(signal) + 1
        assert measurement_dts[0] == signal_dts[0]
        assert measurement_dts[-1] == signal_dts[-1] + relativedelta(months=+1)
        self.trials[trial_id].measurement = measurement
        self.trials[trial_id].measurement_dts = measurement_dts
        self.trials[trial_id].measurement_x = measurement_x
        # Kalman filter
        sw = 1  # step width
        transition_cov_multiply = 0.2
        # average difference between signal and measurement
        mean_d = (np.ma.array(measurement[:-1]) - np.ma.array(signal))[1:]
        if np.ma.count(mean_d) < 1:
            warnings.warn("mean_d is all masked!")
            return 0
        else:
            mean_d = mean_d.mean()
        observations = np.ma.array(measurement)
        if np.ma.count(observations) < 1:
            warnings.warn("no valid measurements found!")
            return 0
        # constant_diff
        # State X = [p,q,v,a,d]
        transition_matrix = np.array(
            [
                [0, 1, sw, 0.5 * sw**2, sw],
                [0, 1, sw, 0.5 * sw**2, 0],
                [0, 0, 1, sw, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
            ]
        )
        observation_matrix = np.array([1, 0, 0, 0, 0])
        # initial states
        p_0 = signal[0] + mean_d  # approximated
        q_0 = signal[0]
        v_0 = (signal[1] - signal[0]) / sw
        a_0 = ((signal[2] - signal[1]) - (signal[1] - signal[0])) / (2 * sw)
        d_0 = mean_d  # approximated
        # fitting and smoothing on historical observations
        kf = KalmanFilter(
            transition_matrices=transition_matrix,
            transition_covariance=transition_cov_multiply
            * np.eye(transition_matrix.shape[0]),
            observation_matrices=observation_matrix,
            initial_state_mean=np.array([p_0, q_0, v_0, a_0, d_0]),
            # em_vars="all",
            em_vars=self.em_vars,
        )
        kf = kf.em(
            X=np.ma.array(observations),
            # em_vars="all",
            em_vars=self.em_vars,
        )
        if filter_type is None:
            if random.uniform(0, 1) > 0.5:
                x_mean, x_covar = kf.filter(np.ma.array(observations))
            else:
                x_mean, x_covar = kf.smooth(observations)
        elif filter_type == "smooth":
            x_mean, x_covar = kf.smooth(observations)
        elif filter_type == "filter":
            x_mean, x_covar = kf.filter(np.ma.array(observations))
        else:
            raise NotImplementedError("Illegal filter_type: {}".format(filter_type))
        # note that the last item in x_mean, x_covar is not valid for extrapolation
        x_mean = x_mean[:-1]
        x_covar = x_covar[:-1]
        ## Extrapolate
        latest_known_signal = signal[-1]
        latest_know_growth = signal_growth[-1]
        for i in range(self.extrapolate_len):
            current_known_dt = signal_dts[-1] + relativedelta(months=+i)
            # perform one-step simulation
            this_measurement, selected_x, percentiles, bpg = self.one_step_simulation(
                latest_known_signal,
                latest_know_growth,
                current_known_dt,
                sample_x_method,
            )
            # If not being able to obtain any "observation" via simulation, stop here!
            if np.ma.is_masked(this_measurement):
                if self.full_trajectory:
                    return i
                else:
                    break
            x_mean_, x_covar_ = kf.filter_update(
                filtered_state_mean=x_mean[-1],
                filtered_state_covariance=x_covar[-1],
                observation=[this_measurement],
            )
            filtered_measurement = x_mean_[0]
            latest_know_growth = (filtered_measurement / latest_known_signal) ** 12
            # logging
            self.trials[trial_id].pred_metric_raw.append(filtered_measurement)
            self.trials[trial_id].pred_growth_metric.append(latest_know_growth)
            self.trials[trial_id].percentiles.append(percentiles)
            self.trials[trial_id].support_samples.append(bpg)
            # rebase
            latest_known_signal = filtered_measurement
            x_mean = np.append(x_mean, x_mean_.reshape(1, -1), axis=0)
            x_covar_ = np.expand_dims(x_covar_, axis=0)
            x_covar = np.append(x_covar, x_covar_, axis=0)
        # global smoothing
        x_mean, x_covar = kf.smooth(x_mean[:, 0])
        self.trials[trial_id].x_mean = x_mean
        self.trials[trial_id].x_covar = x_covar
        assert len(x_mean) >= len(signal)
        # filling the final predictions
        pred_x_means = x_mean[len(signal) :, 1]
        self.trials[trial_id].pred_metric = list(pred_x_means)
        self.trials[trial_id].smooth_metric = list(x_mean[: len(signal), 1])
        pred_x_covars = np.sqrt(x_covar[len(signal) :, 1, 1])
        self.trials[trial_id].pred_metric_upper = list(pred_x_means + pred_x_covars)
        self.trials[trial_id].pred_metric_lower = list(pred_x_means - pred_x_covars)
        assert len(self.trials[trial_id].pred_metric) == len(x_mean) - len(signal)

        return i
