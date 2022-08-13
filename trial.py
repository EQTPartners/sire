"""
Copyright (C) eqtgroup.com Ltd 2022
https://github.com/EQTPartners/sire
License: MIT, https://github.com/EQTPartners/sire/LICENSE.md
"""


from datetime import datetime
from typing import Union

from dateutil.relativedelta import *


class Trial:
    """Trial Class is an abstract of one simulation. Term `metric` can be revenue."""

    def __init__(self, trial_id: int, latest_known_dt: datetime, method: str) -> None:
        """Trial initializer.

        Args:
            trial_id (int): the numerical ID, e.g. 1, 2, 3, etc.
            latest_known_dt (datetime): the latest calendar date when the metric is known.
            method (string): the method to sample metric multiply for the next period;
                the recommended value is "probability_matching".
        """
        self.method = method
        self.id = trial_id
        self.first_dt = latest_known_dt + relativedelta(months=+1)
        self.pred_metric = []
        self.smooth_metric = []
        self.pred_metric_raw = []
        self.pred_growth_metric = []
        self.percentiles = []
        self.support_samples = []

    def n_steps(self) -> int:
        """Get the number of metric points extrapolated.

        Returns:
            int: the number of metric points extrapolated.
        """
        return len(self.pred_metric)

    def is_empty(self) -> bool:
        """Check if the trial is empty.

        Returns:
            bool: empty trial (True) or not (False).
        """
        return self.n_steps() == 0

    def date2index(self, dt: datetime) -> int:
        """Translate a date into an index number.

        Args:
            dt (datetime): a calendar date in the prediction range.

        Returns:
            int: the index number of the input date; the first_dt should have index 0.
        """
        return ((dt.year - self.first_dt.year) * 12) + dt.month - self.first_dt.month

    def get_metric_at(self, dt: datetime) -> Union[float, None]:
        """Retrieve the predicted value for a date.

        Args:
            dt (datetime): a calendar date within the prediction date range.

        Returns:
            Any[float, None]: the predicted metric value if input date is valid;
                otherwise, None is returned.
        """
        idx = self.date2index(dt)
        if len(self.pred_metric) > idx:
            return self.pred_metric[idx]
        else:
            return None
