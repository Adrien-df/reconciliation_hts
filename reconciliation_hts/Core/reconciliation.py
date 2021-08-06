from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import random
from utils import cross_product, scale
from statsmodels.stats.moment_helpers import cov2corr
import warnings
import matplotlib.pyplot as plt


class To_Reconcile:
    """[Class for instantiating the whole problem]

        [This class implements reconciliation methods for hierarchical TS
         including state-of-the-art methods such as MinTShrinkage
         (Wickramasuriya and al. 2021). Assess wether reconcilation improves
         the performances of your models. Reconciliation aims at improving the
          forecasts accross all levels of a set of hierarchical time series.
          Important : reconciliation supposes that beforehand, forecasts were
          computed. Reconciliation comes a posteriori of foreacsting.]

        Parameters
        ----------
        base_forecasts : np.ndarray
            [Numpy array with the base (original) models computed by own models
             Shape=(n,p) with n = number of series accross the whole hierarchy
             and p the number of forecasts (size of test set). n>=3;p>=1
             The first row must be the 'total' time series at the top of
             the hierarchical structure]

        error_matrix : Optional[np.ndarray]
            [The matrix of the residuals (forecast-real value) of your
            models that were evaluated on the train set or a calibration set.
             Shape (n,k). Same n with the same order (i-th row is the i-th
             time series from base_forecasts). 
             Necessary for 'VS', 'MinTSa' and 'MinTSh' reconciliation methods],
             by default None

        data : Optional[pd.DataFrame], optional
            [Pandas dataframe from which the structure will be
             computed. See example for understanding expected format],
              by default None

        columns_ordered : Optional[list[str]], optional
            [Provided with data parameter. List of string
            with name of columns from data parameter and ordered such
            as the first element represent top hierarchical level ],
             by default None

        summing_mat : Optional[np.ndarray], optional
            [Summing matrix. Automatically computed if data
            and columns_ordered provided. Otherwise, must me an
            aggregating matrix. More details in theoretical description],
             by default None

        real_values : Optional[np.ndarray], optional
            [Numpy array with the real values from the test set.
             Shape=(n,p) with n = number of series accross the whole hierarchy
             and p the number of forecasts (size of test set). n>=3;p>=1
             The first row must be the 'total' time series at the top of
             the hierarchical structure], by default None

        lambd : [type], optional
            [Ignore this parameter], by default None

        inputs_are_checked : bool, optional
            [Ignore thi parameter], by default False
        """

    def __init__(
        self,
        base_forecasts: np.ndarray,
        error_matrix: Optional[np.ndarray] = None,
        data: Optional[pd.DataFrame] = None,
        columns_ordered: Optional[list[str]] = None,
        summing_mat: Optional[np.ndarray] = None,
        real_values: Optional[np.ndarray] = None,
        lambd=None,
        inputs_are_checked=False
    ) -> None:

        self.data = data
        self.summing_mat = summing_mat
        self.error_matrix = error_matrix
        self.base_forecasts = base_forecasts
        self.columns_ordered = columns_ordered
        self.real_values = real_values
        self.lambd = lambd
        self.inputs_are_checked = inputs_are_checked

    def _check_compatibility(self) -> None:
        """
        [Performs several checks on the compatibility of the inputs provided]

        [extended_summary]

        Raises
        ------
        ValueError
            [If mutually exclusive parameters are given in input]
        """
        if not self.inputs_are_checked:

            if self.data is not None:
                if self.summing_mat is not None:
                    raise ValueError(
                        "data and summing_mat were provided."
                        "You provide summing_mat OR data + columns_ordered"
                    )

            if self.columns_ordered is not None:
                if self.summing_mat is not None:
                    raise ValueError(
                        "columns_ordered and summing_mat were provided."
                        "You provide summing_mat OR data + columns_ordered "
                    )

            if self.data is None and self.summing_mat is None:
                raise ValueError(
                    "You provide summing_mat OR data + columns_ordered"
                )

    def _check_parameters(self) -> None:
        """
        [Perform several checks on input parameters]

        [extended_summary]

        Raises
        ------
        ValueError
            [If one of the parameters is not valid]
        """

        if not isinstance(self.data, (pd.DataFrame, type(None))):
            raise ValueError(
                "Invalid type for data."
                "Expected pd.DataFrame object"
            )
        if self.summing_mat is not None:
            if not isinstance(self.summing_mat, (np.ndarray, type(None))):
                raise ValueError(
                    "Invalid type for summing matrix"
                    " Expected np.ndarray type"
                )
            if self.summing_mat.shape[0] < 3:
                raise ValueError(
                    "Invalid summing_mat shape. Must be of shape >= 3."
                )

            if self.summing_mat.shape[0] != self.error_matrix.shape[0]:

                raise ValueError(
                    "Summing matrix must have as many rows as data"
                )
        if self.error_matrix.shape[0] != self.base_forecasts.shape[0]:
            raise ValueError(
                "The error matrix and base forecats must have"
                "the same number of rows"
            )

        if self.real_values is not None:
            if self.real_values.shape[0] != self.base_forecasts.shape[0]:
                raise ValueError(
                    "The real values and base forecats must have"
                    "the same number of rows"
                )
            if self.real_values.ndim > 1:
                if self.real_values.shape[1] != self.base_forecasts.shape[1]:
                    raise ValueError(
                        "Real values and base forecats"
                        "must have the same dimension"
                    )

        if not isinstance(self.base_forecasts, (np.ndarray, type(None))):
            raise ValueError(
                "Invalid type for base forecasts. Expected np.ndarray type"
            )

        if not isinstance(self.real_values, (np.ndarray, type(None))):
            raise ValueError(
                "Invalid type for real values. Expected numpy array type"
            )

        if not isinstance(self.error_matrix, (np.ndarray, type(None))):
            raise ValueError(
                "Invalid type for error_matrix. Expected numpy array type"
            )

        if self.data is not None:

            for element in self.columns_ordered:
                check = 1
                if element not in self.data.columns:
                    check = 0
                    break
            if check == 0:
                raise ValueError(
                    "Columns name provided in columns_ordered not in data"
                    "Check the columns and their names"
                )

            if self.data.shape[0] != self.base_forecasts.shape[0]:
                raise ValueError(
                    f" data has {self.data.shape[0]} rows"
                    "base_forecast has {self.base_forecasts.shape[0]} rows"
                    "Same value is expected"
                )

            if self.real_values is not None:
                if self.data.shape[0] != self.real_values.shape[0]:
                    raise ValueError(
                        f" data has {self.data.shape[0]} rows"
                        "real_values has {self.real_values.shape[0]} rows"
                        "Same value is expected."
                    )
            if self.error_matrix is not None:
                n = self.data.shape[0]
                m = self.error_matrix.shape[0]
                if n != m:
                    raise ValueError(
                        f" data has {self.data.shape[0]} rows"
                        "error_matrix has"
                        f"{self.error_matrix.shape[0]} rows"
                        "Same value is expected."
                    )

    def _check_method(
        self,
        method
    ) -> None:
        if method not in ['OLS', 'BU', 'SS', 'VS', 'MinTSa', 'MinTSh']:
            raise ValueError(
                "Invalid reconciliation method. "
                "Allowed values are 'OLS','BU', 'SS', 'VS', 'MinTSa', 'MinTSh'"
            )
        if method in ['VS', 'MinTSa', 'MinTSh']:
            if self.error_matrix is None:
                raise ValueError(
                    "No error_matrix was instantiated in the class."
                    f" For {method} reconciliation method,"
                    " error_matrix is compulsory"
                )

    def _check_metrics(
        self,
        metrics
    ) -> None:
        if metrics not in ['rmse', 'mase', 'mse']:
            raise ValueError(
                "Invalid method. "
                "Allowed values are 'rmse','mase','mse'."
            )

    def _check_real_values(self) -> None:
        if self.real_values is None:
            raise ValueError(
                "For computing the scores,"
                " you must provide the array of real values"
                " If real values has one column, use method score"
                " If real values has multiple columns, use cross_score method"
            )

    def _check_level(
        self,
        level
    ) -> None:
        if self.columns_ordered is None:
            raise ValueError(
                "You must provide columns_ordered parameter when"
                " you instantiate the class"
            )
        if level not in (self.columns_ordered + ['total']):
            raise ValueError(
                "The level parmameter must be 'total' or one element of"
                " columns_ordered"
            )

    def compute_summing_mat(
        self
    ) -> np.ndarray:
        """
        [Computes the summing_matrix]

        [When data and columns_label_ordered is passed in input,
        this method will automatically compute the summing matrix]

        Returns
        -------
        np.ndarray
            [The summing matrix used for reconciliation afterwards]
        """

        L = len(self.columns_ordered)
        n = self.data.shape[0]
        m = self.data[self.columns_ordered].isna().any(axis=1).sum()
        self.summing_mat = np.zeros([n, n-m])
        self.summing_mat[0] = np.ones(n-m)
        self.summing_mat[m:] = np.identity(n-m)

        for level in range(1, L):

            for i in range(1, n-m):
                z = level   # Just for PEP8 accomodation
                if pd.isna(self.data.iloc[i][self.columns_ordered[z]]) and not(
                        pd.isna(self.data.iloc[i][self.columns_ordered[z-1]])):
                    list_of_values = [
                        self.data.iloc[i][self.columns_ordered[k]]
                        for k in range(level)]

                    for j in range(m, n):
                        if [self.data.iloc[j][self.columns_ordered[k]]
                                for k in range(level)] == list_of_values:
                            self.summing_mat[i][j-m] = 1

        return(self.summing_mat)

    def _get_indexes_level(
        self
    ) -> dict:
        """[Generate the indexes of levels]

        [This function is used to compute a dictionnary where for a given
        hierarchical level you can access to all the indexes that correspond
        to this hierarchical level]

        Returns
        -------
        dict
            [Dictionnary where key = 'level' and
             value = [indexes of this level]]
        """
        dictionnary = {'total': 0}
        L = len(self.columns_ordered)  # assert that the lenght is sufficient
        n = self.data.shape[0]

        for level in range(1, L):
            indexes_of_level = []

            for i in range(1, n):
                if (pd.isna(self.data.iloc[i][self.columns_ordered[level]])
                        and not pd.isna(self.data.iloc[i]
                                        [self.columns_ordered[level-1]])):
                    indexes_of_level.append(i)
            dictionnary[self.columns_ordered[level-1]] = indexes_of_level

        indexes_of_level = []
        for i in range(1, n):
            if not pd.isna(self.data.iloc[i][self.columns_ordered[-1]]):
                indexes_of_level.append(i)
        dictionnary[self.columns_ordered[-1]] = indexes_of_level

        return(dictionnary)

    def _get_number_bottom_series(
        self,
    ) -> int:
        """[Compute number of bottom series]

        [This function is used to compute the number of series
        at the smallest hierarchical level (bottom series).
        It is used when doing reconciliation]

        Returns
        -------
        int
            [The number of bootom series]
        """
        n = (self.summing_mat).shape[0]
        i = 0
        while self.summing_mat[i].sum() != 1:
            i += 1

        return (n-i)

    def _compute_lambda(
        self
    ) -> float:
        """[Compute shrinkage parameter]

        [This method computes the shrinkage parameter
        that is necessary for reconciling with the 'MinTSh' method]

        Returns
        -------
        float
            [Shrinkage parameter, belongs to [0,1]]
        """

        res = np.matrix(self.error_matrix).T

        number_error_vectors = res.shape[0]

        covm = cross_product(res)/number_error_vectors

        corm = cov2corr(covm)

        xs = scale(res, np.sqrt(np.diag(covm)))

        v = (1/(number_error_vectors * (number_error_vectors - 1))) * (
            cross_product(np.square(
                np.matrix(xs))) - 1/number_error_vectors * (
                np.square(cross_product(np.matrix(xs)))))

        np.fill_diagonal(v, 0)

        corapn = cov2corr(np.diag(np.diag(covm)))

        d = np.square((corm - corapn))

        lambd = v.sum()/d.sum()
        lambd = max(min(lambd, 1), 0)
        self.lambd = lambd

        return(lambd)

    def reconcile(
        self,
        method: Optional[str] = 'MintSh',
        column_to_reconcile: Optional[int] = 0,
        reconcile_all: Optional[bool] = False,
        show_lambda: Optional[bool] = False,
        _vector_to_proba_reconcile: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """[Method for reconciling a vector or matrix of predictions]

        [This method implements the state-of-the-art reconciliation methods

        Choose first a reconciliation method. For benchmark and testing
        the paramters try 'BU', the bottom-up approach. For more sophisticated
        method try 'MinTSh' (default) or 'MinTSa'

        For reconciling a specific column of predictions :
        use column_to_reconcile, Default value = 0 (First column)

        If you want to reconcile all base forecasts, use reconcile_all =True

        With MintSh method, do show_lambda = True to display shrinkage value

        Ignore _vector_to_proba_reconcile (probabilistic reconciliation)]

        Parameters
        ----------
        method : Optional[str], optional
            [Method chosen for reconciliation. Default 'MintSh'
            is state_of_the_art. 'BU' for benchmark], by default 'MintSh'
        column_to_reconcile : Optional[int], optional
            [Index of the column you want to reconcile from base_forecast.
             Ignore if only one column in base_forecasts], by default 0
        reconcile_all : Optional[bool], optional
            [Wether you want to reconcile all the base forecasts or not],
             by default False
        _vector_to_proba_reconcile : Optional[np.ndarray], optional
            [Ignore it. Used internally for computing prediction intervals.
            (See method proba_reconcile())], by default None
        show_lambda : Optional[bool], optional
            [Wether to display the shrinkage estimator.
             Only for 'MintSh' method], by default False

        Returns
        -------
        np.ndarray
            [The reconciled numpy array]

        Raises
        ------
        ValueError
            [Error raised if parameters are not correct or mutually exclusive]
        """

        if not self.inputs_are_checked:
            self._check_compatibility()
            self._check_parameters()
            self._check_method(method=method)
            self.inputs_are_checked = True

        if self.summing_mat is None:
            self.summing_mat = self.compute_summing_mat()

        if method == 'OLS':
            combination_matrix = np.linalg.inv(np.transpose(
                self.summing_mat)@self.summing_mat)@(
                    np.transpose(self.summing_mat))

        elif method == 'BU':
            combination_matrix = np.concatenate(
                (np.zeros(shape=(self._get_number_bottom_series(),
                                 self.summing_mat.shape[0] -
                                 self._get_number_bottom_series())),
                    np.identity(self._get_number_bottom_series())), axis=1)

        elif method == 'SS':

            W = np.diag(self.summing_mat @
                        np.ones(self._get_number_bottom_series()))
            combination_matrix = np.linalg.inv(
                (self.summing_mat.T)@np.linalg.inv(W)@self.summing_mat)@(
                    self.summing_mat.T)@np.linalg.inv(W)

        elif method in ['VS', 'MinTSa', 'MinTSh']:

            number_error_vectors = self.error_matrix.shape[1]
            W1 = np.zeros(
                (self.summing_mat.shape[0], self.summing_mat.shape[0]))

            for i in range(number_error_vectors):
                W1 += (self.error_matrix[:, i][:, None])@(
                    self.error_matrix[:, i][:, None].T)
            W1 = W1/number_error_vectors

            if method == 'VS':
                W = np.diag(np.diag(W1))

            elif method == 'MinTSa':
                W = W1

            elif method == 'MinTSh':
                if self.lambd is None:
                    lambd = self._compute_lambda()
                    if show_lambda:
                        print(
                            "Shrinkage parameter of MinTSh reconciliation"
                            f" is equal to : {lambd}.")
                    if lambd < 0.1:
                        print(
                            "Shrinkage is close to 0, try 'MintSa' method"
                        )
                else:
                    lambd = self.lambd

                W = lambd*np.diag(np.diag(W1)) + (1-lambd) * W1

            combination_matrix = np.linalg.inv(
                (self.summing_mat.T)@np.linalg.inv(W)@self.summing_mat)@(
                    self.summing_mat.T)@np.linalg.inv(W)

        if _vector_to_proba_reconcile is not None:
            return(
                self.summing_mat@combination_matrix@_vector_to_proba_reconcile
            )

        if self.base_forecasts.ndim == 1:
            reconcile_all = True

        if reconcile_all:
            return(
                self.summing_mat@combination_matrix@self.base_forecasts
            )

        else:
            return(
                self.summing_mat@combination_matrix@(
                    self.base_forecasts[:, column_to_reconcile])
            )

    def score(
        self,
        metrics: Optional[str] = 'rmse',
        reconcile_method: Optional[str] = 'MinTSh',

    ) -> pd.DataFrame:
        """[Assess if reconciliation improves forecast]

        [This method enables you to compute the score of ONE reconciled
         forecast versus the original forecast. You choose your metrics
         and the reconciliation method. For computing the score on
          multiple reconciliations, see cross_score() method]

        Parameters
        ----------
        metrics : Optional[str], optional
            [metrics for evaluating distance to real values], by default 'rmse'
        reconcile_method : Optional[str], optional
            [method chosen for reconciliation], by default 'MinTSh'

        Returns
        -------
        pd.DataFrame
            [Pandas DataFrame with the two scores displayed]

        Raises
        ------
        ValueError
            [If base forecast parmameter has more than one vector of forecasts]
        """

        self._check_metrics(metrics=metrics)
        self._check_real_values()

        if self.base_forecasts.ndim > 1:
            raise ValueError(
                "Use cross_score(). score() method used when assessing "
                "if reconciliation improves ONE vector of base forecast"
                "Here, you instantiated the class with "
                f"{self.base_forecasts.shape[1]} base forecasts."
            )

        if metrics == 'rmse':
            score_dataframe = pd.DataFrame(data={'rmse': [mean_squared_error(
                self.real_values, self.base_forecasts, squared=False),
                mean_squared_error(
                self.real_values, self.reconcile(method=reconcile_method),
                squared=False)]})

        elif metrics == ' mse':
            score_dataframe = pd.DataFrame(data={'rmse': [mean_squared_error(
                self.real_values, self.base_forecasts, squared=False),
                mean_squared_error(
                self.real_values, self.reconcile(method=reconcile_method),
                squared=False)]})

        elif metrics == 'mase':
            score_dataframe = pd.DataFrame(data={'mase': [mean_absolute_error(
                self.real_values, self.base_forecasts),
                mean_absolute_error(
                self.real_values, self.reconcile(method=reconcile_method))]})

        score_dataframe.rename(index={
                               0: "Base forecast",
                               1: f"Reconciled forecast ({reconcile_method})"},
                               inplace=True)

        return(score_dataframe)

    def cross_score(
        self,
        reconcile_method: Optional[str] = 'MinTSh',
        metrics: Optional[str] = 'rmse',
        test_all: Optional[bool] = True,
        cv: Optional[int] = 5,

    ) -> pd.DataFrame:
        """[Asses if reconciliation improves the forecasts]

        [This method compares the performance of reconciled forecasts
         (in regards to the real values) with the performance of the original
          base forecasts. You can decide to compare the performance on all
          the test set or only on a random sample of the test set with
           test_all = False and by setting a value for cv ]

        Parameters
        ----------
        reconcile_method : Optional[str], optional
            [method chosen for reconciliation], by default 'MinTSh'
        metrics : Optional[str], optional
            [metrics for evaluating distance to real values], by default 'rmse'
        test_all : Optional[bool], optional
            [Wether you ant to test all base forecasts], by default False
        cv : Optional[int], optional
            [If test_all = False, size of sample to test], by default 5


        Returns
        -------
        pd.DataFrame
            [Pandas DataFrame with the two scores displayed]


        """

        self._check_metrics(metrics=metrics)
        self._check_real_values()

        if not test_all:
            indexes = random.sample(range(self.base_forecasts.shape[1]), cv)
        else:
            indexes = np.arange(self.base_forecasts.shape[1])
            self.cv = self.base_forecasts.shape[1]

        mean_score_real = 0
        mean_score_reconciled = 0

        if metrics == 'rmse':

            for index in indexes:
                mean_score_real += mean_squared_error(
                    self.real_values[:, index], self.base_forecasts[:, index],
                    squared=False)
                mean_score_reconciled += mean_squared_error(
                    self.real_values[:, index], self.reconcile(
                        method=reconcile_method, column_to_reconcile=index),
                    squared=False)

        elif metrics == 'mase':

            for index in indexes:
                mean_score_real += mean_absolute_error(
                    self.real_values[:, index], self.base_forecasts[:, index])
                mean_score_reconciled += mean_absolute_error(
                    self.real_values[:, index], self.reconcile(
                        method=reconcile_method, column_to_reconcile=index))

        elif metrics == 'mse':

            for index in indexes:
                mean_score_real += mean_squared_error(
                    self.real_values[:, index], self.base_forecasts[:, index])
                mean_score_reconciled += mean_squared_error(
                    self.real_values[:, index], self.reconcile(
                        method=reconcile_method, column_to_reconcile=index))

        score_dataframe = pd.DataFrame(
            data={metrics: [mean_score_real/self.cv,
                            mean_score_reconciled/self.cv]})
        score_dataframe.rename(index={
                               0: "Base forecast",
                               1: f"Reconciled forecast ({reconcile_method})"},
                               inplace=True)

        return(score_dataframe)

    def plot(
        self,
        level: Optional[str] = 'total',
        reconcile_method: Optional[str] = 'MinTSa',
        columns: list[int] = [-1],
        plot_real: Optional[bool] = True,

    ) -> None:
        """[Plotting reconciled forecasts]

        [This method allows you to plot the reconciled forecasts, along with
         the base forecasts and the real values]

        Parameters
        ----------
        level : Optional[str], optional
            [The hierarchical level you wish to plot], by default 'total'
        reconcile_method : Optional[str], optional
            [The reconciliation method chosen], by default 'MinTSa'
        columns : list[int], optional
            [The list of index to plot. Default = all ], by default [-1]
        plot_real : Optional[bool], optional
            [If True, all the reconciled forecasts are plotted],
             by default True
        """

        self._check_real_values()
        self._check_level(level=level)
        indexes_of_series = self._get_indexes_level()

        if columns == [-1]:
            columns = np.arange(self.real_values.shape[1])
        plt.figure(figsize=(15, 6))
        if level == 'total':
            plt.plot(columns, self.base_forecasts[0, columns], color='red')
            plt.plot(columns, np.asarray([self.reconcile(
                method=reconcile_method,
                column_to_reconcile=i) for i in columns]).T[0],
                color='green')
            if plot_real:
                plt.plot(columns, self.real_values[0, columns],
                         color='blue')

        elif level != 'total':
            if len(indexes_of_series[level]) > 5:
                warnings.warn(
                    f"There are {len(indexes_of_series[level])} that are going"
                    " to be plotted"
                    " It is likely that the plot will be illisible"
                )
            for index in indexes_of_series[level]:
                plt.plot(columns, self.base_forecasts[index,
                         columns], color='red')
                plt.plot(columns, np.asarray([self.reconcile(
                    method=reconcile_method, column_to_reconcile=i)
                    for i in columns]).T[index],
                    color='green')
                if plot_real:
                    plt.plot(
                        columns,
                        self.real_values[index, columns], color='blue')
        if plot_real:
            plt.title(
                f"Real (blue), predicted (red), and reconciled with"
                f" {reconcile_method} method(green) values for the "
                f"{level} aggregation level")
        else:
            plt.title(
                f"Predicted (red), and reconciled with"
                f" {reconcile_method} method(green) values for the "
                f"{level} aggregation level")

        plt.show()

    def proba_reconcile(
        self,
        method: Optional[str] = 'MinTSh',
        alpha: Optional[float] = 0.05,
        samples_to_bootstrap: Optional[int] = -1,
        column_to_reconcile: Optional[int] = -1,
        reconcile_all: Optional[bool] = False
    ) -> None:

        # make sure the number of ssamples to bootstrap is smaller or equal
        # tahn error matrix shape [1]

        if samples_to_bootstrap == -1:
            samples_to_bootstrap = self.error_matrix.shape[1]
        if column_to_reconcile == -1:
            column_to_reconcile = self.base_forecasts.shape[1]

        sample_base_forecasts = np.zeros(
            shape=(self.base_forecasts.shape[0], samples_to_bootstrap))

        for s in range(samples_to_bootstrap):
            random_index = random.randint(
                0, self.error_matrix.shape[1]-1)

            sample_base_forecasts[:, s] = self.base_forecasts[
                :, column_to_reconcile] + self.error_matrix[
                    :, random_index]

        sample_reconciled_forecasts = np.zeros(
            shape=(self.base_forecasts.shape[0], samples_to_bootstrap))

        for s in range(samples_to_bootstrap):
            sample_reconciled_forecasts[:, s] = self.reconcile(
                method=method,
                _vector_to_proba_reconcile=sample_base_forecasts[:, s])

        test = np.quantile(sample_reconciled_forecasts,
                           q=[0.05, 0.5, 0.95], axis=1)
        # print(test.shape)
        # print(self.real_values[:,column_to_reconcile])
        high = 0
        low = 0

        for i in range(self.base_forecasts.shape[0]):
            if (self.real_values[:, column_to_reconcile][i] < test[0, i]):
                # print('low')
                low += 1
            if (self.real_values[:, column_to_reconcile][i] > test[2, i]):
                # print('high')
                high += 1

        score_dataframe = pd.DataFrame(data={'lower_bound':
                                             [test[0, i] for i in range(
                                                 114)], 'upper_bound':
                                             [test[2, i] for i in range(114)]})

        # . score_dataframe.rename(index={0: "Base forecast",
        # 1: f"Reconciled forecast ({reconcile_method})"},inplace=True)

        return(score_dataframe)
