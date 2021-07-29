from typing import Any, Union, Optional, Tuple, List


import pytest
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, KFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.validation import check_is_fitted

from Core.reconciliation import To_Reconcile

def test_initialized() -> None:
    """Test that initialization does not crash."""
    To_Reconcile(1, 1, 1)

def test_check_parameters() -> None :
    object = To_Reconcile(np.array([[0,0,1],[1,0,1],[1,1,1]]), np.array([[4,7,6],[1,2,5],[9,3,8]]), np.array([4,7,6]))
    object._check_parameters()

#def test_reconcile() -> None :
    #object = To_Reconcile(np.matrix([[4,7,6],[1,2,5],[9,3,8]]), np.matrix([[4,7,6],[1,2,5],[9,3,8]]), np.array([4,7,6]))
    #object.reconcile(method='OLS')

    
