from typing import Any, Union, Optional, Tuple, List


import pytest
import numpy as np
import pandas as pd


from Core.reconciliation import To_Reconcile

def test_initialized() -> None:
    """Test that initialization does not crash."""
    To_Reconcile(1, 1, 1)

def test_check_parameters() -> None :
    object = To_Reconcile(pd.DataFrame(), np.array([[4,7,6],[1,2,5],[9,3,8]]), np.array([4,7,6]))
    object._check_parameters()

#def test_reconcile() -> None :
    #object = To_Reconcile(np.matrix([[4,7,6],[1,2,5],[9,3,8]]), np.matrix([[4,7,6],[1,2,5],[9,3,8]]), np.array([4,7,6]))
    #object.reconcile(method='OLS')

    
