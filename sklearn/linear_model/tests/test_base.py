# Author: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#         Fabian Pedregosa <fabian.pedregosa@inria.fr>
#
# License: BSD Style.

from numpy.testing import assert_array_almost_equal
import numpy as np
from scipy import sparse

from ..base import LinearRegression
from ...utils import check_random_state


def test_linear_regression():
    """
    Test LinearRegression on a simple dataset.
    """
    # a simple dataset
    X = [[1], [2]]
    Y = [1, 2]

    clf = LinearRegression()
    clf.fit(X, Y)

    assert_array_almost_equal(clf.coef_, [1])
    assert_array_almost_equal(clf.intercept_, [0])
    assert_array_almost_equal(clf.predict(X), [1, 2])

    # test it also for degenerate input
    X = [[1]]
    Y = [0]

    clf = LinearRegression()
    clf.fit(X, Y)
    assert_array_almost_equal(clf.coef_, [0])
    assert_array_almost_equal(clf.intercept_, [0])
    assert_array_almost_equal(clf.predict(X), [0])

def test_fit_intercept():
    """
    Test assertions on betas shape.
    """
    X2 = np.array([[ 0.38349978,  0.61650022],
                   [ 0.58853682,  0.41146318]])
    X3 = np.array([[ 0.27677969,  0.70693172,  0.01628859],
                   [ 0.08385139,  0.20692515,  0.70922346]])
    Y = np.array([1,1])

    lr2_without_intercept = LinearRegression(fit_intercept=False).fit(X2, Y)
    lr2_with_intercept = LinearRegression(fit_intercept=True).fit(X2, Y)

    lr3_without_intercept = LinearRegression(fit_intercept=False).fit(X3, Y)
    lr3_with_intercept = LinearRegression(fit_intercept=True).fit(X3, Y)

    assert lr3_with_intercept.coef_.shape == lr3_without_intercept.coef_.shape
    assert lr2_with_intercept.coef_.shape == lr2_without_intercept.coef_.shape

    shape2 = lr2_without_intercept.coef_.shape
    shape3 = lr3_without_intercept.coef_.shape

    assert len(shape2) == len(shape3)

def test_linear_regression_sparse(random_state=0):
    "Test that linear regression also works with sparse data"
    random_state = check_random_state(random_state)
    n = 100
    X = sparse.eye(n, n)
    beta = random_state.rand(n)
    y = X * beta[:, np.newaxis]

    ols = LinearRegression()
    ols.fit(X, y.ravel())
    assert_array_almost_equal(beta, ols.coef_ + ols.intercept_)
    assert_array_almost_equal(ols.residues_, 0)
