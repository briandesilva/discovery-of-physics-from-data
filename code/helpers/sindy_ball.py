from scipy.integrate import odeint
from sklearn import linear_model
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt

from .library import get_library, get_library_index
from .differentiation import differentiate, smooth_data
from .utils import normalize_columns


class SINDyBall:
    """
    Class for inferring governing dynamical system from data.

    Includes functions for fitting a second order differential equation
    to data and for predicting forward in time from a set of initial
    conditions.

    """

    def __init__(
        self,
        thresh=0.1,
        diff_method="forward difference",
        smoother=None,
        window_length=35,
        remove_const=False,
        drag_coeff=None,
        library=None,
        library_type=None,
    ):
        """
        Parameters
        ----------
        thresh : float, optional
            Threshold for sparse regression
        diff_method : string, optional
            Which differentiation method to use.
            One of 'forward difference' or 'centered difference'
        smoother : string, optional
            Which smoothing method to use when performing differentiation.
            One of 'savgol', 'median', or 'wiener'
        window_length : integer, optional
            Window length to be used when performing smoothing
        drag_coeff : float, optional
            Coefficient multiplying velocity. If set, the model will fix
            the coefficient at this value instead of learning it
        library : dict, optional
            Library of functions to fit to data
        library_type : string, optional
            If library is not specified, the type of library that should be
            used.
            One of 'polynomial', 'polynomial v', or 'nonautonomous'
        remove_const : bool, optional
            Whether to fix constant library function at -9.8
        """

        self.thresh = thresh
        self.diff_method = diff_method
        self.smoother = smoother
        self.remove_const = remove_const
        self.mse = -1
        self.drag_coeff = drag_coeff
        self.library = library
        self.var_name = "v'"
        self.library_type = library_type
        self.Xi = None

        if self.library is None:
            if library_type is None:
                self.library = get_library(lib_type="polynomial", order=3)
            else:
                self.library = get_library(lib_type=library_type)

        if np.mod(window_length, 2) == 0:
            self.window_length = window_length - 1
        else:
            self.window_length = window_length

    def fit(
        self,
        h,
        t=None,
        method="least squares",
        iterations=10,
        normalize=False,
        multiple_runs=False,
        group_sindy=False,
        group_sparse_method="1-norm",
    ):
        """
        Fit a second order differential equation to data.

        Parameters
        ----------

        h : array_like
            Height data for model to fit.
            h can either be height data from a single ball drop or a list
            of array_like elements, each with data from a ball drop
        t : array_like, optional
            Time points corresponding to the data in h.
            Should have the same shape as h
        method : string, optional
            Sparse regression method to be used when fitting a differential
            equation to the data.
            One of 'least squares' (SINDy) or 'lasso'
        iterations : int, optional
            Iterations of sequential thresholded least squares to be used if
            method is set to 'least squares'
        normalize : bool, optional
            Whether to normalize vectors of library function values.
            If true, each library function vector will have unit norm
        multiple_runs : bool, optional
            Whether or not we should concatenate the results from multiple
            drops into one long data array.
        group_sindy : bool, optional
            Whether or not to use group sparsity method.
            If set to true, it is assumed h contains data from multiple drops
        group_sparse_method : string, optional
            The group sparsity penalty applied when using group SINDy.
            One of '1-norm', '2-norm', 'avg', 'quantile', or 'threshold'
        """

        # Approximate derivatives of h
        self.h = h
        self.multiple_runs = multiple_runs

        if t is None:
            if self.multiple_runs or group_sindy:
                t = [1] * len(h)
            else:
                t = 1

        if group_sindy:
            # Compute derivatives for runs separately
            h_smoothed = []
            v = []
            dv = []
            for h_run, t_run in zip(h, t):
                v_run = differentiate(
                    h_run,
                    t_run,
                    diff_method=self.diff_method,
                    smoother=self.smoother,
                    window_length=self.window_length,
                )
                dv_run = differentiate(
                    v_run,
                    t_run,
                    diff_method=self.diff_method,
                    smoother=self.smoother,
                    window_length=self.window_length,
                )
                h_smoothed.append(
                    smooth_data(
                        h_run, smoother=self.smoother, window_length=self.window_length,
                    )
                )
                v.append(v_run)
                dv.append(dv_run)
            self.v0 = [v_run[0] for v_run in v]

            # If multiple_runs, need to concatenate results from multiple runs;
            if self.multiple_runs:
                h = np.concatenate(h_smoothed)
                v = np.concatenate(v)
                dv = np.concatenate(dv)
                t = np.concatenate(t)

        else:
            v = differentiate(
                h,
                t,
                diff_method=self.diff_method,
                smoother=self.smoother,
                window_length=self.window_length,
            )
            dv = differentiate(
                v,
                t,
                diff_method=self.diff_method,
                smoother=self.smoother,
                window_length=self.window_length,
            )
            h = smooth_data(h, smoother=self.smoother, window_length=self.window_length)
            self.v0 = v[0]

        # Evaluate library on data
        if self.library_type == "nonautonomous":
            if group_sindy:
                X_full = [
                    np.stack([f(tt, hh, vv) for f in self.library.values()], axis=1)
                    for tt, hh, vv in zip(t, h, v)
                ]
            else:
                X_full = np.stack([f(t, h, v) for f in self.library.values()], axis=1)
        else:
            if group_sindy:
                X_full = [
                    np.stack([f(hh, vv) for f in self.library.values()], axis=1)
                    for hh, vv in zip(h, v)
                ]
            else:
                X_full = np.stack([f(h, v) for f in self.library.values()], axis=1)

        # Account for known variables
        if group_sindy:
            mask = np.ones(X_full[0].shape[1], dtype=bool)
            if self.remove_const:
                const_idx = get_library_index(self.library, "1")
                mask[const_idx] = False
                self.library["1"].coeff = -9.8
                for i in len(dv):
                    dv[i] += 9.8

            if self.drag_coeff is not None:
                v_idx = get_library_index(self.library, "v")
                mask[v_idx] = False
                self.library["v"].coeff = self.drag_coeff
                for i in len(dv):
                    dv[i] -= self.drag_coeff * v[i]

        else:
            mask = np.ones(X_full.shape[1], dtype=bool)
            if self.remove_const:
                dv += 9.8
                const_idx = get_library_index(self.library, "1")
                mask[const_idx] = False
                self.library["1"].coeff = -9.8

            if self.drag_coeff is not None:
                dv -= self.drag_coeff * v
                v_idx = get_library_index(self.library, "v")
                mask[v_idx] = False
                self.library["v"].coeff = self.drag_coeff

        if group_sindy:
            X = [Xf[:, mask] for Xf in X_full]
            if normalize:
                X_col_norms = [None] * len(X)
                for i in range(len(X)):
                    X[i], X_col_norms[i] = normalize_columns(X[i])
            else:
                X_col_norms = []
                for x in X:
                    X_col_norms.append(np.ones(x.shape[1]))

        else:
            X = X_full[:, mask]
            if normalize:
                X, X_col_norms = normalize_columns(X)
            else:
                X_col_norms = np.ones(X.shape[1])

        # Apply LASSO
        if method == "lasso":

            if group_sindy:
                raise ValueError("lasso method not yet supported when group_sindy=True")

            clf = linear_model.Lasso(alpha=self.thresh, fit_intercept=False, tol=0.001)
            clf.fit(X, dv)

            self.mse = np.mean((np.dot(X, clf.coef_) - dv) ** 2)

            clf.coef_ /= X_col_norms
            counter = 0
            for i, key in enumerate(self.library.keys()):
                if mask[i]:
                    self.library[key].coeff = clf.coef_[counter]
                    counter += 1

        # Apply iterated least-squares thresholding
        else:
            if not group_sindy:

                # Initial coefficients
                Xi = np.linalg.lstsq(X, dv, rcond=None)[0]
                small_inds_prev = 2 * np.ones_like(Xi)

                for k in range(iterations):
                    small_inds = np.abs(Xi) < self.thresh
                    if np.array_equal(small_inds, small_inds_prev):
                        break
                    small_inds_prev = small_inds

                    Xi[small_inds] = 0
                    big_inds = ~small_inds
                    if np.where(big_inds)[0].size == 0:
                        break
                    Xi[big_inds], self.mse = np.linalg.lstsq(
                        X[:, big_inds], dv, rcond=None
                    )[:2]

                self.mse = self.mse / len(h)
                if type(self.mse) != float:
                    self.mse = self.mse.item(0)

                Xi /= X_col_norms
                self.Xi = Xi
                counter = 0
                for i, key in enumerate(self.library.keys()):
                    if mask[i]:
                        self.library[key].coeff = Xi[counter]
                        counter += 1

            else:

                # Initial coefficients
                Xi = []
                for x, dvdt in zip(X, dv):
                    Xi.append(np.linalg.lstsq(x, dvdt, rcond=None)[0])
                small_inds_prev = 2 * np.ones_like(Xi[0])

                total_mse = np.inf
                for k in range(iterations):
                    small_inds = (
                        group_sparse_penalty(Xi, method=group_sparse_method)
                        < self.thresh
                    )
                    big_inds = ~small_inds

                    if np.array_equal(small_inds, small_inds_prev):
                        break
                    small_inds_prev = small_inds

                    if np.where(big_inds)[0].size == 0:
                        for xi in Xi:
                            xi[small_inds] = 0
                        break

                    total_mse = 0
                    for xi, x, dvdt in zip(Xi, X, dv):
                        xi[small_inds] = 0
                        xi[big_inds], mse_temp = np.linalg.lstsq(
                            x[:, big_inds], dvdt, rcond=None
                        )[:2]
                        total_mse += mse_temp / len(dvdt)

                self.mse = total_mse / len(Xi)
                if type(self.mse) != float:
                    self.mse = self.mse.item(0)

                for i, xcn in enumerate(X_col_norms):
                    Xi[i] /= xcn

                # For now, uses average coefficient values
                self.Xi = Xi
                avg_coeffs = np.mean(np.stack(Xi, axis=1), axis=1)

                counter = 0
                for i, key in enumerate(self.library.keys()):
                    if mask[i]:
                        self.library[key].coeff = avg_coeffs[counter]
                        counter += 1

    def predict(self, y0=None, t=None):
        """
        Use learned differential eqaution to predict forward in time.

        Parameters
        ----------

        y0 : array_like, optional
            Initial conditions
        t : array_like, optional
            Time points where predictions should be made

        Output
        ------

        sol : array_like,
            Predictions for each variable at each point in t
        """

        # Construct the right-hand side for
        #   h' = v
        #   v' = f(h,v,t)

        if self.library_type == "nonautonomous":

            def f(y, t):
                h, v = y
                dvdt = sum((f.eval(t, h, v) for f in self.library.values()))
                return [v, dvdt]

        else:

            def f(y, t):
                h, v = y
                dvdt = sum((f.eval(h, v) for f in self.library.values()))
                return [v, dvdt]

        if y0 is None:
            if self.multiple_runs:
                y0 = [self.h[0][0], self.v0[0]]
            else:
                y0 = [self.h[0], self.v0]

        if t is None:
            t = np.arange(len(self.h))

        sol = odeint(f, y0, t)

        return sol

    def get_coefficients(self, numpy=False):
        """
        Get a list or array of learned coefficients for library functions.
        If multiple_runs = True, then this function returns the average
        coefficients across all trials.
        """
        coeffs = [f.coeff for f in self.library.values()]
        if numpy:
            return np.array(coeffs)
        else:
            return coeffs

    def get_labels(self):
        """
        Get a list of names of library functions
        """
        return list(self.library.keys())

    def get_xi(self, numpy=False):
        """
        Get the full set of library function coefficients
        """
        if numpy:
            return np.stack(self.Xi, axis=1)
        else:
            return self.Xi

    # Set coefficients of SINDy model (useful for group sparsity)
    def set_coefficients(self, Xi):
        """
        Set coefficients for library functions
        """
        if isinstance(Xi, list):
            Xi = np.stack(Xi, axis=1)

        # Otherwise, assume Xi is a numpy array
        for i, key in enumerate(self.library.keys()):
            self.library[key].coeff = Xi[i]

    def print_coefficients(self):
        """
        Print library coefficients next to library functions
        """
        coefficients = self.get_coefficients()
        labels = self.get_labels()
        df = DataFrame(data=coefficients, index=labels, columns=["Coefficients"])
        print(df, "\n")

    def print_equation(self):
        """
        Print model discovered by SINDy in human-readable way
        """
        coefficients = self.get_coefficients()
        labels = self.get_labels()
        nonzeros = np.nonzero(coefficients)[0]
        term_list = ["({:f}*{})".format(coefficients[i], labels[i]) for i in nonzeros]

        rhs = " + ".join(term_list)
        if len(rhs) == 0:
            rhs = "0"

        print(self.var_name, " = ", rhs, "\n")

        eqn = self.var_name + " = " + rhs
        return eqn

    def get_equation(self):
        """
        Return string of model discovered by SINDy in human-readable way
        """
        coefficients = self.get_coefficients()
        labels = self.get_labels()
        nonzeros = np.nonzero(coefficients)[0]
        term_list = [
            "({:3.3f}*{})".format(coefficients[i], labels[i]) for i in nonzeros
        ]

        term_list = ["({:f}*{})".format(coefficients[i], labels[i]) for i in nonzeros]

        rhs = " + ".join(term_list)
        if len(rhs) == 0:
            rhs = "0"
        return self.var_name + " = " + rhs

    def plot_derivatives(self, h, t=None, title=None):
        """
        Plot h and its approximate derivatives
        """
        if t is None:
            t = np.arange(len(h))

        if title is None:
            title_text = "Height and approximate derivatives"
        else:
            title_text = title

        v = differentiate(
            h,
            t,
            diff_method=self.diff_method,
            smoother=self.smoother,
            window_length=self.window_length,
        )
        dv = differentiate(
            v,
            t,
            diff_method=self.diff_method,
            smoother=self.smoother,
            window_length=self.window_length,
        )
        h_smoothed = smooth_data(
            h, smoother=self.smoother, window_length=self.window_length
        )

        plt.plot(t, h, label="h")
        plt.plot(t, h_smoothed, "--", label="h smoothed")
        plt.plot(t, v, label="v")
        plt.plot(t, dv, label="v'")
        plt.title(title_text)
        plt.xlabel("Time (s)")
        plt.legend()
        plt.show()


def group_sparse_penalty(Xi, method="1-norm", quantile=None):
    """
    Function for measuring group sparsity.
    Groups are assumed to be rows of Xi
    """
    # Place each coefficient vector as column in new matrix
    Xi = np.stack(Xi, axis=1)

    # 2-norm
    if method == "2-norm":
        return np.sqrt(np.sum(Xi ** 2, axis=1))

    # Average magnitude
    elif method == "avg":
        return np.mean(np.abs(Xi), axis=1)

    # Quantile
    #   quantile = 0: negative infinity-norm (min)
    #   quantile = 1: infinity-norm (max)
    #   quantile = 0.5: median magnitude
    elif method == "quantile":
        if quantile is None:
            quantile = 0.5
        return np.quantile(np.abs(Xi), quantile, axis=1)

    # Threshold
    #   quantile acts as threshold parameter
    elif method == "threshold":
        if quantile is None:
            quantile = 0.1
        return np.sum(np.abs(Xi) > quantile, axis=1)

    # 1-norm
    else:
        return np.sum(np.abs(Xi), axis=1)
