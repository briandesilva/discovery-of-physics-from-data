from numpy import ceil, count_nonzero
from scipy.integrate import odeint
from pandas import DataFrame
import matplotlib.pyplot as plt

from .sindy_ball import SINDyBall
from .differentiation import differentiate
from .utils import need_to_clip, plot_prediction


def test_group_sparsity(
    bd_data,
    thresh=0.1,
    drops=[3],
    plot_drop=None,
    smoother="savgol",
    window_length=35,
    diff_method="centered difference",
    library_type="polynomial",
    plot_error=False,
    normalize=False,
    group_sparse_method="1-norm",
    coefficient_dict=None,
    model_to_subtract=None,
    h_plot="h smoothed",
    all_coefficient_dict=None,
    n_terms=None,
    train_inds=[],
    test_inds=[],
):
    """Perform various tests using the group sparsity version of SINDy"""

    n_balls = len(bd_data)

    h = []
    t = []

    diff_kws = {
        "diff_method": diff_method,
        "smoother": smoother,
        "window_length": window_length,
    }

    for ball in bd_data.keys():
        for drop in drops:
            ball_df = bd_data[ball].loc[bd_data[ball]["Drop #"] == drop]

            h_run = ball_df["Height (m)"].values
            t_run = ball_df["Time (s)"].values

            ntc = need_to_clip(t_run)
            if ntc:
                h_run = h_run[:ntc]
                t_run = t_run[:ntc]

            drag_coeff = None
            if train_inds:
                h_run = h_run[train_inds[0] : train_inds[1]]
                t_run = t_run[train_inds[0] : train_inds[1]]

            # Modify h by subtracting known model contribution
            if model_to_subtract:
                h_run = subtract_known_model(
                    h_run,
                    t_run,
                    model_to_subtract,
                    drag_coeff=drag_coeff,
                    diff_kws=diff_kws,
                )

            h.append(h_run)
            t.append(t_run)

    sb = SINDyBall(
        thresh=thresh,
        diff_method=diff_method,
        smoother=smoother,
        window_length=window_length,
        library_type=library_type,
    )

    sb.fit(
        h,
        t,
        normalize=normalize,
        group_sindy=True,
        group_sparse_method=group_sparse_method,
    )

    # Store or print average coefficients
    avg_coeffs = sb.get_coefficients(numpy=True)

    if n_terms is not None:
        n_terms.append(count_nonzero(avg_coeffs))

    if coefficient_dict:
        labels = sb.get_labels()
        for label, coeff in zip(labels, avg_coeffs):
            if label not in coefficient_dict:
                coefficient_dict[label] = []
            coefficient_dict[label].append(coeff)

    else:
        print("Average equation: ")
        sb.print_equation()

    # Get learned coefficients for all balls
    Xi = sb.get_xi()

    # Store all coefficients for all balls
    if all_coefficient_dict:
        labels = sb.get_labels()
        for k, ball in enumerate(bd_data.keys()):
            for label, coeff in zip(labels, Xi[k]):
                if label not in all_coefficient_dict[ball]:
                    all_coefficient_dict[ball][label] = []
                all_coefficient_dict[ball][label].append(coeff)

    # Plot error from predicting every trajectory with the average model
    if plot_error:
        if plot_drop is None:
            plot_drop = 3 if (drops[0] == 2) else 2

        n_rows = int(2 * ceil(n_balls / 4))
        fig, axs = plt.subplots(n_rows, 4, figsize=(16, n_rows * 3))

    error = {}
    for k, ball in enumerate(bd_data.keys()):
        ball_df = bd_data[ball].loc[bd_data[ball]["Drop #"] == plot_drop]

        h = ball_df["Height (m)"].values
        t = ball_df["Time (s)"].values

        ntc = need_to_clip(t)
        if ntc:
            h = h[:ntc]
            t = t[:ntc]

        drag_coeff = None
        if model_to_subtract:
            h = subtract_known_model(
                h, t, model_to_subtract, drag_coeff=drag_coeff, diff_kws=diff_kws,
            )

        v = differentiate(h, t, **diff_kws)

        ax_col = k % 4
        ax_row = 2 * (k // 4)

        if test_inds:
            t = t[test_inds[0] : test_inds[1]]
            h = h[test_inds[0] : test_inds[1]]
        init_cond = [h[0], v[0]]

        sb.set_coefficients(avg_coeffs)
        h_baseline_pred = -4.9 * t ** 2 + h[0]

        if len(drops) == 1:
            sb.set_coefficients(Xi[k])
        else:
            sb.set_coefficients(Xi[2 * k])
        h_tailored_pred = sb.predict(y0=init_cond, t=t)[:, 0]

        error[ball] = [
            h[-1] - h_tailored_pred[-1],
            h[-1] - h_baseline_pred[-1],
        ]

        if plot_error:
            learned_equation = sb.get_equation()
            plot_prediction(
                h,
                [h_baseline_pred, h_tailored_pred],
                axs=axs[ax_row : ax_row + 2, ax_col],
                t=t,
                ball=ball,
                eqns=["avg model", learned_equation],
                compare="h smoothed",
                h_plot=h_plot,
            )

    if plot_error:
        plt.tight_layout()
        plt.show()

    error = DataFrame.from_dict(error, orient="index", columns=["tailored", "baseline"])
    return error


def predict_with_model(h, t, model, diff_kws={}):
    """Use a model to simulate forward in time."""
    v = differentiate(h, t, **diff_kws)
    y0 = [h[0], v[0]]

    h_predicted = odeint(model, y0, t)[:, 0]
    return h_predicted


def subtract_known_model(h, t, model_type, drag_coeff=None, diff_kws={}):
    """Subtract the results of a model simulation from h."""
    if model_type == "constant":

        def model(y, t):
            return [y[1], -9.8]

    elif model_type == "drag":
        if drag_coeff:

            def model(y, t):
                return [y[1], -9.8 + drag_coeff * y[1]]

        else:

            def model(y, t):
                return [y[1], -9.8]

    else:
        raise ValueError(
            """
                model_type must be either 'constant' or 'drag'. {} was received
            """.format(
                model_type
            )
        )

    h -= predict_with_model(h, t, model, diff_kws=diff_kws)

    return h
