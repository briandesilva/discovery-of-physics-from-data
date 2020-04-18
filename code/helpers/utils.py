"""
A collection of utility functions used either by other provided functions
or directly in the primary figure-generating code.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from .differentiation import smooth_data


def normalize_columns(X):
    """Normalize columns of a matrix"""
    X_col_norms = np.sqrt(np.sum(X ** 2, axis=0))
    return X / X_col_norms[np.newaxis, :], X_col_norms


def need_to_clip(t):
    """
    Check if we need to clip times from the end.

    Sometimes the measurement times differ significantly near the end of the
    drop, messing up the finite difference methods.
    """
    if np.isscalar(t):
        return 0

    t_diff = t[1:] - t[:-1]
    m = np.median(t_diff)
    inds = np.nonzero(np.abs(t_diff - m) > 5.0e-3)[0]

    if len(inds) > 0:
        max_ind = np.max(inds) + 1
        return max_ind
    else:
        return 0


def synthetic_ball_drop(
    accel, drag, h0=47.33, timesteps=49, dt=1 / 15, noise=0, v2=None
):
    """
    Generate synthetic ball drop data (initial velocity = 0).

    Solution is h = -t(accel/drag) + h0 + (accel/drag^2)(exp(drag*t) - 1)
    """

    t = np.arange(timesteps) * dt

    if drag == 0:
        h = h0 + (accel / 2) * t ** 2
    elif v2:

        def f(y, t):
            h, v = y
            dvdt = accel + drag * v + v2 * v ** 2
            return [v, dvdt]

        h = odeint(f, [h0, 0], t)[:, 0]
    else:
        const = accel / drag ** 2
        h = -const * drag * t + h0 + const * (np.exp(drag * t) - 1)

    h += noise * np.random.randn(timesteps)

    return h, t


def reynolds_number(velocity, diameter, k_viscosity=2 / 3):
    return 1e5 * velocity * diameter / k_viscosity


def approx_drag_coeff(re):
    if re:
        return (24 / re) * (1 + 0.15 * np.power(re, 0.681)) + 0.407 / (1 + 8710 / re)

    else:
        return 0


def re_dependent_synthetic_ball_drop(
    diameter,
    accel=-9.8,
    air_density=1.211,  # at 65 F, sea level
    mass=1,
    h0=47.33,
    timesteps=49,
    dt=1 / 15,
    noise=0,
):
    """
    Simulate a falling ball using a Reynolds number-dependent drag coefficient.
    """
    t = np.arange(timesteps) * dt

    cross_sectional_area = np.pi * (diameter / 2) ** 2
    const = air_density * cross_sectional_area / (2 * mass)

    def f(y, t):
        h, v = y
        re = -reynolds_number(v, diameter)
        dvdt = accel + const * approx_drag_coeff(re) * v ** 2
        return [v, dvdt]

    h = odeint(f, [h0, 0], t)[:, 0]
    h += noise * np.random.randn(timesteps)

    return h, t


def plot_prediction(
    h,
    predicted_hs,
    t=None,
    axs=None,
    compare="h",
    ball=None,
    eqns=None,
    figsize=None,
    smoother="savgol",
    window_length=35,
    h_plot="h",
    t_long=None,
    drop_flag=None,
):
    """
    Plot the true and predicted ball heights, and the difference
    between the two as functions of time.

    Generates two plots.

    Parameters
    ----------
    h : array_like
        true ball heights at each time point
    predicted_hs : array_like
        entries are lists of predicted heights at each time point
    t : array_like, optional
        time points corresponding to true ball heights
    axs : array_like, optional
        axes on which to plot the ball heights and error
    compare : string, optional
        Either 'h' or 'h_smoothed'; which version of ball height to
        compare predictions against
    ball : string, optional
        Ball name; used for title
    eqns : array_like, optional
        List of strings to use as labels for the entries of predicted_hs
    figsize : tuple, optional
        Size of the figure generated if no axes are passed in
    smoother : string, optional
        Smoother to apply when computing smoothed version of height
    window_length : integer, optional
        Length of smoothing window used to smooth height
    h_plot : string, optional
        Either 'h' or 'h_smoothed'; which version of true ball height to plot
    t_long : array_like, optional
        Extended list of time points (extended beyond t) corresponding to the
        heights in the entries of predicted_hs. h will only be plotted against
        t, but entries of predicted_hs will be plotted against t_long
    drop_flag : array_like, optional
        Length 2 array_like allowing for a model's predictions to be omitted
        from the plots after a specified time.
        The first entry should give the index corresponding to the model's
        predictions in predicted_hs and the second entry should give the time
        after which the predictions are omitted.
    """

    if t is None:
        t = np.arange(len(h))

    # Generate figure if no axes passed in
    if axs is None:
        if figsize is None:
            figsize = (8, 3)
        fig, axs = plt.subplots(1, 2, figsize=figsize)

    if ball is None:
        ball = ""

    title = str(ball)
    plot_styles = ["--"]

    if eqns is None:
        eqns = [""] * len(predicted_hs)
    elif isinstance(eqns, str):
        eqns = [eqns]

    h_smoothed = smooth_data(h, smoother=smoother, window_length=window_length)

    if t_long is None:
        # Provide option to plot smoothed height
        if h_plot == "h":
            axs[0].plot(t, h, label="Observed", linewidth=3.5)
        else:
            axs[0].plot(t, h_smoothed, label="Observed", linewidth=3.5)

        axs[1].plot(t, np.abs(h - h_smoothed), label="Smoothed height")

        for k, predicted_h in enumerate(predicted_hs):
            axs[0].plot(
                t, predicted_h, plot_styles[np.mod(k, len(plot_styles))], label=eqns[k],
            )

            # Detect larger errors and use log-scale if necessary
            if compare == "h smoothed":
                err = np.abs(h_smoothed - predicted_h)
            else:
                err = np.abs(h - predicted_h)

            axs[1].plot(t, err, plot_styles[np.mod(k, len(plot_styles))], label=eqns[k])
            if np.max(err) > 15:
                axs[1].set(yscale="log")

        axs[0].set(ylabel="Height (m)", title=title)
        axs[0].legend()
        axs[1].set(xlabel="Time (s)", ylabel="Error (m)", title="Error")

        # Fix ticks
        axs[0].set_xticks(np.arange(int(t[-1]) + 1))
        axs[1].set_xticks(np.arange(int(t[-1]) + 1))

    else:
        if h_plot == "h":
            axs.plot(t, h, label="Observed", linewidth=3.5)
        else:
            axs.plot(t, h_smoothed, label="Observed", linewidth=3.5)

        model = -1
        if drop_flag:
            model, t_end = drop_flag[0], drop_flag[1]
            inds_to_nan = t_long > t_end
        for k, predicted_h in enumerate(predicted_hs):
            if k == model:
                predicted_h[inds_to_nan] = np.nan
            axs.plot(
                t_long,
                predicted_h,
                plot_styles[np.mod(k, len(plot_styles))],
                label=eqns[k],
                linewidth=2,
            )

        axs.set(xlabel="Time (s)", ylabel="Height (m)", title=title)
        axs.legend()

        # Fix ticks
        axs.set_xticks(np.arange(int(t_long[-1]) + 1))


def relative_error(u, u_approx, ord=None):
    return np.linalg.norm(u - u_approx, ord=ord) / np.linalg.norm(u, ord=ord)


def resize_fonts(ax, title=20, xaxis=15, yaxis=15, ticks=None):
    """
    Resize fonts for title, x-axis, y-axis, and ticks of a given axis.
    """
    if isinstance(ax, (list, np.ndarray)):
        for a in ax:
            a.title.set_fontsize(title)
            a.xaxis.label.set_fontsize(xaxis)
            a.yaxis.label.set_fontsize(yaxis)
            if ticks:
                for i in a.get_xticklabels() + a.get_yticklabels():
                    i.set_fontsize(ticks)

    else:
        ax.title.set_fontsize(title)
        ax.xaxis.label.set_fontsize(xaxis)
        ax.yaxis.label.set_fontsize(yaxis)
        for i in ax.get_xticklabels() + ax.get_yticklabels():
            i.set_fontsize(ticks)
