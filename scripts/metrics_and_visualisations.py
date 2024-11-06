"""Metrics and visualisations for hydrological models."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    """Calculate the Root Mean Squared Error (RMSE) for two given arrays of values.

    Parameters
    ----------
    y_true : array-like
        The true values.
    y_pred : array-like
        The predicted values.

    Returns
    -------
    rmse : float
        The Root Mean Squared Error between ``y_true`` and ``y_pred``.

    """
    return np.sqrt(np.nanmean((np.array(y_true) - np.array(y_pred)) ** 2))


def qh_plot(
    qh_df: pd.DataFrame,
    partition_df: pd.DataFrame = pd.DataFrame(),
    manning_df: pd.DataFrame = pd.DataFrame(),
):
    """Plot the discharge level relationship (Q(h)) from a given DataFrame.

    Parameters
    ----------
    qh_df : pd.DataFrame
        The DataFrame containing the discharge level relationship data.
    partition_df : pd.DataFrame, optional
        An optional DataFrame containing the discharge level relationship
        data from partitions. If not provided, no points will be plotted from
        partitions.
    manning_df : pd.DataFrame, optional
        An optional DataFrame containing the discharge level relationship
        data from Manning's equation. If not provided, no points will be
        plotted from Manning's.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the plot.

    """
    # Extract discharge and water level data from the main DataFrame
    x = qh_df["q_cms"].to_numpy()  # Discharge values in m3/s
    y = qh_df["lvl_sm"].to_numpy()  # Water levels in cm
    y_qh = qh_df["lvl_qh"].to_numpy()  # Water levels in cm, calculated from Q(h)

    # Create a new matplotlib figure and axis
    _, ax = plt.subplots()

    # Plot scatter points for discharge vs water level from the main DataFrame
    ax.scatter(x, y, s=5, label="Q(h) from AIS")  # Scatter points from AIS
    ax.scatter(
        x, y_qh, s=15, label="Q(h) from Q(h) relationship", color="green"
    )  # Scatter points from Q(h) relationship

    # Plot scatter points if partition DataFrame is not empty
    if not partition_df.empty:
        ax.scatter(
            partition_df["q_cms"].to_numpy(),
            partition_df["lvl_sm"].to_numpy(),
            s=20,
            label="Q(h) from partitions",
            color="red",
        )

    # Plot scatter points if Manning DataFrame is not empty
    if not manning_df.empty:
        ax.scatter(
            manning_df["q_manning"].to_numpy(),
            manning_df["lvl_sm"].to_numpy(),
            s=20,
            label="Q(h) from Mannings",
            color="black",
        )

    # Set axis labels
    ax.set_xlabel("Discharge, m3/s")  # Discharge units
    ax.set_ylabel("Water level, cm")  # Water level units

    # Add a legend to the plot
    ax.legend()

    # Close the plot to prevent display in non-GUI environments
    plt.close()

    # Return the figure object
    return ax.get_figure()
