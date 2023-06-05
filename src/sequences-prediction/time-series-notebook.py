"""
Reference link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W1/ungraded_labs/C4_W1_Lab_1_time_series.ipynb#scrollTo=w_XJiOdr-MSM

This notebook aims to show different terminologies and attributes of a time series
by generating and plotting synthetic data.

Trying out different prediction models on this kind of data is a good way to
develop your intuition when you get hands-on with real-world data later in the course.
 Let's begin!

"""

import matplotlib.pyplot as plt
import numpy as np

def plot_series(time, series, format="-", start=0, end=None, label=None):
    """
    Visualize time series data
    Args
    :param time: (array of int) - contains the time steps 
    :param series: (array of int) - contains the measurements for each time step
    :param format: (string) - line style when plotting the graph
    :param start: (int) - first time step to plot
    :param end: (int) - ast time step to plot
    :param label: (list of strings)- tag for the line
    :return: 
    """

    # Setup dimensions of the graph figure
    plt.figure(figsize=(10, 6))

    # Plot the time series data
    plt.plot(time[start:end], series[start:end], format)

    # Label the x-axis
    plt.xlabel("Time")

    # Label the y-axis
    plt.ylabel("Value")

    if label:
        plt.legend(fontsize=14, labels=label)

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()




def trend(time, slope=0):
    """
    Generates synthetic data that follows a straight line given a slope value.

    Args:
      time (array of int) - contains the time steps
      slope (float) - determines the direction and steepness of the line

    Returns:
      series (array of float) - measurements that follow a straight line
    """

    # Compute the linear series given the slope
    trend_series = slope * time

    return trend_series



# Generate time steps. Assume 1 per day for one year (365 days)
time = np.arange(365)

# Define the slope (You can revise this)
# When updating this from .1 to 1000, they produce the same graph,
# but the Values change depending on the slope, this shows that no matter the slope
# value, the graph plots the same graph, which is a constant increase per day,
# it could be that you get paid $1, or $100 dollars per day, the graph will be same,
# as you win the same amount of money per day.
slope = 1

# Generate measurements with the defined slope
series = trend(time, slope)

# Plot the results
plot_series(time, series, label=[f'slope={slope}'])