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


def noise(time, noise_level=1, seed=None):
    """Generates a normally distributed noisy signal

    Args:
      time (array of int) - contains the time steps
      noise_level (float) - scaling factor for the generated signal
      seed (int) - number generator seed for repeatability

    Returns:
      noise (array of float) - the noisy signal

    """

    # Initialize the random number generator
    rnd = np.random.RandomState(seed)

    # Generate a random number for each time step and scale by the noise level
    noise = rnd.randn(len(time)) * noise_level

    return noise


# Define noise level
noise_level = 5

# Generate time steps. Assume 1 per day for one year (365 days)
time = np.arange(365)


# Generate noisy signal
noise_signal = noise(time, noise_level=noise_level, seed=42)

# Plot the results
plot_series(time, noise_signal)
