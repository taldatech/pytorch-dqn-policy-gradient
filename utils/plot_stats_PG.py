import numpy as np
import matplotlib.pyplot as plt
import pickle


def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_traces(data, title, legends):
    """
    each row is considered a data trace to be plotted
    :param data: 2d np array
    :param title: title of the plot
    :param legends: legends of the traces
    :return: None
    """

    plt.figure()

    for trace in range(data.shape[0]):
        plt.plot(data[trace, :])

    plt.grid()
    plt.legend(legends)
    plt.title(title)
    plt.show()


rewards_128 = pickle.load(open(r'.\data\rewards_HL128_trained50000.pt', 'rb'))
rewards_1024 = pickle.load(open(r'.\data\rewards_HL1024_trained50000.pt', 'rb'))

plot_traces(np.array([moving_average(rewards_128), moving_average(rewards_1024)]), 'average rewards on 100 episodes', ['128', '1024'])

losses_128 = np.abs(pickle.load(open(r'.\data\losses_HL128_trained100000.pt', 'rb')))
losses_1024 = np.abs(pickle.load(open(r'.\data\losses_HL1024_trained100000.pt', 'rb')))

plot_traces(np.array([moving_average(losses_128), moving_average(losses_1024)]), 'average policy loss on 100 episodes', ['128', '1024'])

episode_128 = pickle.load(open(r'.\data\episode_lens_HL128_trained100000.pt', 'rb'))
episode_1024 = pickle.load(open(r'.\data\episode_lens_HL1024_trained100000.pt', 'rb'))

plot_traces(np.array([moving_average(episode_128), moving_average(episode_1024)]), 'average episode length on 100 episodes', ['128', '1024'])
