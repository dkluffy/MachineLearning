import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

create_time_steps = lambda x: [i for i in range(-1*x,0,1)]


def show_plot(plot_data, delta, title):
    """
    plot_data : []
    """

    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
      future = delta
    else:
      future = 0       
    plt.title(title)
    for i, _ in enumerate(plot_data):
      #神特么诡异的东西，if i 等价于 if i <>0 ...
      if i:
        plt.plot(future, plot_data[i], marker[i], markersize=10,
                 label=labels[i])
      else:
        plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt


def multi_step_plot(history, true_future, prediction,step):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)        
    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/step, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
      plt.plot(np.arange(num_out)/step, np.array(prediction), 'ro',
               label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()


def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']    
    epochs = range(len(loss))    
    plt.figure()    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()    
    plt.show()