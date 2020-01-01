import tensorflow as tf
from tensorflow import keras
from datetime import datetime

def train_with_checkpoin_tensorboard(model,**kwargs):
    try:
        num_epoches = kwargs["epoches"]
               
        y_train = kwargs["y_train"]
        val_data   = kwargs["val_data"] # (X_val,y_val)
        steps_per_epoch = kwargs["steps_per_epoch"] #len(X)//batch_size
        validation_steps = kwargs["validation_steps"]

        run_logdir = kwargs["run_logdir"]
        model_save_file = kwargs["model_save_file"]
        is_generator = kwargs["is_generator"]
        device = kwargs["device"]
        
    except:
        #init
        num_epoches = 1000

        y_train = None
        val_data   = None
        steps_per_epoch = 100
        validation_steps = 100

        run_logdir = "tb_logs"
        model_save_file = "model_%s.h5" % datetime.now().isoformat(timespec='minutes')
        is_generator = True
        device = "/GPU:0"
    
    #
    # input
    X_train = kwargs["X_train"]
    
    #callbacks
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    checkpoint_cb = keras.callbacks.ModelCheckpoint(filepath=model_save_file, verbose=1)

    if not is_generator:
        with tf.device(device):
            history = model.fit_generator(X_train,
                steps_per_epoch=steps_per_epoch,
                epochs=num_epoches,
                validation_data=val_data,
                validation_steps=validation_steps,
                callbacks=[tensorboard_cb,checkpoint_cb])
    else:
        with tf.device(device):
            history = model.fit(X_train,y_train,
                steps_per_epoch=steps_per_epoch,
                epochs=num_epoches,
                validation_data=val_data,
                validation_steps=validation_steps,
                callbacks=[tensorboard_cb,checkpoint_cb])

    # -- save --
    model.save(model_save_file)


    #-- visualize --
    import matplotlib.pyplot as plt
    
    #import matplotlib as mpl
    #mpl.rcParams['figure.figsize'] = (8, 6)
    #mpl.rcParams['axes.grid'] = False
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(num_epoches)
    
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

# class LossHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.losses = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))