import numpy as np
import tensorflow as tf
from tensorflow import keras

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<1e-4):
            print("\nTarget telah dicapai, berhenti training !!!")
            self.model.stop_training = True

def solution_A1():
    X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0], dtype=float)

    # YOUR ALTERNATIVE CODE HERE
    model = keras.Sequential([
        keras.layers.Dense(units=8, activation='relu', input_shape=[1]),
        keras.layers.Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    callback = myCallback()
    
    model.fit(X, Y, epochs=1000, callbacks=[callback])

    print(model.predict([-2.0, 10.0]))
    return model

if __name__ == '__main__':
    model = solution_A1()
    model.save("model_A1.h5")
