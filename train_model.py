import tensorflow as tf
import numpy as np
import pandas as pd


def train_model(model,data,model_name):
    Y=data['misstate'].values
    X=data.drop(columns=['misstate']).values
    history=model.fit(X, Y,epochs = 5,shuffle=True,verbose=0)
    model.save('trained_models/'+model_name+'.h5')
    print("Average test loss: ", np.average(history.history['loss']))    
    print("Saved Model")