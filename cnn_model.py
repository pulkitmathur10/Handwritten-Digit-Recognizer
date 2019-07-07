# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np

def cnn_predict(imge):
    x = 0
    model = load_model('model.h5')
    guess = model.predict(imge)
    guess = np.argmax(guess, axis=-1)
    x= guess[0]
    x=int(x)
    num = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    for i in range(len(num)):
        if(i==x):
            num_alpha = str(x) + ' (' + num[i] + ')'
    return num_alpha