### IMPORTS ###############################################################################################################################

import os
import ast
import csv

import keras
import numpy

import matplotlib.pyplot as plt

import poker_functions_  as pf 

###########################################################################################################################################

if __name__ == '__main__':

    ###
    ###   SPECIFICS
    ###  
    
    data_file = './data/poker_data_.dat'
    
    model_dir = './models/m_dense_001_/'
    
    ###
    ###   DEFINE MODEL
    ###

    try:
        os.makedirs(model_dir)
    except:
        pass
    
    try:

        with(open(model_dir+'model_.json', 'r')) as fp:
            p_model = keras.models.model_from_json(fp.read())
        print '# model loaded'
    
    except:
        
        p_model = keras.models.Sequential()
        
        p_model.add(keras.layers.Dense(1032,
                                       input_dim = 52,
                                       activation = 'sigmoid'
                                       #activity_regularizer=keras.regularizers.l2(0.001)
                ))
        
        p_model.add(keras.layers.Dense(1032,
                                       activation = 'sigmoid'
                                       #activity_regularizer=keras.regularizers.l2(0.001)
                ))
        
        p_model.add(keras.layers.core.Dropout(0.5))
        
        p_model.add(keras.layers.Dense(32,
                                       #activation = 'sigmoid'
                                       activation = 'softmax'
                                       #activity_regularizer=keras.regularizers.l2(0.001)
                ))

        with open(model_dir+'model_.json', 'w') as fp:
            fp.write(p_model.to_json())
        print '# model created'
    
    #p_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001))
    p_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001))
    print '# model compiled'
    
    try:
        p_model.load_weights(model_dir+'weights_.h5')
    except:
        pass
        
    ###
    ###   LOAD DATA
    ###
    
    data_x = []
    data_y = []
    
    with open(data_file, 'r') as fp:
        
        csr = csv.reader(fp, delimiter='\t')
        for row in csr:
            
            hand   = ast.literal_eval(row[0])
            vect   = [ (i in hand) for i in range(52) ]
            strats = ast.literal_eval(row[1])
            for strat in strats:
                
                f_num = [ strat[i] * 2**i for i in range(5) ]
                f_sum = sum(f_num)
                f_vec = [ int(i==f_sum) for i in range(32) ]

                data_x.append(vect)
                data_y.append(f_vec)
                #data_y.append(strat)

    data_x, data_y = numpy.asarray(data_x), numpy.asarray(data_y)
    
    print '# data loaded'        
        
    ###
    ###   TRAIN MODEL
    ###
    
    max_epochs = 200

    batch_size = 25
    
    patience   = 5
    
    try:
        with open(model_dir+'epochs_.dat', 'r') as fp:
            start = int(fp.readline())
    except:
        start = 0
    
    fit_logs = p_model.fit(
                data_x,
                data_y,
                validation_split = 0.15,
                batch_size = batch_size,
                nb_epoch = max_epochs,
                initial_epoch = start,    
                verbose = 1,
                callbacks = [keras.callbacks.EarlyStopping(patience = patience)]
               )
    
    try:
        tp = fit_logs.history['loss']    
        vp = fit_logs.history['val_loss']
    except:
        pass
        
    #plt.plot(range(len(tp)), tp, 'bo', range(len(vp)), vp, 'ro')
    #plt.show()
    
    #########
    #########   SAVE MODEL WEIGHTS
    #########

    try:
        os.rename(model_dir+'epochs_.dat', model_dir+'epochs_.dat.bak')
        os.rename(model_dir+'weights_.h5', model_dir+'weights_.h5.bak')
    except:
        pass
    
    with open(model_dir+'epochs_.dat', 'w') as fp:
        fp.write(str(start + max_epochs))
    p_model.save_weights(model_dir+'weights_.h5')    
        
    #########
    #########   PLAY
    #########
    
    num_hands = 10
    
    #########
    
    print '# playing poker...', '\n'
    
    score = 0
    deal  = pf.pHand()
    
    for n in range(num_hands):
        
        deal.deal()
        print deal.human(),
        
        x = numpy.asarray(deal.vector()).reshape((1,52))
        y = p_model.predict(x)
        
        pred = numpy.argmax(y)
        hold = [ int( (pred//(2**i)) % 2 == 1) for i in range(5) ]

        deal.discard(hold)
        deal.refill()

        sc     = (pf.scores[pf.get_score(deal)] - 1)
        score += sc
        
        print hold, sc, score
    
    try:
        plt.plot(range(len(tp)), tp, 'bo', range(len(vp)), vp, 'ro')
        plt.show()
    except:
        pass
    
###########################################################################################################################################

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    