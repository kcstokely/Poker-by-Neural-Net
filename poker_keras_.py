### IMPORTS ###############################################################################################################################

import os
import ast
import csv

import keras
import numpy

###########################################################################################################################################

if __name__ == '__main__':

    try:
        os.makedirs('./params/')
    except:
        pass
    
    ###
    ###   DEFINE MODEL
    ###

    try:

        with(open('./params/poker_keras_model_.json', 'r')) as fp:
            p_json = fp.read()
        p_model = keras.models.model_from_json(p_json)
        p_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001))
        p_model.load_weights('./params/poker_keras_weights_.h5')
        print '# model loaded'
    
    except:
        
        p_model = keras.models.Sequential()
        p_model.add(keras.layers.Dense(output_dim = 50, input_dim = 52 , activation = 'sigmoid'))
        p_model.add(keras.layers.Dense(output_dim = 50, activation = 'sigmoid'))
        p_model.add(keras.layers.Dense(output_dim = 50, activation = 'sigmoid'))
        p_model.add(keras.layers.Dense(output_dim = 32, activation = 'sigmoid'))
        p_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.001))
        with open('./params/poker_keras_model_.json', 'w') as fp:
            fp.write(p_model.to_json())
        print '# model created'
        
    ###
    ###   LOAD DATA
    ###
    
    data_file = './poker_data_.dat'
    
    data_x = []
    data_y = []
    
    with open(data_file, 'r') as fp:
        
        num_entries = 0
        
        csr = csv.reader(fp, delimiter='\t')
        for row in csr:
            
            num_entries += 1
            
            hand = ast.literal_eval(row[0])
            vect = [ (i in hand) for i in range(52) ]
            data_x.append(vect)
            
            strat = ast.literal_eval(row[1])
            first = strat[0]
            f_num = [ first[i] * 2**i for i in range(5) ]
            f_sum = sum(f_num)
            f_vec = [ int(i==f_sum) for i in range(32) ]
            data_y.append(f_vec)

    data_x, data_y = numpy.asarray(data_x), numpy.asarray(data_y)
    
    print '# data loaded'        
        
    ###
    ###   TRAIN MODEL
    ###
    
    num_epochs = 200

    batch_size = 25
    
    patience = 5
    
    num_batches = num_entries // batch_size
    
    try:
        with open('./params/poker_keras_epochs_.dat', 'r') as fp:
            start = int(fp.readline())
    except:
        start = 0
            
    p_model.fit(
        data_x,
        data_y,
        validation_split = 0.15,
        batch_size = batch_size,
        nb_epoch = num_epochs + start,
        initial_epoch = start,    
        verbose = 2,
        callbacks = [keras.callbacks.EarlyStopping(patience = patience)]
    )
    #########
    #########   SAVE MODEL WEIGHTS
    #########
    
    try:
        os.rename('./params/poker_keras_epochs_.dat', './params/poker_keras_epochs_.dat.bak')
        os.rename('./params/poker_keras_weights_.h5', './params/poker_keras_weights_.h5.bak')
    except:
        pass
    
    with open('./params/poker_keras_epochs_.dat', 'w') as fp:
        fp.write(str(start + num_epochs))
    p_model.save_weights('./params/poker_keras_weights_.h5')
    
###########################################################################################################################################

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    