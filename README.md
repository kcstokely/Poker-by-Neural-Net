# Poker-by-Neural-Net

Creates and trains a feed-forward neural-net to play video poker at optimal strategy.

Files:

poker_functions_.py :

    This can be used to create a training set of (hand, strategy) pairs, where:
  
    'Hand' looks like [1, 32, 14, 8, 45], giving the index of each card in the deck, and
  
    'Strategy' looks like [(0, 1, 1, 0, 0), (0, 1, 1, 1, 0)], with each entry a length-5 binary vector
    
    representing whether to hold / redeal each card in 'hand' (there may be multiple optimal strategies).
  
poker_keras_.py:

    Creates and trains the network on the data created above.
    
    This was originally done in Theano, but wow is Keras simpler for standard architectures.
  
    All the magic happens in the model definiton: one can play with the layer sizes and network depth.

poker_data_.dat:

    200 000 of the 2 600 000 hands (possible repeats), with correct strategies.
  
  
  
  
  
  
  
  
  
  
  
