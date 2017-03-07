# Poker-by-Neural-Net

Creates and trains a feed-forward neural-net to play video poker at optimal strategy.

poker_functions_.py :

  This can be used to create a training set of (hand, strategy) pairs.
  
  Hand looks like [1, 32, 14, 8, 45], giving the index of each card in the deck.
  
  Strategy looks like [(0, 1, 1, 0, 0), (0, 1, 1, 1, 0)], a list with each entry giving whether to hold or not.
  
  
 poker_keras_.py:
 
  Creates and trains the network on the data created above.
  
  All the magic happens in the model definiton: one can play with the layer sizes and network depth.
  
  
  
  
  
