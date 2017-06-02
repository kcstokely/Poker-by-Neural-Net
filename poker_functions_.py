### IMPORTS ###############################################################################################################################

import ast
import csv
import sys
import copy
import numpy
import scipy
import pickle
import random
import datetime
import itertools

from scipy import special
    
### HELPER FUNCTION #######################################################################################################################

def compress(inlist):
    m = sorted(list(set(inlist)))
    n = [ m.index(i) for i in inlist ]
    return n
    
### POKER HAND CLASS ######################################################################################################################
###
###   Useful for converting between various representations.
###

class pHand(object):
    
    nums = ['A','2','3','4','5','6','7','8','9','T','J','Q','K']
    pics = ['D','C','H','S']
    
    def __init__(self):    
        self.deck = range(52)
        self.pile = range(52)
        self.hand = []
    
    def deal(self):
        self.pile = range(52)
        self.hand = random.sample(self.pile, 5)
        for c in self.hand:
            self.pile.remove(c)
            
    def discard(self, hold):
        self.hand.sort()
        for j in range(4,-1,-1):
            if not hold[j]:
                del self.hand[j]
                
    def refill(self):
        ext = random.sample(self.pile, 5-len(self.hand))
        self.hand.extend(ext)
        for c in ext:
            self.pile.remove(c)
        
    ### in/out
    
    def read(self, hand):
        self.hand = hand
        self.pile = range(52)
        for c in self.hand:
            self.pile.remove(c)

    def readcs(self, cards, suits):
        self.hand = [ cards[i] + suits[i]*13 for i in range(5) ]
        self.pile = range(52)
        for c in self.hand:
            self.pile.remove(c)
        return numpy.argsort(self.hand)

    def readbs(self, bysuit):
        self.hand = [ i*13+j for i,m in enumerate(bysuit) for j in m ]
        self.pile = range(52)
        for c in self.hand:
            self.pile.remove(c)

    def readh(self, cards, suits):
        cs = [ nums.index(i) for i in cards ]
        ss = [ pics.index(i) for i in suits ]
        self.readcs(self, cs, ss)
    
    def readv(self, vector):
        self.hand = [ i for i in range(52) if vector[i]==1 ]
        self.pile = range(52)
        for c in self.hand:
            self.pile.remove(c)
            
    def cards(self):
        return [ c%13 for c in self.hand ]

    def suits(self):
        return [ c//13 for c in self.hand ]
        
    def bysuit(self):
        return [ sorted([ cards[j] for j in range(5) if suits[j]==i ]) for i in range(4) ]

    def human(self):
        self.hand.sort()
        return [ self.pics[self.suits()[i]]+":"+self.nums[self.cards()[i]] for i in range(len(self.hand)) ]

    def vector(self):
        return [ int(i in self.hand) for i in range(52) ]
    
### SCORES LIST ############################################################################################################################

scores = {}
scores['nada']     = 0
scores['pair']     = 0
scores['jack']     = 1
scores['twop']     = 2
scores['trip']     = 3
scores['straight'] = 4
scores['flush']    = 6
scores['full']     = 9
scores['four']     = 25
scores['strflush'] = 50
scores['royal']    = 800

### SCORING FUNCTION ######################################################################################################################
###
###   Takes a pHand instance with len(hand) == 5.
###
###   Returns a string giving the type of hand.
###

def get_score(hand):
    ### hello
    assert len(hand.hand)==5
    cards = hand.cards()
    suits = hand.suits()
    ### check similars
    for i in range(4):
        remain = list(cards)
        remain.pop(i)
        if cards[i] in remain:
            # then we have a pair
            remain.remove(cards[i])
            if cards[i] in remain:
                #then we have a triple
                remain.remove(cards[i])
                if remain[0] == remain[1]:
                    return 'full'
                elif cards[i] in remain:
                    return 'four'
                else:
                    return 'trip'
            elif remain[0]==remain[1] or remain[0]==remain[2] or remain[1]==remain[2]:
                if remain[0]==remain[1] and remain[0]==remain[2]:
                    return 'full'
                else:
                    return 'twop'
            elif cards[i]==0 or cards[i]>9:
                return 'jack'
            else:
                return 'pair'
    ### check flush
    flush = False
    if suits[:-1] == suits[1:]:
        flush = True
    ### check straight
    cards.sort()
    straight = 0
    if cards == [0, 9, 10, 11, 12]: 
        if not flush:
            return 'straight'
        else:
            return 'royal'
    if cards[1] == cards[0]+1:
        if cards[2] == cards[1]+1:
            if cards[3] == cards[2]+1:
                if cards[4] == cards[3]+1:
                    if not flush:
                        return 'straight'
                    else:
                        return 'strflush'                
    ### return
    if not flush:
        return 'nada'
    else:
        return 'flush'
    
### SCORE DICTIONARY ######################################################################################################################
###
###   Creates or loads the score dictionary.
###
###      eg: score['[0, 1, 2, 3, 4]'] = 'strflush'
###
###      contains all (52 choose 5) entries --- no equivalence classes b/c less expensive to store
###
###      out of order hands will need to be sorted before lookup --- (52 perm 5) is too big
###

def load_score_dict():
    try:
        score = pickle.load(open('score_num_dictionary_.pkl','rb'))
    except(OSError, IOError):
        hist = {}
        for key in scores:
            hist[key] = 0
        score = {}  
        hand = pHand()
        for i, h in enumerate(itertools.combinations(range(52), 5)):
            hand.read(list(h))
            s = get_score(hand)
            hist[s] += 1
            score[str(hand.hand)] = scores[s]
        print 'made'
        pickle.dump(hist, open('score_hist_.pkl', 'wb'))
        pickle.dump(score, open('score_num_dictionary_.pkl', 'wb'))
        print 'dumped'
    else:    
        print 'loaded'
    return score
        
### EXPECTED RETURN OF A STRATEGY #########################################################################################################
###
###   Takes a freshly dealt pHand instance, and a strategy in [0,1]^5.
###
###   Returns a float; input is left discared after call.
###

def expected_return(deal, hold):
    # deal should have len(hand)==5 and len(pile)==47
    deal.discard(hold)
    # now deal has 0<=len(hand)<=5 and len(pile)==47
    to_go = 5-len(deal.hand)
    # split cases
    if to_go == 5:
        return 0.339 # (expectation value) pre-computed 
    else:
        count = 0
        normal = scipy.special.binom(47, to_go)
        for i in itertools.combinations(deal.pile, to_go):
            deal.hand.extend(list(i))
            count += score[str(sorted(deal.hand))]
            del deal.hand[-to_go:]
        return count/normal
    # when we return, 'deal' is discarded according to 'hold'

### BEST STRATEGY FOR A DEALT HAND ########################################################################################################
###
###   Takes a freshly gealt pHand instance.
###

def best_strategy(deal):
    epsilon = 0.01
    best_score = 0
    best_strat = []
    for h in itertools.product(range(2), repeat=5):
        d = copy.deepcopy(deal)
        s = expected_return(d, h)
        if s > best_score - epsilon:
            if s > best_score + epsilon:
                best_score = s
                best_strat = [h]
            else: 
                best_strat.append(h)
    return [best_score, best_strat]
    
### CREATE TRAINING SET ###################################################################################################################

def create_random_training_set(filename, number, verbose):
    with open(filename, 'a') as fp:
        csw = csv.writer(fp, delimiter='\t')
        deal = pHand()
        for i in range(number):
            deal.deal()
            bs = best_strategy(deal)
            csw.writerow([deal.hand, bs[1]])
            fp.flush()
            if verbose == True:
                print i, '\t', deal.human(), '\t', bs                    
                    
### BOOST TRAINING SET ####################################################################################################################
###
###   Takes a training set and outputs a larger set.
###
###   For each original entry, the output set comtains all equivalent hands,
###      under permutation of suits.
###

def boost_training_set(file_in, file_out):
    hand = pHand()
    newh = pHand()
    with open(file_out, 'w') as wp:
        csw = csv.writer(wp, delimiter='\t')
        with open(file_in, 'r') as fp:
            csr = csv.reader(fp, delimiter='\t')
            for row in csr:
                h = ast.literal_eval(row[0])
                s = ast.literal_eval(row[1])
                hand.read(h)
                cards = hand.cards()
                suits = hand.suits()
                compr = compress(suits)
                used  = [] # this method is ok because used will be max 24 long
                for i in itertools.permutations(range(4), len(set(compr))):
                    news = [ i[j] for j in compr ]
                    newh.readcs(cards, news)
                    newv = newh.vector()
                    if not newv in used:
                        used.append(newv)
                        csw.writerow([newh.hand, s])
    
### HISTORGRAM TRAINING SET ###############################################################################################################
###
###   Creates histogram over best strategies: first and all.
###

def histogram_training_set(filenames):
    counts = numpy.zeros((32,32))
    for file in filenames:
        with open(file, 'r') as fp:
            csr = csv.reader(fp, delimiter='\t')
            for row in csr:
                w = ast.literal_eval(row[1])
                for i, h in enumerate(w):
                    c = sum([ h[k] * 2**k for k in range(5) ])
                    counts[c][i] += 1
    for c in range(32):
        print c, counts[c][0], sum(counts[c])
    return counts

###########################################################################################################################################

if __name__ == '__main__':
 
    ### params

    number     = sys.argv[1]
    
    file_orig  = './kcs_data_.dat'
    
    file_boost = './kcs_data_boosted_.dat'

    ### run

    score = load_score_dict()
    
    ti = datetime.datetime.now()
    print "\n  Beginning at:", ti, '\n'

    print 'creating set'
    create_random_training_set(file_orig, number, 1)

    print 'boosting set'
    boost_training_set(file_orig, file_boost)   

    print 'original set'
    histogram_training_set([file_orig])
    
    print 'boosted set'
    histogram_training_set([file_boost]) 

    tf = datetime.datetime.now()
    print "\n  Ending at:", tf

    dt = tf - ti
    dd = divmod(dt.days * 86400 + dt.seconds, 60)
    print "\n  Time Elapsed: (min, sec) =", dd, "\n"
    
    exit()
       
###########################################################################################################################################

'''

# how to build allowed permutations for boost algorithm
# though, this still creates one long list of all perms and then delete the unallowed
# generating only the allowed in place is not something I want to do

# example:
    
    cards = [2, 2, 7, 8, 9]
    suits = [0, 1, 2, 2, 2]
    
# step 1:

    # by_suit = [ [2], [2], [7, 8, 9], [] ]

# step 2:

    # hash it
    
    def hash(inset):
        return sum([ (inset[k]+1)*(13**k) for k in len(inset) ])

    by_suit_hashed = [ hash(i) for i in by_suit ]

    # by_suit_hashed = [ 3, 3, 1815, 0 ]

    # compress(by_suit_hashed) = [1, 1, 2, 0]

# step 3:

    # zip it
    
    long = [ frozenset(zip(list(p)), compress(by_suit_hashed)) for p in itertools.permutations(range(4)) ]

    # long = [ fs([(0, 1), (1, 1), (2, 2), (3, 0)]), fs([]), fs([]), fs([]) ... fs([]) ]

# step 4:

    # remove duplicates:
    
    short = list(set(long))

    # short = [ fs([]), fs([]) ... fs([]) ]

#step 5:

    # turn inner sets into lists
    
    listed = map(list, short)

# step 6:

    # throw out the hashes
    
    final = [ [n[0] for n in m] for m in listed ]

'''    


    
