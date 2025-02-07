#some calculations for my masters paper and slides.

import sys, os
#Importing functionality from Allan Downey:
sys.path.append('/home/johannes/Documents/ThinkBayes')
from thinkbayes import Pmf
import numpy as np
from numpy.random import uniform
from thinkbayes import Suite

def gen_indepvar(Pr):
    """
    generates a binary independent variable based on the probability, Pr, that the variable takes on the value 1.
    """
    var=1 if uniform(0, 1)<Pr else 0
    return var

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def graph0(pa, pb, w0):
    A=1 if uniform(0, 1)<pa else 0
    B=1 if uniform(0, 1)<pb else 0
    C=1 if uniform(0, 1)<w0*A else 0
    return A, B, C

def graph1(pa, pb, w0, w1):
    A=1 if uniform(0, 1)<pa else 0
    B=1 if uniform(0, 1)<pb else 0
    C=1 if uniform(0, 1)< 1-((1-w0)**A)*((1-w1)**B) else 0
    return A, B, C

#first draw probabilities of the non-caused variables, (in this simple example A and B).
#pr_A=[0.01*i for i in range(1,100)]
#pr_B=[0.01*i for i in range(1,100)]

#IndepProbs=[(p, q) for q in pr_B for p in pr_A] #This is a table of all combinations (integrating out the independent probabilities)

#creating the independent data, based on the combinations of probabilities.
#DataIndep=[[(gen_indepvar(p[0]), gen_indepvar(p[1])) for i in range(10)] for p in IndepProbs]
#we want to start with low probabilities and then move up (integrate out each dimension)
#w_0=[0.01*i for i in range(1,100)]
#w_1=[0.01*i for i in range(1,100)]

#A=1 if uniform(0, 1)<pr_A else 0
#B=1 if uniform(0, 1)<pr_B else 0

#pr_C=1-((1-w_0)**A)*((1-w_1)**B)

class Cookie(Suite):
    """
    Suite is just a beginning for a general class in which hypotheses can be tested.
    A Cookie object is a Pmf that maps from hypotheses to their probabilities.
    """
    def __init__(self, hypos):
        Suite.__init__(self)
    """
    ''mixes'' is a dictionary that maps from the name of a bowl
    to the mix of cookies in the bowl (this is where probabilities in the Likelihood come from):
    """
    mixes = {
        'Bowl 1' :dict(vanilla=0.75, chocolate=0.25),
        'Bowl 2':dict(vanilla=0.5, chocolate=0.5),
        }

    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        like = mix[data]
        return like

class Monty(Suite):
    def Likelihood(self, data, hypo):
        if hypo == data:
            return 0
        elif hypo =='A':
            return 0.5
        else:
            return 1

class Die(Pmf):
    def __init__(self, sides):
        Pmf.__init__(self)
        for x in xrange(1, sides+1):
            self.Set(x, 1)
        self.Normalize()
