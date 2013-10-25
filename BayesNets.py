__author__ ='Johannes Castner'

"""
BayesNet inherits from both, a networkx DiGraph and Joint, a class object representing a joint distribution which is code that was originally written by by Allen B. Downey, for his book "Think Bayes", available from greenteapress.com
Copyright 2013 Johannes Castner, 2012 Allen B. Downey,
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html

2006-2011 Aric Hagberg <hagberg@lanl.gov>, Dan Schult <dschult@colgate.edu>, Pieter Swart <swart@lanl.gov>. BSD license.
"""
import sys
sys.path.append('/home/johannes/Documents/ThinkBayes')
import bisect
import copy
import logging
import math
import numpy
import random
from thinkbayes import * #must have thinkbayes.py in the current directory or in the system path.
import scipy.stats
from scipy.special import erf, erfinv
from networkx import DiGraph
from numpy import zeros, array, outer, linspace, mean
from math import sqrt, log

#import the beta function for priors on causal effects in BayesNet:
import matplotlib.pyplot as plt
from scipy.stats import beta

class BayesNet(DiGraph, Joint):
    """
A BayesNet is both, a graph and a joint distribution. For now, it only
allows for binary variables. All methods that start with capital letters
relate more to joint distributions and those that are more graph-related start
with lower case letters.
The joint probability is encoded, using the Noisy-OR encoding (Pearl 1988).
"""

    def __init__(self, data=None, name='', p=0.5, effect_pdf=lambda x, a, b : beta(a, b).pdf(x), a=2, b=2):
        """
Effect pdf is the pdf that a causal effect is assumed to be drawn from. In the absense of a "causal_effect" that is explicitly
included in the link between two nodes, we integrate over this distribution.
"""

        DiGraph.__init__(self, data=data, name=name)
        Joint.__init__(self)
        self.p=p
        self.n=len(self.nodes())
        self.support=[]
        self.names=[]
        self.indep_vars=[]
        self.dep_vars=[]
        self.effect_pdf= lambda x: effect_pdf(x, a, b)


    def add_edges_from(self, ebunch):
        #first, we rename the nodes to be indices. This is practical
        #for manipulating the joint distribution (maybe not ideal).
        #newbunch=[e for e in ebunch if len(e)==3]
        #singletons=[v for v in e for e in ebunch if len(e)==1]
        #It would be interesting to extend the graphing capabilities of networkx to be able
        #to attach edge labels ('+', '-', '0' etc.) and show them.
        if self.names:
            self.names=sorted(set([v for e in ebunch for v in e[:2]] + self.names))

        #if we don't sort, we get in trouble with comparing joint distributions (the same nodes can be ordered differently)
            check_bunch=[(self.names[i], self.names[j]) for i, j in self.edges()]
            if [e[:2] for e in ebunch] in check_bunch:
                return
        else:
            self.names=sorted(set([v for e in ebunch for v in e[:2]]))


        for e in ebunch:
            index=ebunch.index(e)
            if len(e)==3:
                l, k, dd=e
                if type(dd)==dict:
                    dd=dd
                elif type(dd)==float or type(dd)==int:
                    dd={'weight': dd}
                else:
                    dd={'causal_effect': dd}

            elif len(e)==2:
                l, k = e
                dd={}
            i, j =self.names.index(l), self.names.index(k)
            ebunch[index]=i, j, dd

        for i, n in enumerate(self.names):
            for node in self.nodes():
                if n ==self.node[node]['name']:
                    self.node[i]={'name': n}
                    self.edge[i]={}

        DiGraph.add_edges_from(self, ebunch=ebunch)

        self.n=len(self.nodes())
        #attach the names:
        for node in self.nodes():
            self.node[node]['name']=self.names[node]
        #number of nodes
        fro, to = zip(*self.edges())
        self.indep_vars=sorted(set(f for f in self.nodes() if f not in set(to)))
        self.dep_vars =sorted(set(to))
        for var in self.indep_vars:
            self.node[var]['pmf']=Pmf()
            self.node[var]['pmf'].Set(1,self.p)
            self.node[var]['pmf'].Set(0, 1-self.p)
        #for var in self.dep_vars:
         # self.node[var]['pmf']=Pmf()
         # self.node[var]['pmf'].Set(1,0) #first set it all to 0
         # self.node[var]['pmf'].Set(0, 1)


        for w in self.dep_vars:
            self.node[w]['causes']={}

        for i in self.nodes():
            for j in self.edge[i]:
                self.node[j]['causes'][i]=self.edge[i][j]['causal_effect'] if 'causal_effect' in self.edge[i][j] else self.edge[i][j]['weight'] if 'weight' in self.edge[i][j] else '+'


        self.SetProbs()

    def SetProbs(self):
        self.d={}
        self.support=[]
        n=len(self.nodes())
        for outcome in range(2**n):
            self.support.append(tuple([(outcome>>i)&1 for i in xrange(n-1,-1,-1)]))


        for outcome in self.support:
            pr=[0]*len(outcome)
            p_out=1 #initializing the total probability of outcome
            for i in range(len(outcome)):
                if i in self.indep_vars:
                    pr[i]=self.node[i]['pmf'].d[outcome[i]]
                    p_out *=pr[i]
                else:
                    tot=1
                    for node in self.node[i]['causes']:
                        if type(self.node[i]['causes'][node])==float and self.node[i]['causes'][node]>=0:
                            tot *=((1.0-self.node[i]['causes'][node])**outcome[node])
                        elif self.node[i]['causes'][node] =='+':
                            tot=outer(tot, [self.effect_pdf(w)*(1.0-w)**outcome[node] for w in linspace(0, 1, 100)])

                        elif self.node[i]['causes'][node] =='-':
                            tot=outer(tot, [self.effect_pdf(w)*(1.0-w)**(1-outcome[node]) for w in linspace(0, 1, 100)])
                        elif type(self.node[i]['causes'][node])==float and self.node[i]['causes'][node]<=0:
                            tot *=((1.0-abs(self.node[i]['causes'][node]))**(1-outcome[node]))

                    pr[i]=1-mean(tot) if outcome[i]==1 else 1-(1-mean(tot))
                    p_out *=pr[i]

            self.Set(outcome, p_out)

    def add_nodes_from(self, nodes, **attr):
        H=DiGraph()
        H.add_nodes_from(self.names)
        h_names=sorted(H.nodes())
        H.add_edges_from([(h_names[e[0]], h_names[e[1]], self.edge[e[0]][e[1]]) for e in self.edges()])

        causes={h_names[v]: {h_names[item]: self.node[v]['causes'][item] for item in self.node[v]['causes']} for v in self.dep_vars}

        self.clear()
        self.indep_vars=[]
        self.dep_vars=[]
        if not H.nodes():
            DiGraph.add_nodes_from(self, nodes, **attr)

            self.names=names=sorted(nodes)
            for i, n in enumerate(self.names):
                self.node[i]={'name': n, 'pmf': Pmf()}
                self.node[i]['pmf'].Set(1,self.p)
                self.node[i]['pmf'].Set(0, 1-self.p)
                self.remove_node(n)
                self.edge[i]={}
                self.indep_vars+=[i]
            self.SetProbs()
            return

        #DiGraph.add_nodes_from(self, nodes, **attr)
        #ind_vars=[var for var in H.indep_vars]
        #DiGraph.add_nodes_from(self, ind_vars)
        self.names=names=sorted(set(H.nodes() + nodes))
        for i, n in enumerate(names):
            if n in H.nodes():
                self.node[i], self.edge[i]=H.node[n], {names.index(item): H.edge[n][item] for item in H.edge[n]}
                self.node[i]['causes']={names.index(item): causes[n][item] for item in causes[n]} if n in causes else {}
                self.node[i]['name']=n
                self.node[i]['pmf']=Pmf()
                if not self.node[i]['causes']:
                    self.node[i]['pmf'].Set(1,self.p)
                    self.node[i]['pmf'].Set(0, 1-self.p)
                    self.indep_vars+=[i]
                else: self.dep_vars+=[i]
            else:
                self.node[i]={'name': n, 'pmf': Pmf()}
                self.node[i]['pmf'].Set(1,self.p)
                self.node[i]['pmf'].Set(0, 1-self.p)
                #self.remove_node(n)
                self.edge[i]={}
                self.indep_vars+=[i]

        self.SetProbs()

    def MakeMixture(self, other, lamb=0.5):
        mixed = Joint() #mixing the two probability distributions
        for x, p in self.Items():
            mixed.Set(x,lamb * p + (1 - lamb) * other.d[x])
        return mixed

    def KL_divergence(self, other):
        """ Compute KL divergence of two BayesNets."""
        try:
            return sum(p * log((p /other.d[x])) for x, p in self.Items() if p != 0.0 or p != 0)
        except ZeroDivisionError:
            return float("inf")

    def JensenShannonDivergence(self, other):
        JSD = 0.0
        lamb=0.5
        mix=self.MakeMixture(other=other, lamb=0.5)
        JSD=lamb * self.KL_divergence(mix) + lamb * other.KL_divergence(mix)
        return JSD

    def Entropy(self):
        return -sum(p * log(p, 2) for x, p in self.Items() if p != 0.0 or p != 0)


def MixedBN(ls):
    """
    ls: a list of BayesNets.
    """
    #for g in ls:
    #    if type(g)!=BayesNet:
    #        raise TypeError
    if len(ls)==1:
        return ls[0]
    lamb = 1.0/len(ls)
    items=list({x for g in ls for x, p in g.Items()})
    mix = BayesNet()
    [mix.Set(x,sum(lamb*g.d[x] for g in ls if x in g.d)) for x in items]
    return mix

def N_point_JSD(ls):
    mix=MixedBN(ls)
    return mix.Entropy()-(1.0/len(ls))*sum(g.Entropy() for g in ls)



def make_committee(alpha=2, beta=2):
    Bernanke=BayesNet(a=alpha, b=beta); Paulson=BayesNet(a=alpha, b=beta); Morgenthau=BayesNet(a=alpha, b=beta); Becker=BayesNet(a=alpha, b=beta); Stiglitz=BayesNet(a=alpha, b=beta); Born=BayesNet(a=alpha, b=beta); Greenspan=BayesNet(a=alpha, b=beta) ;Buffett=BayesNet(a=alpha, b=beta); Krugman=BayesNet(a=alpha, b=beta); Rodrik=BayesNet(a=alpha, b=beta); Soros=BayesNet(a=alpha, b=beta)

    committee=[Bernanke, Paulson, Morgenthau, Becker, Stiglitz, Born, Greenspan, Buffett, Krugman, Rodrik, Soros]



    Bernanke.add_edges_from([('S', 'C', '-'), ('R', 'C', '-')]); Paulson.add_edges_from([('GSE', 'C', '+')]); Morgenthau.add_edges_from([('O', 'T', '-'), ('S', 'T', '+'), ('T', 'C', '-')]); Becker.add_edges_from([('I', 'C', '-'), ('R', 'C', '+'), ('GSE', 'C', '+')]); Stiglitz.add_edges_from([('MC', 'B', '+'), ('R', 'B', '-'), ('B', 'C', '+')]); Born.add_edges_from([('S', 'C', '-'), ('B', 'C', '+'), ('R', 'C', '-')]); Greenspan.add_edges_from([('GSE', 'C', '+'), ('O', 'C', '+')]); Buffett.add_edges_from([('CD', 'C', '-')]); Krugman.add_edges_from([('R', 'C', '-')]); Rodrik.add_edges_from([('O', 'C', '+'), ('G', 'C', '+'), ('R', 'C', '-')]); Soros.add_edges_from([('MC', 'C', '+'), ('B', 'C', '+')])

    vars=list({name for member in committee for name in member.names})

    [member.add_nodes_from(vars) for member in committee]
    return committee

committee_names=['Bernanke', 'Paulson', 'Morgenthau', 'Becker', 'Stiglitz', 'Born', 'Greenspan', 'Buffett', 'Krugman', 'Rodrik', 'Soros']

def distance_matrix(committee):
    distances=zeros([len(committee)]*2)
    n, m=distances.shape
    for i in range(n):
        for j in range(m):
            distances[i, j]=round(sqrt(committee[i].JensenShannonDivergence(committee[j])), 4)
    print " \\\\\n".join([" & ".join(map(str,line)) for line in distances])
    return distances

def marginal_diversity(committee):
    ls=[]
    [ls.append(round(sqrt(N_point_JSD(committee)/log(len(committee), 2))-sqrt(N_point_JSD(committee[:i]+committee[i+1:])/log(len(committee[:i]+committee[i+1:]), 2)), 4)) for i in range(len(committee))]
    return ls

def stiglitz_adjustment(committee):
#resetting Stiglitz:
    committee[4]=BayesNet()
#giving him a causal belief structure that is more similar to the coarse graining of others:
    committee[4].add_edges_from([('MC', 'C', '+'), ('R', 'C', '-'), ('B', 'C', '+')])

    vars=list({name for member in committee for name in member.names})
    committee[4].add_nodes_from(vars)
    return committee

def plot_betas():
    xs=linspace(0, 1, 30)
    plt.plot(xs, [beta(4, 2).pdf(x) for x in xs], 'bs', xs, [beta(2, 2).pdf(x) for x in xs], 'g^' )
    font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 18,
        }
    plt.title('The Beta Distribution', fontdict=font)
    plt.text(0.2, 1.5, r'$\alpha=\beta=2$', fontdict=font)
    plt.text(0.45, 2, r'$\alpha=4$,$\beta=2$', fontdict=font)
    plt.xlabel('causal-strength', fontdict=font)
    plt.ylabel('Density', fontdict=font)

    plt.show()

def sort_distances(distances, committee_names):
    n, m=distances.shape
    return sorted([(distances[i][j], committee_names[i], committee_names[j]) for i in range(m) for j in range(n)], reverse=True)

S_9=committee[1:]
t_9=marginal_diversity(S_9)
min_9=min(marginal_diversity(S_9))
S_8=S_9[:5]+S_9[6:]
t_8=marginal_diversity(S_8)
min_8=min(t_8)
S_7=S_8[:-1]
t_7=marginal_diversity(S_7)
min_7=min(t_7)
S_6=S_7[:4]+S_7[5:]
t_6=marginal_diversity(S_6)
min_6=min(t_6)
S_5=S_6[1:]
t_5=marginal_diversity(S_5)
min_6=min(t_5)
