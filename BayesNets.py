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
from scipy.special import beta as B, digamma as psi
from networkx import DiGraph
from numpy import zeros, array, outer, linspace, mean, ones, copy, cos, tan, pi #note that I'm not replacing the built in "sum" function with the numpy version, because there is dependency on the usual sum function in the Entropy, Mixed, etc. methods, despite the fact that this would allow for summing over multi-dimensional arrays, which is needed for the multi-variate integrals. The fix is to call numpy.sum()
from math import sqrt, log

#import the beta function for priors on causal effects in BayesNet:
import matplotlib.pyplot as plt
from scipy.stats import beta

#The below two functions are borrowed from the suplimentary materials of the book "Computational Physics", written by Mark Newman, for integration of functions or data points, using the gaussian quadrature approach:

######################################################################
#
# Functions to calculate integration points and weights for Gaussian
# quadrature
#
# x,w = gaussxw(N) returns integration points x and integration
#           weights w such that sum_i w[i]*f(x[i]) is the Nth-order
#           Gaussian approximation to the integral int_{-1}^1 f(x) dx
# x,w = gaussxwab(N,a,b) returns integration points and weights
#           mapped to the interval [a,b], so that sum_i w[i]*f(x[i])
#           is the Nth-order Gaussian approximation to the integral
#           int_a^b f(x) dx
#
# This code finds the zeros of the nth Legendre polynomial using
# Newton's method, starting from the approximation given in Abramowitz
# and Stegun 22.16.6.  The Legendre polynomial itself is evaluated
# using the recurrence relation given in Abramowitz and Stegun
# 22.7.10.  The function has been checked against other sources for
# values of N up to 1000.  It is compatible with version 2 and version
# 3 of Python.
#
# Written by Mark Newman <mejn@umich.edu>, June 4, 2011
# You may use, share, or modify this file freely
#
######################################################################

def gaussxw(N):

    # Initial approximation to roots of the Legendre polynomial
    a = linspace(3,4*N-1,N)/(4*N+2)
    x = cos(pi*a+1/(8*N*N*tan(a)))

    # Find roots using Newton's method
    epsilon = 1e-15
    delta = 1.0
    while delta>epsilon:
        p0 = ones(N,float)
        p1 = copy(x)
        for k in range(1,N):
            p0,p1 = p1,((2*k+1)*x*p1-k*p0)/(k+1)
        dp = (N+1)*(p0-x*p1)/(1-x*x)
        dx = p1/dp
        x -= dx
        delta = max(abs(dx))

    # Calculate the weights
    w = 2*(N+1)*(N+1)/(N*N*(1-x*x)*dp*dp)

    return x,w

def gaussxwab(N,a,b):
    x,w = gaussxw(N)
    return 0.5*(b-a)*x+0.5*(b+a),0.5*(b-a)*w

##########################################################################################

#constants:

int_points, weights=gaussxwab(100,0, 1)

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
        #This is the method that needs to be exchanged if the BayesNet is to take on different forms (continuous variables, different dependencies etc.)
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

                            #Integral using gaussian quadrature
                        elif self.node[i]['causes'][node] =='+':
                            tot=outer(tot, [w*self.effect_pdf(x)*(1.0-x)**outcome[node] for x, w in zip(int_points, weights)])

                            #Integral using gaussian quadrature
                        elif self.node[i]['causes'][node] =='-':
                            tot=outer(tot, [w*self.effect_pdf(x)*(1.0-x)**(1-outcome[node]) for x, w in zip(int_points, weights)])
                        elif type(self.node[i]['causes'][node])==float and self.node[i]['causes'][node]<=0:
                            tot *=((1.0-abs(self.node[i]['causes'][node]))**(1-outcome[node]))

                    pr[i]=1-numpy.sum(tot) if outcome[i]==1 else 1-(1-numpy.sum(tot)) #Need to use numpy.sum to summ all entries in these multi-dimensional arrays.
                    p_out *=pr[i]

            self.Set(outcome, abs(p_out)) #have to take the absolute value as probabilities that are very close to 0 might become slightly negative, due to rounding errors

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

def masters_paper():
    committee=make_committee(alpha=2, beta=2)
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


def BetaEntropy(alpha, beta):
    return log(B(alpha, beta))-(alpha-1)*psi(alpha)-(beta-1)*psi(beta)+ (alpha+beta-2)*psi(alpha+beta)

ExpCausalEffect=lambda alpha, beta: float(alpha)/(alpha + beta)

EffectUncertainty=lambda alpha, beta: (ExpCausalEffect(alpha, beta), BetaEntropy(alpha, beta))

def make_entropy_axes(alphas=linspace(0, 100, 1000)[1:]):
    """
    This function first calculates Entropy values of the beta distribution,
    where alpha=beta, so that the expected value is alpha/(alpha +beta)=1.0/2.
    next, the function reorders the values for alpha such that the entropy increases.
    These alphas are later to be used as inputs in the graphs, so as to calculate
    diversity as a function of entropy, holding constant the expected value of the
    beta distribution. The entropy values are themselves exported as axis labels.
    """
    y_x=sorted((EffectUncertainty(x, x)[1], x) for x in alphas)
    alphas, entrop=[x for y, x in y_x], [y for y, x in y_x]
    return alphas, entrop

#The plot must be something of the sort: plot(entrop, Diversity(alphas))

def expected_vals_diff(alphas=linspace(0.0001, 2, 20), beta=1):
    return [float(alpha)/(alpha+beta)-float(beta)/(alpha+beta) for alpha in alphas]

#The plot must be something of the sort: plot(expected_vals_diff(), [Diversity(a, b)-Diversity(b, a) for a in linspace(0, 2)])

def Diversity(ls):
    jsd=N_point_JSD(ls)
    return sqrt(float(jsd)/log(len(ls), 2))

def diversity_diff(alphas=linspace(0.0001, 2, 20), beta=1):
    div_diff=[]
    for alpha in alphas:
        committee1=make_committee(alpha, beta)
        committee2=make_committee(beta, alpha)
        div1=Diversity(committee1)
        div2=Diversity(committee2)
        div_diff.append(div1-div2)
    return div_diff

#some stored results, the calculation of which takes rather long:
#diversity_diffs=diversity_diff()

#Diversity(alpha, beta)-Diversity(beta, alpha)
div_diff= [-0.15226733260185138, -0.1937414829631236, -0.16722646433556618, -0.13272222878491455, -0.10115433001090396, -0.0735703750643158, -0.04937081620511091, -0.027936233971641355, -0.008805360909485804, 0.008370526684388091, 0.02386825193723696, 0.03791314431690529, 0.050694291698181904, 0.062370156195378945, 0.07307411579663792, 0.08291915717118259, 0.0920016062136223, 0.10040407440799981, 0.10819780202337359]

#alpha/(alpha+beta)-beta/(alpha+beta), when beta=1 and alpha goes from 0-2:

vals_diff=[-0.8093687207763145, -0.65205180486659686, -0.51990272622552158, -0.40732922245060466, -0.31028157261373585, -0.22575504898181692, -0.1514732797595239, -0.085680164337692266, -0.026999270289992161, 0.025663510329248795, 0.073188815827983866, 0.11629345568793065, 0.15556681466469335, 0.19149796278762993, 0.22449612656223211, 0.25490634368817439, 0.28302157350349622, 0.30909216528697214, 0.33333333333333331]

def make_div_plot():
    plot(vals_diff, div_diff, 'ro')

    plt.axis([min(vals_diff)-0.05, max(vals_diff)+0.05, min(div_diff)-0.0125, max(div_diff)+0.0125])

    font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 18,
        }
    plt.title('Diversity as a Function of Expected Causal Strength', fontdict=font)
    plt.text(0.2, 1.5, r'$\alpha=\beta=2$', fontdict=font)
    plt.text(0.45, 2, r'$\alpha=4$,$\beta=2$', fontdict=font)
    plt.xlabel(r'$(\alpha-\beta)/(\alpha+\beta)$', fontdict=font)
    plt.ylabel(r'Diversity($\alpha$, $\beta$)-Diversity($\beta$, $\alpha$)', fontdict=font)

    plt.show()
