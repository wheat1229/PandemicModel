# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 09:10:21 2022

@author: wheat.39
"""

import numpy as np
import matplotlib.pyplot as plt


def elba(n0= (21999, 10000, 1, 0), R = (0.2, 0.2, 0.1, 0.1), \
         timeSpan = 120, nMax = 2000000, nRun = 1):

    """
    Description:

    The small isolated population of 32,000 people.
    Propose a network of elementary reactions to model how a 
    contagious disease will spread in the population. Based on kinetic rate 
    equations. 

    Parameters:\

    n0: initial healthy vaccine willing people (nH),\
        healthy free riders (nF),\
        sick people (nS), \
        vaccine doses (nV) \

    R: Rate constant adjustment factors (enter as tuple) \

    timeSpan: length of simulation (days) \

    nMax: maximum allowed number of reactions (time steps) for Gillespie \
        algorithm

    nRun: number of times to run discrete variable stochastic simulation. >= 1

    Returns: None
    """

    # ERROR MESSAGES
    if n0[0] + n0[1] + n0[2] != 32000:
        raise ValueError('Elba has 32000 residents. Revise nH, nF, nS')
    
    if n0[3] < 0:
        raise ValueError('nV must be a positive integer or zero')
    
    if nRun < 1:
        raise ValueError('nRun must be a positive integer')

    # put rate adjustmet factors into tuple R
    R5 = R[0]
    R6 = R[1]
    R7 = R[2]
    R8 = R[3]

    #rate constant adjustments
    k1 = 1.76e-5
    k2 = 0.1
    k3 = 0.01
    k4 = 3.52e-6
    k5 = R5 * k1
    k6 = R6 * k1
    k7 = R7 * k3
    k8 = R8 * k3

    # Tuple of rate constants (1/day)
    k = (k1, k2, k3, k4, k5, k6, k7, k8)

    #if nRun = 1, 3 plots 
    if nRun == 1:

        gills, nReactions = gillespie(n0, timeSpan, k, nMax)
        t = gills[0]  # time in days
        n = gills[1]  # 2D array of # rvs by timestep
        vCount = gills[2]  # 2D array of reaction event counter
        nDead = n[-1, 8]  # mortality at final time

        plt.cla()
        fig = plt.figure()
        fig.suptitle('Discrete-Variable Stochastic Model based on \
Gillespie algorithm\n\
n0 = (' + str(int(n[0][0])) + ', ' + str(int(n[0][1])) + ', '\
+ str(int(n[0][2])) + ', ' + str(int(n[0][3])) + '), nReactions = ' +\
str(int(nReactions))) 

        # plot 1, 4 curves
        a = plt.subplot(1, 3, 1)

        # current total of healthy people (H + F)
        a.step(t, n[:, 0] + n[:, 1])
        # current total of sick people (S + SIN + SIV)
        a.step(t, n[:, 2] + n[:, 6] + n[:, 7])
        # current total of people immune (IN + IV)
        a.step(t, n[:, 5] + n[:, 6])
        # current total deaths (D)
        a.step(t, n[:, 8])

        plt.xlim(0, timeSpan)
        plt.ylim(-500, 35000)
        plt.title('Pandemic Model')
        plt.xlabel('time (days)')
        plt.ylabel('number of people')
        plt.legend(('healthy', 'sick', 'immune', 'dead'), loc='best')
        plt.text(timeSpan-30, nDead+500, 'Death toll = ' + str(int(nDead)))

        # plot 2, 2 curves
        b = plt.subplot(1, 3, 2)

        #cummulative vaccinated cases
        #rxn events: IV + S -> SIV + S, IV + SIN -> SIV + SIN,IV + SIV -> 2SIV
        b.step(t, vCount[:, 12] + vCount[:, 13] + vCount[:, 14])
        #cummulative unvaccinated cases
        #rxn events: IN + S -> SIN + S, IN + SIN -> 2SIN, IN + SIV -> SIN +SIV
        b.step(t, vCount[:, 9] + vCount[:, 10] + vCount[:, 11])

        plt.xlim(0, timeSpan)
        plt.ylim(-500, 100000)
        plt.title('Cases')
        plt.xlabel('time (days)')
        plt.ylabel('number of people')
        plt.legend(('vaccinated', 'unvaccinated'), loc='best')

        # plot 3, 3 curves
        c = plt.subplot(1, 3, 3)

        # death count in each group (S, SIV, SIN)
        # S -> D
        c.step(t, vCount[:, 7])
        # SIN -> D
        c.step(t, vCount[:, 17])
        # SIV -> D
        c.step(t, vCount[:, 18])

        plt.xlim(0, timeSpan)
        plt.ylim(-5, 3000)
        plt.title('Deaths')
        # str(int(nReactions)))
        plt.xlabel('time (days)')
        plt.ylabel('number of deaths')
        plt.legend(('S', 'SIN', 'SIV'), loc='best')
        #plt.text(timeSpan-30, nDead+500, 'Death toll = ' + str(int(nDead)))

    # if nRun > 1, 1 historgram of deaths by run
    else:
        #create array with number of runs that counts number of deaths
        dCount = np.zeros((nRun))
        for i in range(0, nRun):
            gills, nReactions = gillespie(n0, timeSpan, k, nMax)
            n = gills[1]  # 2D array of # rvs by timestep
            nDead = n[-1, 8]  # mortality at final time
            #store death count for each run in dCount array
            dCount[i] = nDead

        plt.hist(dCount)
        plt.xlabel('number of deaths')
        plt.ylabel('frequency')
        plt.title('Histogram of Death Toll (' + str(int(nRun)) + '\
 simulations)')

def gillespie(n0, timeSpan, k, nMax):

    """
    Gillespie algorithm implemented for epidemic.

    Parameters:

        n0: initial values of dependent variables
        timeSpan: duration of simulation
        k: reaction rate constants (1/day)
        nMax: maximum allowed number of reactions (time steps)

    Returns:

        (t, n): tuple of 1D array of time (t) and 2D array n
        nReactions: number of reactions (time steps)
    """
    # unpack rate constants
    k1, k2, k3, k4, k5, k6, k7, k8 = k  # unpack rate constants

    # create a new tuple n1 with added parameters
    # nIN = immunity from illness (natural immunity)
    # nIV = immunity from vaccine
    # nSIN = sick individual, naturally immunized
    # nSIV = sick indivudual, vaccinated
    # nD = dead indivudual
    n1 = n0 + (0, 0, 0, 0, 0)

    """
    Reaction network has 19 reactions and we have 9 species. We need a
    19 x 9 matrix of stoiciometric cofficients. Column order must be the same
    as used in n1: [nH, nF, nS, nV, nIN, nIV, nSIN, nSIV, nD]
    """

    # STEP 2: compute rate constants
                #[[H, F, S ,V,IN,IV,SIN,SIV,D]]
    v = np.array([[-1, 0, 1, 0, 0, 0, 0, 0, 0],  # H + S -> 2S (0)
                  [-1, 0, 1, 0, 0, 0, 0, 0, 0],  # H + SIN -> S + SIN (1)
                  [-1, 0, 1, 0, 0, 0, 0, 0, 0],  # H + SIV -> S + SIV (2)
                  [0, -1, 1, 0, 0, 0, 0, 0, 0],  # F + S -> 2S (3)
                  [0, -1, 1, 0, 0, 0, 0, 0, 0],  # F + SIN -> S + SIN (4)
                  [0, -1, 1, 0, 0, 0, 0, 0, 0],  # F + SIV -> S + SIV (5)
                  [0, 0, -1, 0, 1, 0, 0, 0, 0],  # S -> IN (6)
                  [0, 0, -1, 0, 0, 0, 0, 0, 1],  # S -> D (7)
                  [-1, 0, 0, -1, 0, 1, 0, 0, 0],  # H + V -> IV (8)
                  [0, 0, 0, 0, -1, 0, 1, 0, 0],  # IN + S -> SIN + S (9)
                  [0, 0, 0, 0, -1, 0, 1, 0, 0],  # IN + SIN -> 2SIN (10)
                  [0, 0, 0, 0, -1, 0, 1, 0, 0],  # IN + SIV -> SIN + SIV (11)
                  [0, 0, 0, 0, 0, -1, 0, 1, 0],  # IV + S -> SIV + S (12)
                  [0, 0, 0, 0, 0, -1, 0, 1, 0],  # IV + SIN -> SIV + SIN (13)
                  [0, 0, 0, 0, 0, -1, 0, 1, 0],  # IV + SIV -> 2SIV (14)
                  [0, 0, 0, 0, 1, 0, -1, 0, 0],  # SIN -> IN (15)
                  [0, 0, 0, 0, 0, 1, 0, -1, 0],  # SIV -> IV (16)
                  [0, 0, 0, 0, 0, 0, -1, 0, 1],  # SIN -> D (17)
                  [0, 0, 0, 0, 0, 0, 0, -1, 1]])  # SIV -> D (18)

    # Pre-allocate arrays for t (time in days) and n (number of rxns)
    t = np.zeros((nMax, 1))  # column vector for time with 1 column
    n = np.zeros((nMax, 9))  # column vector for rxn elements with 9 columns
    # column vector for rxn events with 9 columns
    vCount = np.zeros((nMax, 19))

    # STEP 1: set first row of n to initial conditions at t = 0
    n[0, :] = n1

    # Initialize reaction counter
    nReactions = 0

    # STEP 2: ensure units are correct. Check.

    # Begin for loop to iterate though defined max number of reactions
    for i in range(1, nMax):
        # set rxn events to the initial conditions
        nH, nF, nS, nV, nIN, nIV, nSIN, nSIV, nD = n[i-1, :]

        # calculate reaction probabilities
        r = np.array([k1 * nH * nS,
                      k1 * nH * nSIN,
                      k1 * nH * nSIV,
                      k1 * nF * nS,
                      k1 * nF * nSIN,
                      k1 * nF * nSIV,
                      k2 * nS,
                      k3 * nS,
                      k4 * nH * nV,
                      k5 * nIN * nS,
                      k5 * nIN * nSIN,
                      k5 * nIN * nSIV,
                      k6 * nIV * nS,
                      k6 * nIV * nSIN,
                      k6 * nIV * nSIV,
                      k2 * nSIN,
                      k2 * nSIV,
                      k7 * nSIN,
                      k8 * nSIV])

        # STEP 3: Compute proportional probability of each rxn
        rtot = np.sum(r)

        # stop if rtot is 0
        if rtot == 0:
            break
        # increment reaction counter otherwise
        else:
            nReactions += 1

        # STEP 4: Generate random number U(0,1)
        w = np.random.uniform()

        # STEP 5: Compute time interval until next rxn event
        tau = -np.log(w) / rtot

        # STEP 6: Update simulation time
        t[i] = t[i-1] + tau

        # STEP 7: Calculated vector of rxn probabillities
        p = r / rtot

        # STEP 8: Calculate vector of cumulative sums
        csp = np.cumsum(p)

        # STEP 9: Generate random number U(0,1)
        q = np.random.uniform()

        # STEP 10: determine where rxn occurs, generates number btwn 0 and 18
        # if q < sp
        j = np.where(q < csp)[0][0]

        # keep track of which reactions are occuring
        # update current timestep with previous timestep data
        vCount[i, :] = vCount[i-1, :]
        # reaction event that occurs, j, get incremented
        vCount[i, j] += 1

        # STEP 11: adjust population numbers depending on selected rxn
        n[i, :] = n[i-1, :] + v[j, :]

        # STEP 12: repeat steps 3-12 unless t > timeSpan or nRxns > nMax
        if t[i] >= timeSpan:
            break

    # if statement ensures dimensions match for plotting purposes
    if nReactions < nMax:
        t = t[0:nReactions + 1]
        n = n[0:nReactions + 1, :]
        vCount = vCount[0:nReactions + 1, :]

    return (t, n, vCount), nReactions


# =============================================================================
# Self-test code
if __name__ == '__main__':
    elba( n0 = (21999, 10000, 1, 14000), R = (0.2, 0.2, 0.1, 0.1),\
timeSpan = 120, nMax = 2000000, nRun = 1)