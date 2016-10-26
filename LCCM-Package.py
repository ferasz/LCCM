"""
@name:      Latent Class Choice Model (EM algorithm)
@author:    Feras El Zarwi and Akshay Vij
@summary:   Contains functions necessary for estimating latent class choice models
            using the Expectation Maximization algorithm. This version also accounts
            for choice based sampling using Weighted Exogenous Sample Maximum
            Likelihood.
General References
------------------
Jordan, M. I. An Introduction to Probabilistic Graphical Models. 
    University of California, Berkeley, 2003.
"""


import numpy as np 
from scipy.sparse import coo_matrix
from scipy.optimize import minimize
from datetime import datetime
import warnings


def processClassMem(expVars, dmID, nClasses, availClasses):
    """
    Method that constructs two sparse matrices containing information on the 
    classes available to each decision-maker and the utility specification 
    for each of the classes.
    
    Parameters
    ----------
    expVars : 2D numpy array of size (nVars x nRows).
        The rows correspond to explanatory variables and the columns correspond
        to decision-makers. 
        Note that multiple columns may correspond to the same decision-maker.
    dmID : 1D numpy array of size nRows.
        Identifies which columns in expVars correspond to which decision-maker.
    nClasses : integer.
        Identifies the number classes to be modeled.
    availClasses : 2D numpy array of size (nClasses x nRows).
    The (i, j)th element equals 1 if the ith class is available to the
    decision-maker corresponding to the jth row in the dataset, and 0 otherwise.
    Note that we usually assume all classes are avialable to all decision-makers.
    
    Returns
    -------
    newExpVars : 2D numpy array of size ((nVars * (nClasses - 1)) x (nDms * nClasses)). 
        The (((n - 1) * nClasses) + s)th column of the returned array specifies 
        the utility of the sth class for the nth decision-maker in the dataset.     
    classAv : sparse matrix of size ((nDms * nClasses) x nDms).
        The (i, j)th element of the returned matrix equals 1 if the class corresponding 
        to the ith column of newExpVars is available to the jth decision-maker,
        and 0 otherwise.
    """

    dms = np.unique(dmID)
    nDms = dms.shape[0]
    nExpVarsPerClass = expVars.shape[0]
    nExpVarsTotal = nExpVarsPerClass * (nClasses - 1)
    newExpVars = np.zeros((nExpVarsTotal, nDms * nClasses))
    classAvail = np.zeros(nDms * nClasses)
    xDm, yDm = np.zeros((nDms * nClasses)), np.zeros((nDms * nClasses))

    currentCol, currentRow, currentInd = 0, 0, 0
    for n in dms:
        cExpVars = np.mean(expVars[:, dmID == n], axis = 1)
        currentCol += 1
        for s in range(1, nClasses):
            tExpVars = np.vstack((np.zeros((s - 1, nExpVarsPerClass)), cExpVars, 
                    np.zeros((nClasses - (s + 1), nExpVarsPerClass))))
            newExpVars[:, currentCol] = tExpVars.reshape((nExpVarsTotal), order = 'F')
            currentCol += 1
        
        cAvail = (np.mean(availClasses[:, dmID == n], axis = 1) > 0).astype(int)
        for s in range(0, nClasses):
            if cAvail[s] == 1:
                classAvail[currentRow] = 1
            xDm[currentRow] = currentRow
            yDm[currentRow] = currentInd
            currentRow += 1
        
        currentInd += 1

    classAv = coo_matrix((classAvail, (xDm, yDm)), shape = (nDms * nClasses, nDms))
    return newExpVars, classAv


def processClassSpecificPanel(dms, dmID, obsID, altID, choice):
    """
    Method that constructs a tuple and three sparse matrices containing information 
    on available observations, and available and chosen alternative
    
    Parameters
    ----------
    dms : 1D numpy array of size nDms.
        Each element identifies a unique decision-maker in the dataset.
    dmID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which decision-maker.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation.
    altID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which alternative.
    choice : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to alternatives that were 
        chosen during the corresponding observation.
    
    Returns
    -------
    altAvTuple : a tuple containing arrays of size nRows.
        The first array denotes which row in the data file corresponds to which 
        row in the data file (redundant but retained for conceptual elegance) and 
        the second array denotes which row in the data file corresponds to which 
        observation in the data file.    
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element of the returned matrix  is 1 if the alternative corresponding 
        to the ith row in the data file was chosen during observation j, and 0 otherwise.   
    obsAv : sprase matrix of size (nObs x nDms).
        The (i, j)th element of the returned matrix is 1 if observation i corresponds to 
        decision-maker j, and 0 otherwise.
    rowAv : sparse matrix of size (nRows x (nAlts * nDms)).
        The (i, ((n - 1) * nAlts) + j)th element of the returned matrix is 1 if the ith row 
        in the data file corresponds to the jth alternative and the nth decision-maker, 
        and 0 otherwise.   
    """
    
    nRows = choice.shape[0]
    alts = np.unique(altID)
    nAlts = alts.shape[0]
    obs = np.unique(obsID)
    nObs = obs.shape[0]
    nDms = dms.shape[0]
    
    xAlt, yAlt = np.zeros((nRows)), np.zeros((nRows))
    xChosen, yChosen = np.zeros((nObs)), np.zeros((nObs))
    xObs, yObs = np.zeros((nObs)), np.zeros((nObs))
    xRow, yRow = np.zeros((nRows)), np.zeros((nRows))

    currentRow, currentObs, currentDM = 0, 0, 0    
    for n in dms:
        obs = np.unique(np.extract(dmID == n, obsID))
        for k in obs:      
            xObs[currentObs], yObs[currentObs] = currentObs, currentDM
            cAlts = np.extract((dmID == n) & (obsID == k), altID)        
            for j in cAlts:
                xAlt[currentRow], yAlt[currentRow] = currentRow, currentObs  
                xRow[currentRow], yRow[currentRow] = currentRow, (np.where(dms == n)[0][0] * nAlts) + np.where(alts == j)[0][0]
                if np.extract((dmID == n) & (obsID == k) & (altID == j), choice) == 1:                
                    xChosen[currentObs], yChosen[currentObs] = currentRow, currentObs
                currentRow += 1
            currentObs += 1
        currentDM += 1
            
    altAvTuple = (xAlt, yAlt)
    altChosen = coo_matrix((np.ones((nObs)), (xChosen, yChosen)), shape = (nRows, nObs))
    obsAv = coo_matrix((np.ones((nObs)), (xObs, yObs)), shape = (nObs, nDms))
    rowAv = coo_matrix((np.ones((nRows)), (xRow, yRow)), shape = (nRows, nDms * nAlts))
    
    return altAvTuple, altChosen, obsAv, rowAv
    
    
def imposeCSConstraints(altID, availAlts):
    """
    Method that constrains the choice set for each of the decision-makers across the different
    latent classes following the imposed choice-set by the analyst to each class. 
    Usually, when the data is in longformat, this would not be necessary, since the 
    file would contain rows for only those alternatives that are available. However, 
    in an LCCM, the analyst may wish to impose additional constraints to introduce 
    choice-set heterogeneity.
    
    Parameters
    ----------
    altID : 1D numpy array of size nRows.
        Identifies which rows in the data correspond to which alternative.
    availAlts : List of size nClasses.
        Determines choice set constraints for each of the classes. The sth element is an 
        array containing identifiers for the alternatives that are available to decision-makers 
        belonging to the sth class.
    
    Returns
    -------
    altAvVec : 1D numpy array of size nRows.
        An element is 1 if the alternative corresponding to that row in the data 
        file is available, and 0 otherwise.
    """   
    altAvVec = np.zeros(altID.shape[0]) != 0   
    for availAlt in availAlts:
        altAvVec = altAvVec | (altID == availAlt)
    return altAvVec.astype(int)


def calClassMemProb(param, expVars, classAv):
    """
    Function that calculates the class membership probabilities for each observation in the
    dataset.
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values of class membership model.
    expVars : 2D numpy array of size (nExpVars x (nDms * nClasses)).
        Contains explanatory variables of class membership model.
    classAv : sparse matrix of size ((nDms * nClasses) x nDms).
        The (i, j)th element equals 1 if ith row in expVars corresponds to the 
        jth decision-maker, and 0 otherwise.
    
    Returns
    -------
    p : 2D numpy array of size 1 x (nDms x nClasses).
        Identifies the class membership probabilities for each individual and 
        each available latent class.
    """
    
    v = np.dot(param[None, :], expVars)       # v is 1 x (nDms * nClasses)
    ev = np.exp(v)                            # ev is 1 x (nDms * nClasses)
    ev[np.isinf(ev)] = 1e+20                  # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                  # As precaution when exp(v) is too close to zero
    nev = ev * classAv                        # nev is 1 x (nDms * nClasses)
    nnev = classAv * np.transpose(nev)        # nnev is (nDms * nClasses) x 1
    p = np.divide(ev, np.transpose(nnev))     # p is 1 x (nDms * nClasses) 
    p[np.isinf(p)] = 1e-200                   # When the class is unavailable
    
    return p


def calClassSpecificProbPanel(param, expVars, altAvMat, altChosen, obsAv):
    """
    Function that calculates the class specific probabilities for each decision-maker in the
    dataset
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAvMat : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars was chosen by the decision-maker corresponding to the 
        jth observation, and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    
    Returns
    -------
    np.exp(lPInd) : 2D numpy array of size 1 x nInds.
        Identifies the class specific probabilities for each individual in the 
        dataset.
    """
    v = np.dot(param[None, :], expVars)       # v is 1 x nRows
    ev = np.exp(v)                            # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                  # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                  # As precaution when exp(v) is too close to zero
    nev = ev * altAvMat                     # nev is 1 x nObs
    nnev = altAvMat * np.transpose(nev)     # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))     # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                   # When none of the alternatives are available
    pObs = p * altChosen                      # pObs is 1 x nObs
    lPObs = np.log(pObs)                      # lPObs is 1 x nObs
    lPInd = lPObs * obsAv                     # lPInd is 1 x nInds
    return np.exp(lPInd)                      # prob is 1 x nInds

          
def wtClassMem(param, expVars, classAv, weightsProb, weightsInd):
    """
    Function that calculates the log-likelihood function and the gradient for a weighted 
    multinomial logit model where there is no observable dependent variable. 
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nDms * nClasses)).
        Contains explanatory variables.
    classAv : sparse matrix of size ((nDms * nClasses) x nDms).
        The (i, j)th element equals 1 if the class corresponding to the ith column 
        of newExpVars is available to the jth decision-maker, and 0 otherwise.
    weightsProb : 2D numpy array of size (nDms x nClasses).
        The ((n - 1) * nClasses + s)th element is the weight to be used for the 
        sth class and the nth decision-maker.
    weightsInd : 2D numpy array of size (nDms x nClasses,1).
        The ((n-1)xnClasses + s)th element is the weight to be used for the nth
        decision-maker for the sth class. The weights across all classes for the 
        same decision-maker is the same. This array facilitates the computation.
    
    Returns
    -------
    ll : a scalar.
        Log-likelihood value for the weighted multinomidal logit model.
    np.asarray(gr).flatten() : 1D numpy array of size nExpVars.
        Gradient for the weighted multinomial logit model.
    
    """ 
    v = np.dot(param[None, :], expVars)               # v is 1 x (nDms * nClasses)
    ev = np.exp(v)                                    # ev is 1 x (nDms * nClasses)
    ev[np.isinf(ev)] = 1e+20                          # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                          # As precaution when exp(v) is too close to zero
    nev = ev * classAv                                # nev is 1 x nDms
    nnev = classAv * np.transpose(nev)                # nnev is (nDms * nClasses) x 1
    p = np.divide(np.transpose(ev), nnev)             # p is (nDms * nClasses) x 1
    p[np.isinf(p)] = 1e-200                           # When the class is unavailable

    tgr = np.multiply(weightsInd, p - weightsProb)    # tgr is (nDms * nClasses) x 1
    gr = np.dot(expVars, tgr)                         # gr is nExpVars x 1    

    pInd = np.multiply(np.log(p), weightsProb)        # pInd is (nDms * nClasses) x 1
    nPInd = np.multiply(pInd, weightsInd)             # nPInd is (nDms * nClasses) x 1
    ll = -np.sum(nPInd)                               # ll is a scalar
    
    return ll, np.asarray(gr).flatten()

def calStdErrClassMem(param, expVars, classAv, weightsProb):
    """
    Function that calculates the standard errors for a weighted multinomial logit model 
    where there is no observable dependent variable.
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nDms * nClasses)).
        Contains explanatory variables.
    classAv : sparse matrix of size ((nDms * nClasses) x nDms).
        The (i, j)th element equals 1 if the class corresponding to the ith column 
        of newExpVars is available to the jth decision-maker, and 0 otherwise.
    weightsProb : 2D numpy array of size (nDms x nClasses).
        The ((n - 1) * nClasses + s)th element is the weight to be used for the 
        sth class and the nth decision-maker.
    
    Returns
    -------
    se : 2D numpy array of size (nExpVars x 1).
        Standard error for the weighted multinomidal logit model.
    
    """   
    v = np.dot(param[None, :], expVars)       # v is 1 x (nDms * nClasses)
    ev = np.exp(v)                            # ev is 1 x (nDms * nClasses)
    ev[np.isinf(ev)] = 1e+20                  # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                  # As precaution when exp(v) is too close to zero
    nev = ev * classAv                        # nev is 1 x nDms
    nnev = classAv * np.transpose(nev)        # nnev is (nDms * nClasses) x 1
    p = np.divide(np.transpose(ev), nnev)     # p is (nDms * nClasses) x 1
    p[np.isinf(p)] = 1e-200                   # When the class is unavailable
    gr = np.transpose(weightsProb - p)        # gr is 1 x (nDms * nClasses) 
    tgr = np.tile(gr, (expVars.shape[0], 1))  # tgr is nExpVars x (nDms * nClasses) 
    ttgr = np.multiply(expVars, tgr)          # ttgr is nExpVars x (nDms * nClasses) 
    hess = np.dot(ttgr, np.transpose(ttgr))   # hess is nExpVars x nExpVars 
    try:                                      # iHess is nExpVars x nExpVars 
        iHess = np.linalg.inv(hess)           # If hess is non-singular
    except:
        iHess = np.identity(expVars.shape[0]) # If hess is singular
    se = np.sqrt(np.diagonal(iHess))          # se is nExpVars x 1
    
    return se
    

def wtLogitPanel(param, expVars, altAv, weightsProb, weightsGr, altChosen, obsAv, choice):
    """
    Function that calculates the log-likelihood function and the gradient for a weighted
    multinomial logit model with panel data. 
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAv : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise. 
    weightsProb : 1D numpy array of size nInds.
        The jth element is the weight to be used for the jth decision-maker.
    weightsGr : 1D numpy array of size nRows.
        The jth element is the weight to be used for the jth row in the dataset. 
        The weights will be the same for all rows in the dataset corresponding to 
        the same decision-maker. However, passing it as a separate parameter speeds up the optimization.   
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith column 
        in expVars was chosen by the decision-maker corresponding to the jth observation,
        and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    choice : 1D numpy array of size nRows.
        The jth element equals 1 if the alternative corresponding to the jth column 
        in expVars was chosen by the decision-maker corresponding to that observation, and 0 otherwise.
        
    Returns
    -------
    ll : a scalar.
        Log-likelihood value for the weighted multinomidal logit model.
    np.asarray(gr).flatten() : 1D numpy array of size nExpVars.
        Gradient for the weighted multinomial logit model.
    
    """       
    v = np.dot(param[None, :], expVars)         # v is 1 x nRows
    ev = np.exp(v)                              # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                    # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                    # As precaution when exp(v) is too close to zero
    nev = ev * altAv                            # nev is 1 x nObs
    nnev = altAv * np.transpose(nev)            # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))       # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                     # When none of the alternatives are available
    p[p < 1e-200] = 1e-200                      # As precaution when p is too close to zero
    tgr = choice - np.transpose(p)              # ttgr is nRows x 1
    ttgr = -np.multiply(weightsGr, tgr)         # tgr is nRows x 1
    gr = np.dot(expVars, ttgr)                  # gr is nExpVars x 1
    pObs = p * altChosen                        # pObs is 1 x nObs
    lPObs = np.log(pObs)                        # lPObs is 1 x nObs
    lPInd = lPObs * obsAv                       # lPInd is 1 x nInds
    wtLPInd = np.multiply(lPInd, weightsProb)   # wtLPInd is 1 x nInds
    ll = -np.sum(wtLPInd)                       # ll is a scalar
    
    return ll, np.asarray(gr).flatten()
    

def calStdErrWtLogitPanel(param, expVars, altAv, weightsProb, weightsGr, altChosen, obsAv, choice):
    """
    Function that calculates the log-likelihood function and the gradient for a weighted
    multinomial logit model with panel data. 
    
    Parameters
    ----------
    param : 1D numpy array of size nExpVars.
        Contains parameter values.
    expVars : 2D numpy array of size (nExpVars x (nRows)).
        Contains explanatory variables.
    altAv : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith 
        column in expVars is available to the decision-maker corresponding to the 
        jth observation, and 0 otherwise. 
    weightsProb : 1D numpy array of size nInds.
        The jth element is the weight to be used for the jth decision-maker.
    weightsGr : 1D numpy array of size nRows.
        The jth element is the weight to be used for the jth row in the dataset. 
        The weights will be the same for all rows in the dataset corresponding to 
        the same decision-maker. However, passing it as a separate parameter speeds up the optimization.   
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element equals 1 if the alternative corresponding to the ith column 
        in expVars was chosen by the decision-maker corresponding to the jth observation,
        and 0 otherwise.
    obsAv : sparse matrix of size (nObs x nInds).
        The (i, j)th element equals 1 if the ith observation in the dataset corresponds 
        to the jth decision-maker, and 0 otherwise.
    choice : 1D numpy array of size nRows.
        The jth element equals 1 if the alternative corresponding to the jth column 
        in expVars was chosen by the decision-maker corresponding to that observation, and 0 otherwise.
        
    Returns
    -------
    se : 2D numpy array of size (nExpVars x 1).
        Standard error for the weighted multinomidal logit model.
    
    """ 
    v = np.dot(param[None, :], expVars)         # v is 1 x nRows
    ev = np.exp(v)                              # ev is 1 x nRows
    ev[np.isinf(ev)] = 1e+20                    # As precaution when exp(v) is too large for machine
    ev[ev < 1e-200] = 1e-200                    # As precaution when exp(v) is too close to zero
    nev = ev * altAv                            # nev is 1 x nObs
    nnev = altAv * np.transpose(nev)            # nnev is nRows x 1
    p = np.divide(ev, np.transpose(nnev))       # p is 1 x nRows 
    p[np.isinf(p)] = 1e-200                     # When none of the alternatives are available
    p[p < 1e-200] = 1e-200                      # As precaution when p is too close to zero
    tgr = choice - np.transpose(p)              # ttgr is nRows x 1
    ttgr = np.multiply(weightsGr, tgr)          # tgr is nRows x 1
    gr = np.tile(ttgr, (1, expVars.shape[0]))   # gr is nRows x nExpVars 
    sgr = np.multiply(np.transpose(expVars),gr) # sgr is nRows x nExpVars 
    hess = np.dot(np.transpose(sgr), sgr)       # hess is nExpVars x nExpVars 
    try:                                        # iHess is nExpVars x nExpVars 
        iHess = np.linalg.inv(hess)             # If hess is non-singular
    except:
        iHess = np.identity(expVars.shape[0])   # If hess is singular
    se = np.sqrt(np.diagonal(iHess))            # se is nExpVars x 1

    return se
    
    
                
# Method that takes as input the results object from the optimizer, and the list
# of variable names, and displays the table of estimation results 

def displayOutput(outputFile, startTime, llEstimation, nIndMods, 
        namesExpVarsIndMod, paramIndMod, stdErrIndMod,
        namesExpVarsMan, param, stdErr): 
            
    # Display screen
    
    for s in range(0, nIndMods):
        print
        print 'Class %d Model: ' %(s + 1)
        print '------------------------------------------------------------------------------'
        print 'Parameter                                            Est         SE     t-stat'
        print '------------------------------------------------------------------------------'
        for k in range(0, len(namesExpVarsMan[s])):
            print '%-45s %10.4f %10.4f %10.4f' %(namesExpVarsMan[s][k], param[s][k], 
                    stdErr[s][k], param[s][k]/stdErr[s][k])
        print '------------------------------------------------------------------------------'

        
    print
    print 'Class Membership Model:'
    print '------------------------------------------------------------------------------'
    print 'Parameter                                            Est         SE     t-stat'
    print '------------------------------------------------------------------------------'
    cParam = 0
    for k in range(0, len(namesExpVarsIndMod)):
        for s in range(1, nIndMods):
            varName = namesExpVarsIndMod[k] + ' (Class %d)' %(s + 1)
            print '%-45s %10.4f %10.4f %10.4f' %(varName, paramIndMod[cParam], 
                    stdErrIndMod[cParam], paramIndMod[cParam]/stdErrIndMod[cParam])
            cParam += 1
    print '------------------------------------------------------------------------------'

    timeElapsed = datetime.now() - startTime
    timeElapsed = (timeElapsed.days * 24.0 * 60.0) + (timeElapsed.seconds/60.0)
    print '\nEstimation time: %.2f minutes\n' %timeElapsed
    print 'Log-likelihood function at convergence for estimation sample: %.2f' %llEstimation


def processData(inds, indID, nClasses, expVarsClassMem, availIndClasses, 
        obsID, altID, choice, availAlts):
    """
    Function that takes the raw data and processes it to construct arrays and matrices that
    are subsequently used during estimation. 
    
    Parameters
    ----------
    inds : 1D numpy array of size nInds (total number of individuals in the dataset).
        Depicts total number of decision-makers in the dataset.    
    indID : 1D numpy array of size nRows.
        The jth element identifies the decision-maker corresponding to the jth row
        in the dataset.
    nClasses : Integer.
        Number of classes to be estimated by the model.
    expVarsClassMem : 2D numpy array of size (nExpVars x nRows).
        The (i, j)th element is the ith explanatory variable entering the class-membership 
        model for the decision-maker corresponding to the jth row in the dataset.
    availIndClasses : 2D numpy array of size (nClasses x nRows).
        Constraints on available latent classes. The (i,j)th element equals 1 if the ith 
        latent class is available to the decision-maker corresponding to the jth row in the 
        dataset, and 0 otherwise.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation.
    altID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which alternative.
    choice : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to alternatives that were 
        chosen during the corresponding observation.
    availAlts : List of size nClasses.
        Determines choice set constraints for each of the classes. The sth element is an 
        array containing identifiers for the alternatives that are available to decision-makers 
        belonging to the sth class.
        
    Returns
    -------
    nInds : Integer.
        Total number of individuals/decision-makers in the dataset.    
    expVarsClassMem : 2D numpy array of size ((nVars * (nClasses - 1)) x (nDms * nClasses)). 
        The (((n - 1) * nClasses) + s)th column of the returned array specifies 
        the utility of the sth class for the nth decision-maker in the dataset.   
    indClassAv : sparse matrix of size ((nInds * nClasses) x nInds).
        The ((n - 1) * nClasses + s, n)th element is 1 if latent class s is avilable
        to individual n, and 0 otherwise. This matrix is useful both for computational purpose,
        and when the analyst may wish to impose constraints on the availability of latent classes
        to different individuals.
    altAv : List of size nClasses. 
        The sth element of which is a sparse matrix of size (nRows x nObs), where the (i, j)th 
        element equals 1 if the alternative corresponding to the ith column in expVarsMan is 
        available to the decision-maker corresponding to the jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element of the returned matrix  is 1 if the alternative corresponding 
        to the ith row in the data file was chosen during observation j, and 0 otherwise.   
    obsAv : sprase matrix of size (nObs x nDms).
        The (i, j)th element of the returned matrix is 1 if observation i corresponds to 
        decision-maker j, and 0 otherwise.
    rowAv : sparse matrix of size (nRows x (nAlts * nDms)).
        The (i, ((n - 1) * nAlts) + j)th element of the returned matrix is 1 if the ith row 
        in the data file corresponds to the jth alternative and the nth decision-maker, 
        and 0 otherwise.  

    
    """ 
    # Class membership model
    nInds = inds.shape[0]
    expVarsClassMem, indClassAv = processClassMem(expVarsClassMem, indID, nClasses, availIndClasses)

    # Class-specific model
    altAvTuple, altChosen, obsAv, rowAv = processClassSpecificPanel(inds, indID, obsID, altID, choice)
    nRows = altID.shape[0]
    nObs = np.unique(obsID).shape[0]

    altAv = []
    for s in range(0, nClasses):
        altAv.append(coo_matrix((imposeCSConstraints(altID, availAlts[s]), 
                (altAvTuple[0], altAvTuple[1])), shape = (nRows, nObs)))
    

    return (nInds, expVarsClassMem, indClassAv, 
            altAv, altChosen, obsAv, rowAv) 
    

def calProb(nClasses, nInds, paramClassMem, expVarsClassMem, indClassAv,
        paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv, indWeights):
    """
    Function that calculates the expectation of the latent variables in E-Step of the 
    EM Algorithm and the value of the log-likelihood function.
    
    Parameters
    ----------
    nClasses : Integer.
        Number of classes to be estimated by the model.
    nInds : Integer.
        Total number of individuals/decision-makers in the dataset.
    paramClassMem : 1D numpy array of size nVars x ( nClasses - 1 ).
        Entails parameters of the class memebrship model, excluding those of the first class.
        It treats the first class as the base class and hence no parameters are estimated
        for this class.
    expVarsClassMem : 2D numpy array of size ((nVars * (nClasses - 1)) x (nDms * nClasses)). 
        The (((n - 1) * nClasses) + s)th column of the returned array specifies 
        the utility of the sth class for the nth decision-maker in the dataset.  
    indClassAv : sparse matrix of size ((nInds * nClasses) x nInds).
        The ((n - 1) * nClasses + s, n)th element is 1 if latent class s is avilable
        to individual n, and 0 otherwise. This matrix is useful both for computational purpose,
        and when the analyst may wish to impose constraints on the availability of latent classes
        to different individuals.
    paramClassSpec : List of size nClasses.
        The jth element is a 1D numpy array containing the parameter estimates associated with 
        the explanatory variables entering the class-specific utilities for the jth latent class.
    expVarsClassSpec : List of size nClasses.
        Entails the utility specification for each of the latent classes.
        The sth element is a 2D numpy array of size (nExpVars x nRows) containing the explanatory 
        variables entering the class-specific utilities for the sth latent class.
        The (i, j)th element of the 2D array denotes the ith explanatory 
        variable entering the utility for the alternative corresponding to the jth row 
        in the data file.
    altAv : List of size nClasses. 
        The sth element of which is a sparse matrix of size (nRows x nObs), where the (i, j)th 
        element equals 1 if the alternative corresponding to the ith column in expVarsClassSpec is 
        available to the decision-maker corresponding to the jth observation, and 0 otherwise.
    altChosen : sparse matrix of size (nRows x nObs).
        The (i, j)th element of the returned matrix  is 1 if the alternative corresponding 
        to the ith row in the data file was chosen during observation j, and 0 otherwise.   
    obsAv : sprase matrix of size (nObs x nDms).
        The (i, j)th element of the returned matrix is 1 if observation i corresponds to 
        decision-maker j, and 0 otherwise.
        
    indWeights : 1D numpy array of size nDms.
        Each element accounts for the associated weight for each individual in the data file
        to cater for the choice based sampling scheme.
        
    Returns
    -------
    weights : 2D numpy array of size (nClasses x nDms).
        The expected value of latent variable for each individual and each of the available
        classes.
    ll : Float.
        The value of log-likelihood.
    """    
    pIndClass = calClassMemProb(paramClassMem, expVarsClassMem, indClassAv).reshape((nClasses, nInds), order = 'F')

    p = calClassSpecificProbPanel(paramClassSpec[0], expVarsClassSpec[0], altAv[0], altChosen, obsAv)
    for s in range(1, nClasses):
        p = np.vstack((p, calClassSpecificProbPanel(paramClassSpec[s], expVarsClassSpec[s], altAv[s], altChosen, obsAv)))
    
    weights = np.multiply(p, pIndClass)
    ll = np.sum(np.multiply(np.log(np.sum(weights, axis = 0)), indWeights))
    weights = np.divide(weights, np.tile(np.sum(weights, axis = 0), (nClasses, 1)))     # nClasses x nInds
    return weights, ll
 
                                                                                                                                                                                                                                                                                                                                                                                     
def emAlgo(outputFilePath, outputFileName, outputFile, nClasses, 
        indID, expVarsClassMem, namesExpVarsClassMem, availIndClasses, 
        obsID, altID, choice, availAlts, expVarsClassSpec, namesExpVarsClassSpec, indWeights):
    """
    Function that implements the EM Algorithm to estimate the desired model specification. 
    
    Parameters
    ----------
    outputFilePath : String.
        File path to where all the output files should be stores.
    outputFileName : String.
        Name without extension that should be given to all output files.    
    outputFile : File.
        A file object to which the output on the display screen is concurrently written.
    nClasses : Integer.
        Number of classes to be estimated by the model.
    indID : 1D numpy array of size nRows.
        The jth element identifies the decision-maker corresponding to the jth row
        in the dataset.
    expVarsClassMem : 2D numpy array of size (nExpVars x nRows).
        The (i, j)th element is the ith explanatory variable entering the class-membership 
        model for the decision-maker corresponding to the jth row in the dataset.
    namesExpVarsClassMem : List of size nExpVars.
        The jth element is a string containing the name of the jth explanatory variable 
        entering the class-membership model.
    availIndClasses : 2D numpy array of size (nClasses x nRows).
        Constraints on available latent classes. The (i,j)th element equals 1 if the ith 
        latent class is available to the decision-maker corresponding to the jth row in the 
        dataset, and 0 otherwise.
        We usually assume that all classes are available to all individuals in the dataset.
    obsID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which observation.
    altID : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to which alternative.
    choice : 1D numpy array of size nRows.
        Identifies which rows in the dataset correspond to alternatives that were 
        chosen during the corresponding observation.
    availAlts : List of size nClasses.
        Determines choice set constraints for each of the classes. The sth element is an 
        array containing identifiers for the alternatives that are available to decision-makers 
        belonging to the sth class.
    expVarsClassSpec : List of size nClasses.
        Entails the utility specification for each of the latent classes.
        The sth element is a 2D numpy array of size (nExpVars x nRows) containing the explanatory 
        variables entering the class-specific utilities for the sth latent class.
        The (i, j)th element of the 2D array denotes the ith explanatory 
        variable entering the utility for the alternative corresponding to the jth row 
        in the data file.
    namesExpVarsClassSpec : List of size nClasses.
        The jth element is a list containing the names of the explanatory variables
        entering the class-specific utilities for the jth latent class.
    indWeights : 1D numpy array of size nDms.
        Each element accounts for the associated weight for each individual in the data file
        to cater for the choice based sampling scheme.
        
    Returns
    -------
    
    """ 
    
    startTime = datetime.now()
    print 'Processing data'
    outputFile.write('Processing data\n')

    inds = np.unique(indID)
    (nInds, expVarsClassMem, indClassAv, 
            altAv, altChosen, obsAv, rowAv) \
            = processData(inds, indID, 
            nClasses, expVarsClassMem[:, ], availIndClasses, 
            obsID, altID,
            choice, availAlts) 

    paramClassMem = np.zeros(expVarsClassMem.shape[0])
    paramClassSpec = []
    paramClassSpec.append(np.array([-1,0,0]))    
    paramClassSpec.append(np.array([-2,0,0,0]))
    paramClassSpec.append(np.array([-15]))
    
    #for s in range(0, nClasses):
    #    param.append(-np.random.rand(expVarsClassSpec[s].shape[0])/10)

    print 'Initializing EM Algorithm...\n'
    outputFile.write('Initializing EM Algorithm...\n\n')
    converged, iterCounter, llOld = False, 0, 0
    while not converged:
        
        # E-Step: Calculate the expectations of the latent variables, using the current 
        # values for the model parameters.
        
        weights, llNew = calProb(nClasses, nInds, paramClassMem, expVarsClassMem, indClassAv,
                paramClassSpec, expVarsClassSpec, altAv, altChosen, obsAv,indWeights)
        
        currentTime = datetime.now().strftime('%a, %d %b %Y %H:%M:%S')
        print '<%s> Iteration %d: %.4f' %(currentTime, iterCounter, llNew)
        outputFile.write('<%s> Iteration %d: %.4f\n' %(currentTime, iterCounter, llNew))

        # M-Step: Use the weights derived in the E-Step to update the model parameters.
        cWeights = np.tile(indWeights[None, :], (nClasses, 1))
        paramClassMem = minimize(wtClassMem, paramClassMem, args = (expVarsClassMem, indClassAv, 
                weights.reshape((nClasses * nInds, 1), order = 'F'), cWeights.reshape((nClasses * nInds, 1), order = 'F')), 
                method = 'BFGS', jac = True, tol = llTol, options = {'gtol': grTol})['x']        
        
        for s in range(0, nClasses):
            cWeights = np.multiply(weights[s, :], indWeights)
            paramClassSpec[s] = minimize(wtLogitPanel, paramClassSpec[s], args = (expVarsClassSpec[s], altAv[s], 
                    cWeights, altAv[s] * obsAv * cWeights[:, None], altChosen, 
                    obsAv, choice), method = 'BFGS', jac = True, tol = llTol, options = {'gtol': grTol})['x']
                    
        converged =  (abs(llNew - llOld) < emTol)
        llOld = llNew
        iterCounter += 1

    # Calculate standard errors
    stdErrIndMod = calStdErrClassMem(paramClassMem, expVarsClassMem, indClassAv,
                                     weights.reshape((nClasses * nInds, 1), order = 'F'))
    
    stdErr = []
    for s in range(0, nClasses):
        stdErr.append(calStdErrWtLogitPanel(paramClassSpec[s], expVarsClassSpec[s], altAv[s], 
                    weights[s, :], altAv[s] * obsAv * weights[s, :][:, None], 
                    altChosen, obsAv, choice))

    # Print results
                    
    displayOutput(outputFile, startTime, llNew, nClasses, 
            namesExpVarsClassMem, paramClassMem,stdErrIndMod,
            namesExpVarsClassSpec, paramClassSpec, stdErr) 

    # Write parameters to file and store them in an outputfile for the user

    with open(outputFilePath + outputFileName + 'Param.txt', 'wb') as f:                        
        for s in range(0, nClasses):
            np.savetxt(f, paramClassSpec[s][None, :], delimiter = ',')
        np.savetxt(f, paramClassMem[None, :], delimiter = ',')
                

      
# Entry point to script

if __name__ == "__main__":
    
    # Output file 
    
    outputFilePath = ('C:/Users/Feras/Desktop/LCCM Package/')
    outputFileName = 'ModelResults' 
    outputFile = open(outputFilePath + outputFileName + 'Log.txt', 'w')

    # Input parameters from the data file
    
    inputFilePath = 'C:/Users/Feras/Desktop/LCCM Package/'
    inputFileName = 'TrainingData.txt'

    emTol = 1e-04
    llTol = 1e-06
    grTol = 1e-06
    maxIters = 10000

    print '\nReading %s' %inputFileName 
    outputFile.write('\nReading %s\n' %inputFileName )
    data = np.loadtxt(open(inputFilePath + inputFileName, 'rb'), delimiter='\t')


    # Class-membership model: 
    # The first step is to specify the number of latent classes and to identify the column 
    # in the data file denoting the individual IDs. 
    
    nClasses = 3
    
    # accounting for weights - choice-based sampling (Moshe & Lerman)
    # Weighted Exogenous Sample Maximum Likelihood (WESML)
    weightAdopters = 0.0003853/0.404
    weightNonAdopters = 0.9996147/0.596    
    indWeightsA = np.repeat(weightAdopters, 300)
    indWeightsNA = np.repeat(weightNonAdopters,201 )        
    indWeights = np.hstack((indWeightsA,indWeightsNA))  

    # Utility for each of the latent classes is specified by creating matrices 
    # expVarsClassMem, of size (nExpVars x nRows). The (i, j)th element of the matrix denotes 
    # the ith explanatory variable entering the utility for the decision-maker corresponding 
    # to the jth row in the data file.
    
    
    hhIncome = data[:, 5]
    hhIncome = hhIncome / 1000

    male = (data[:, 6] == 1).astype(int)
    
    expVarsClassMem = np.vstack((np.ones(data.shape[0]),hhIncome, male))
    
    namesExpVarsClassMem = ['Class-specific constant','Monthly Income (1000s $)', 'male' ]
            

    # Constraints on available latent classes are imposed through the variable availClasses, 
    # a 2D array of size (nClasses x nRows), where the (i,j)th element equals 1 if the ith 
    # latent class is available to the decision-maker corresponding to the jth row in the 
    # dataset, and 0 otherwise.
    
    availIndClasses = np.vstack((np.logical_not(hhIncome < 0).astype(int), 
                              np.logical_not(hhIncome < 0).astype(int),np.logical_not(hhIncome < 0).astype(int)))
    

    # Class-specific choice model
    # The first step is to identify key columns in the data file denoting the individual,
    # observation, alternatives and choices.
    
    indID = data[:, 0]    
    obsID = data[:, 2]
    altID = data[:, 1]
    choice = np.reshape(data[:, 3], (data.shape[0], 1))
    
    # Choice set constraints are imposed through the variable availAlts, a list of 
    # size nClasses, where the sth element is an array containing identifiers for the 
    # alternatives that are available to decision-makers belonging to the sth class.
        
    availAlts = [np.array([0, 1]), 
                    np.array([0, 1]), np.array([0, 1])]    
    
    # Utility for each of the classes is specified by creating a list expVarsClassSpec, of size 
    # nClasses, where the sth element is a matrix of size (nExpVars x nRows) containing
    # the explanatory variables entering the class-specific utilities for the sth 
    # latent class. The (i, j)th element of the matrix denotes the ith explanatory 
    # variable entering the utility for the alternative corresponding to the jth row 
    # in the data file.
    
    expVarsClassSpec, namesExpVarsClassSpec = [], []

    altcarshare = (altID == 1).astype(int)
    altnocarshare = (altID == 0).astype(int)  
    
    zipInd = data[:,4]
    googleDummy = data[:,9]
    
    
    stationDummy = data[:,8]
    adopters = data[:,7]
    
    accessibility = data[:,10]    
    
    stationDummyNeeded = altID * stationDummy
    adoptersNeeded = altID * adopters
    googleDummyNeeded = altID * googleDummy 
    accessibilityNeeded = altID * accessibility
    
    expVarsClassSpec.append(np.vstack(( altcarshare, accessibilityNeeded,  googleDummyNeeded)))    
    namesExpVarsClassSpec.append(['ASC (CarShare)','Accessibility', 'Google Employee'])
    
    expVarsClassSpec.append(np.vstack((altcarshare,  accessibilityNeeded, adoptersNeeded, googleDummyNeeded)))  
    namesExpVarsClassSpec.append(['ASC (CarShare)', 'Accessibility', 'Cumulative Adopters (t-1)', 'Google Employee']) 
    
            
    expVarsClassSpec.append(np.vstack((altcarshare)).transpose())   
    namesExpVarsClassSpec.append(['ASC (CarShare)'])
            
  
    # Parameter Estimation
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        emAlgo(outputFilePath, outputFileName, outputFile, nClasses, 
                indID, expVarsClassMem, namesExpVarsClassMem, availIndClasses,
                obsID, altID, choice, availAlts, expVarsClassSpec, namesExpVarsClassSpec, indWeights)
        outputFile.close()


