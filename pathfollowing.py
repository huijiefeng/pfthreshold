import numpy as np
import pandas as pd
from scipy.stats import norm

Gkernel = norm.pdf

def Gkernel_loss(t):
    return 1 - norm.cdf(t)

def _SubGrad(dat,delta,coeff,kernel,weight):
    X,Y,Z = dat['X'],dat['Y'],dat['Z']
    n = Z.shape[0]
    K = kernel(Y*(X - Z.dot(coeff))/delta)
    return (weight*Y*K).reshape(1,-1).dot(Z).flatten()/(n*delta)

def _SubOpti(lbda,coeff,grad):
    subopti = np.zeros(coeff.shape[0])
    # 0 coeff and lambda is less than abs of grad, so two options to get minumum
    idx_0 = np.minimum((coeff == 0), (np.abs(grad) >= lbda))
    temp = np.minimum(np.abs(grad[idx_0] + lbda),np.abs(grad[idx_0] - lbda))
    subopti[idx_0] = temp
    #nonzero coeff, has deterministic entry
    idx_1 = coeff != 0
    subopti[idx_1] = np.abs(grad[idx_1] + lbda*np.sign(coeff[idx_1]))
    return np.max(subopti)

def _SoftTreshold(lbda,eta,coeff,grad):
    coeff_new = np.zeros(coeff.shape[0])
    coeff_bar = coeff - eta*grad
    idx = np.abs(coeff_bar) > lbda*eta
    coeff_new[idx] = np.sign(coeff_bar[idx])*(np.abs(coeff_bar)[idx] - lbda*eta)
    return coeff_new

def ProxGrad(dat,delta,weight,coeff,kernel,epsilon,lbda,eta,maxiter = 10000,stage = 'NA'):
    #initialization
    grad = _SubGrad(dat,delta,coeff,kernel,weight)
    subopti = _SubOpti(lbda,coeff,grad)
    idx = 0
    while subopti > epsilon and idx < maxiter:
        idx+=1
        coeff = _SoftTreshold(lbda,eta,coeff,grad)
        grad = _SubGrad(dat,delta,coeff,kernel,weight)
        subopti = _SubOpti(lbda,coeff,grad)
    if idx == maxiter:
        print('maximum iteration reached with sub-optimality {:.6f} at stage {}'.format(subopti,stage))
    return coeff

def PathFollowing(dat,delta,kernel,lbda_tgt,epsilon_final = 0.0001,stages = 10 ,nu = 0.25 ,eta = 0.5,
                  true_coeff = None,maxiter_it = 1000,maxiter_final = 50000):
    """
     Path-following algorithm for high-dimensional threshold estimation with l1 regularization
     Parameters
     ----------
        dat : dict
            A python dictictionary that contains three numpy arrays ('X','Y','Z')
        delta : float
            Bandwidth
        kernel: object
            A kernel function
        lbda_tgt: float
            The tuing parameter for the l1 penalty
        epsilon_final: float
            The precision of the optimization at the final stages (default 0.0001)
        stages: int
            The number of stages for the path-following algorithm (default 10)
        nu: float
            The precision for the middle stages of the path-following algorithm (default 0.25)
        eta: float
            The learning rate (default 0.5)
        true_coeff numpy.ndarray (optional)
            The true coefficient when using simulated dataset
        maxiter_it: int
            The maximum number of iterations for the middle stages (default 1000)
        maxiter_final: int
            The maximum number of iterations for the final stage (default 50000)
     Returns
     ---------
        numpy.ndarray
            The fitted coefficients.
    """
    if 'true_coeff' in dat and dat['true_coeff'] is not None:
        true_coeff = dat['true_coeff']
    n = dat['X'].shape[0]
    d = dat['Z'].shape[1]
    P = np.mean(dat['Y'] == 1)
    
    #initialization
    weight = np.repeat(1/P,n)
    weight[dat['Y'] == -1] = 1/(1-P)
    coeff = np.zeros(d)
    lbda = np.max(np.abs(_SubGrad(dat,delta,coeff,kernel,weight)))
    phi = (lbda_tgt/lbda)**(1.0/stages)
    #print('progress: ',end = '')
    for stage in range(stages):
        #print('{}=>'.format(stage),end = '')
        lbda*=phi
        epsilon = nu*lbda
        coeff = ProxGrad(dat,delta,weight,coeff,kernel,epsilon,lbda,eta,maxiter = maxiter_it,stage= str(stage))
    #print('final')
    coeff = ProxGrad(dat,delta,weight,coeff,kernel,epsilon_final,lbda,eta,maxiter = maxiter_final,stage = 'Final')
    return coeff

##################################
####### cross calidation #########
##################################
def CVSplit(dat,fold = 5):
    n = dat['X'].shape[0]
    cut = int((1.0*n)/fold)
    for i in range(fold):
        idx = np.ones(n)
        idx[list(range(i*cut,(i+1)*cut))] = 0
        yield ({'X': dat['X'][np.bool_(idx)],
                'Y': dat['Y'][np.bool_(idx)],
                'Z': dat['Z'][np.bool_(idx),:],
                'true_coeff' : dat['true_coeff']},
              {'X': dat['X'][np.invert(np.bool_(idx))],
                'Y': dat['Y'][np.invert(np.bool_(idx))],
                'Z': dat['Z'][np.invert(np.bool_(idx)),:],
                'true_coeff' : dat['true_coeff']}
              )


def CVChoose(delta_seq,tuningC_seq,cvoutput):
    CV_mean,CV_std = cvoutput
    min_idx = np.unravel_index(CV_mean.argmin(), CV_mean.shape)
    print(min_idx)
    cv_min_output = (delta_seq[min_idx[0]],tuningC_seq[min_idx[1]])
    upper = np.min(CV_mean) + CV_std[min_idx]
    #print(upper)
    print(CV_mean <= upper)
    onestd_idx_0 = max([i for i in range(CV_mean.shape[0]) if any(CV_mean[i,:] <= upper)])
    onestd_idx_1 = max([j for j in range(CV_mean.shape[1]) if CV_mean[onestd_idx_0,j] <= upper])
    cv_onestd_output = (delta_seq[onestd_idx_0],tuningC_seq[onestd_idx_1])
    return cv_min_output,cv_onestd_output


def CVPathFollowing(dat,delta_seq,kernel,epsilon_final,stages,nu,eta,tuningC_seq,
                  true_coeff = None,maxiter_it = 1000,maxiter_final = 50000,
                  fold = 5,eta_growth = 1,loss = None):
    CV_mean = [[0]*len(tuningC_seq) for _ in range(len(delta_seq))]
    CV_std = [[0]*len(tuningC_seq) for _ in range(len(delta_seq))]
    for i,delta in enumerate(delta_seq):
        for j,tuningC in enumerate(tuningC_seq):
            eta_curr = eta*eta_growth**min(i,j)
            rst = []
            for dat_cv_train,dat_cv_test in CVSplit(dat):
                n,d = dat_cv_train['X'].shape[0],dat_cv_train['Z'].shape[1]
                lbda_tgt = tuningC*np.sqrt(np.log(d)/(n*delta))
                coeff = PathFollowing(dat_cv_train,delta,kernel,epsilon_final,stages,nu,eta_curr,lbda_tgt,
                  true_coeff = true_coeff,maxiter_it = maxiter_it,maxiter_final = maxiter_final)
                # use classification error as cv error
                X,Y,Z = dat_cv_test['X'],dat_cv_test['Y'],dat_cv_test['Z']
                if loss:
                    rst.append(np.mean(loss(Y*(X - Z.dot(coeff))/delta)))
                else:
                    rst.append(1 - np.mean(Y == X - Z.dot(coeff)))
            rst = np.array(rst)
            CV_mean[i][j] = np.nanmean(rst)
            CV_std[i][j] = np.nanstd(rst)
    return np.array(CV_mean),np.array(CV_std)

