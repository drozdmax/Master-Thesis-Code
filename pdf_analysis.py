from scipy import stats, signal, optimize
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tnrange, tqdm
from tqdm.contrib import tzip

def get_pdf(data, lim=None, own_pdf_calc=False):
    '''
    Calculates the propability density function of given data up to x=lim
    Parameters:
        data
        lim: integer: limit on x-axis for pdf calculation. If not given maximum of data will be taken
    Returns
        x,y: x and y values of calculated pdf
    '''
    
    if own_pdf_calc:
        print('Own PDF Calculation Used')
        if lim==None:
            lim = int(max(data))
    
    
        x = np.linspace(0, lim, int(lim/4))
        y = np.zeros_like(x)
        
        for val in data:
            y += 1/np.sqrt(2*np.pi*val) * np.exp(-(x-val)**2/(2*val))
        y /= len(data)
    
    else:
    
        if lim==None:
            lim = int(max(data))
        
        
        x = np.linspace(0, lim, int(lim/4))
        y = stats.gaussian_kde(data).pdf(x)
    
    return x,y


def fit_function(pdfs, *alphas):
    '''
    Returns sum(alpha_i * pdf_i) with pdf_i denoting the probability density function of oligomeric state i and alpha_i the fraction of that state
    Parameters:
        pdfs: list of np.arrays
        alphas: list of floats
    Returns:
        y
    '''
    n = len(pdfs)
    max_x = len(pdfs[-1][0])

    pdy = [pd[1] for pd in pdfs]
    res = 1 - np.sum(alphas)

    alpha_end = 1-np.sum(alphas)
    
    pda = []
    for pd, a in zip(pdfs[:-1], alphas):
        temp = [a*y for y in pd[1]]
        pda.append(temp)
    pda.append([alpha_end*y for y in pdfs[-1][1]])

    y = [sum(i) for i in zip(*pda)]
        
    return y

def random_sampling(data, perc=0.5):
    '''
    randomly samples subset of given data. Length of sampled subsets is determined by given fraction (perc).
    '''
    num = int(np.ceil(perc*len(data)))
    subset = data.sample(n=num)
    
    return subset


def get_pdfs(toccsl_data, sm_data=None, sm_start_frame=None, rec_frame=5, lim=4000, lim_con=2, sampled_sm=False, perc=0.5, own_pdf_calc=False):
    '''
    Returns probability density functions of TOCCSL data in first recovery frame and probabilty density function of single molecule data (sm pdf).
    If dataset sm_data is given sm pdf will be calculated from this dataset
    If dataset sm_data is not given sm pdf will be calculated from the TOCCSL dataset either starting at given sm_start_frame or considering the last frame only.
    Parameters:
        toccsl_data: pd.DataFrame
        sm_data: pd.DataFrame
        rec_frame: int; specifies frist recovery frame in TOCCSL data
        lim: limit of pdfs on x axis
        lim_con: int; limits the number of convolutions, i.e. the number of oligomeric states to consider
        sampled_sm: if True only a subset of the data available to generate the monomeric pdf will be used (size of subset specfied by perc)
        perc: fraction of data available to generate the monomeric pdf that will be used if sampled_sm is True
    Returns:
        pdfs: list of pdf_i (i from 1 to lim_con) with pdf_i = (x_i, y_i)
        pdf_data: pdf from first recovery image of TOCCSL data with pdf_data = (x_data, y_data)
    '''
    pdfs = []
    
    if sm_data is not None:
        data_rho1 = sm_data['mass']
    elif sm_data == None and sm_start_frame == None:
        last_frame = max(toccsl_data['frame'])
        data_rho1 = toccsl_data[toccsl_data['frame']==last_frame]['mass']
        if len(data_rho1) < 10:
            raise Exception(f'Not enough data in last frame! Only {len(data_rho1)} datapoint(s) found! \n' + 'Choose sm_start_frame to include more frames instead!')
    else:
        data_rho1 = toccsl_data[toccsl_data['frame']>sm_start_frame]['mass']
        
    if sampled_sm == True:
        data_rho1 = random_sampling(data_rho1, perc=perc)
    
    data_rhodata = toccsl_data[toccsl_data['frame']==rec_frame]['mass']

    x1, y1 = get_pdf(data_rho1, lim=lim, own_pdf_calc=own_pdf_calc)
    xdata, ydata = get_pdf(data_rhodata, lim=lim, own_pdf_calc=own_pdf_calc)
    pdf_data = (xdata, ydata)
    pdfs.append((x1, y1))
    
    for i in range(1,lim_con):#, desc='convolutions'):
        y_temp = signal.convolve(pdfs[i-1][1], y1, mode='full', method='auto')/sum(pdfs[i-1][1])
        ###### make all pdfs same size by cutting off at lim
        pdfs.append((x1,y_temp[:len(x1)]))
    
    
    return pdfs, pdf_data


def get_results(toccsl_data, sm_data=None, sm_start_frame=None, rec_frame=5, lim=4000, lim_con=2, iterations=100, perc=0.5, save_path=os.getcwd() + '\\data\\TOCCSL_results_n{}.png', own_pdf_calc=False):
    '''
    Creates probabilty density functions, fits given data, does bootstrapping and plots final results.
    If dataset sm_data is given sm pdf will be calculated from this dataset
    If dataset sm_data is not given sm pdf will be calculated from the TOCCSL dataset either starting at given sm_start_frame or considering the last frame only.
    Parameters:
        toccsl_data: pd.DataFrame
        sm_data: pd.DataFrame
        rec_frame: int; specifies frist recovery frame in TOCCSL data
        lim: limit of pdfs on x axis
        lim_con: int; limits the number of convolutions, i.e. the number of oligomeric states to consider
        iterations: number of iterations used for bootstrapping (defaults to 100)
        perc: fraction of data available to generate the monomeric pdf that will be used for bootstrapping
        save_path: formatable string specifying path to save final plot to. Will be formated with lim_con
    Returns:
        alphas: list containing fraction of each oligomeric structure
        fitted_pd: fitted y data
        means: list containing means from bootstrapping
        SEMs: list containing SEMs from bootstrapping
        fig: final plot
    '''
    bias = 0.001
    min_ = 0.0001
    
    ####### Get results using all data
    ### Generate PDFs
    start = time.time()
    pdfs, pdf_data = get_pdfs(toccsl_data=toccsl_data, sm_data=sm_data, sm_start_frame=sm_start_frame, rec_frame=rec_frame, lim=lim, lim_con=lim_con, own_pdf_calc=own_pdf_calc)
    ### Fit making sure no alpha_i is samller than zero
    alphas = [1]*(lim_con-1) + [-1]
    while alphas[-1] < 0:
        bounds_up = [a-bias for a in alphas[:-1]]
        if alphas[-1]>min_:
            bounds_up = [a if a>0 else a-min_ for a in bounds_up]
        else:
            bounds_up = [a if a>0 else min_ for a in bounds_up]
        fit, cov = optimize.curve_fit(fit_function, xdata=pdfs, ydata=list(pdf_data[1]), p0=bounds_up, bounds=([0]*(lim_con-1), bounds_up))
        alphas = list(fit) + [1-np.sum(fit)]
    
    fitted_pd = fit_function(pdfs, *fit)
    end = time.time()
    print('Initial analysis completed - Elapsed time: {}s'.format(end-start))
    
    
    ####### Estimate error by bootstrapping
    start = time.time()
    alphas_lbs = []
    for n in range(lim_con):
        alphas_lbs.append([])
    
    for i in tqdm(range(iterations), desc='bootstrap iterations'):
        ### Generates PDFs
        pdfs_s, pdf_data_s = get_pdfs(toccsl_data=toccsl_data, sm_data=sm_data, sm_start_frame=sm_start_frame, rec_frame=rec_frame, lim=lim, lim_con=lim_con, sampled_sm=True, perc=perc, own_pdf_calc=own_pdf_calc)
        ### Fit making sure no alpha_i is samller than zero
        alphas_bs = [1]*(lim_con-1) + [-1]
        while alphas_bs[-1] < 0:
            bounds_up = [a-bias for a in alphas_bs[:-1]]
            if alphas_bs[-1]>min_:
                bounds_up = [a if a>0 else a-min_ for a in bounds_up]
            else:
                bounds_up = [a if a>0 else min_ for a in bounds_up]
            fit, cov = optimize.curve_fit(fit_function, xdata=pdfs_s, ydata=list(pdf_data_s[1]), p0=bounds_up, bounds=([0]*(lim_con-1), bounds_up))
            alphas_bs = list(fit) + [1-np.sum(fit)]
    
    
        for n in range(lim_con):
            alphas_lbs[n].append(alphas_bs[n])
    end = time.time()
    print('Bootstrap completed - Elapsed time: {}s'.format(end-start))
    
    SEMs = []
    means = []
    for n in range(lim_con):
        cur = alphas_lbs[n]
        means.append(np.mean(cur))
        SEMs.append(np.std(cur)/np.sqrt(len(cur)))
    
    
    ########### Final plot
    text = []
    for n in range(lim_con):
        text.append(r'$\rho_{}$'.format(n+1) + ': ({}'.format(round(alphas[n]*100,2)) + ' $\pm$ ' + '{})%'.format(round(SEMs[n]*100,2)))

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(pdf_data[0], pdf_data[1], label='data')
    ax.plot(pdf_data[0], fitted_pd, label='fit')
    for a, pd, t in zip(alphas, pdfs, text):
        ax.plot(pd[0], [a*y for y in pd[1]], label=t)
    ax.legend()
    ax.set_xlabel('brightness [counts]')
    ax.set_ylabel('PDF')
    fig.savefig(save_path.format(lim_con))
    
    
    return alphas, fitted_pd, means, SEMs, fig