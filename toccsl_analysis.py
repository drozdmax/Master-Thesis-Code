from scipy import stats, signal, optimize, interpolate
from math import log10, floor
import numpy as np
import pandas as pd
import time
import os
import pims
import matplotlib.pyplot as plt
from tqdm.notebook import tnrange, tqdm
from tqdm.contrib import tzip
from sdt import roi, io, multicolor
from sm_handling import get_sm_dist
import trackpy
from helpers import getfiles, remove_empty_panda

# The functions get_pdf_local, model_pdf and get_pdfs_local were written by Marina Bishara and at most slightly adapted
# The function get_alphas was written by Marina Bishara and heavily adapted by Maximilian Roy Drozd
# The remaining functions were written by Maximilian Roy Drozd (generate_pdf, plot_fraction_development, pdf_dependence_rois, pdf_dependence_startframe, plot_fraction_rois, plot_fraction_frames, fraction_rois_brightness, get_track_data, track_length_data, fit_survival_probability, get_alphas_bleaching, fraction_rois_bleaching, plot_fraction_development_brightness)

def generate_pdf(data, sigma_factor=1.0, fit_lognorm=False):
    '''
    Generates a probability density function as a sum of gaussians for each data point:
    
    Parameters:
        data: randomly sampled data points of the PDF
        sigma_factor: controls the width of the contributing Gaussian curves
        fit_lognorm: bool that determines wether the generated pdf is fitted to a lognormal form
    Returns:
        f: resulting probability density function
    '''
    
    if fit_lognorm:
        lognorm_fit = stats.lognorm.fit(data, floc=0)
        return lambda x: stats.lognorm.pdf(x, *lognorm_fit)
    
    def f(x):
        y = np.zeros_like(x)
        for i in range(len(x)):
            y[i] = np.sum(1/(sigma_factor*np.sqrt(2*np.pi*data)) * np.exp(-(x[i]-data)**2/(2*data*sigma_factor**2))) / len(data)
        return y
    
    return f

def get_pdf_local(data, lim=None, sigma_factor=None, fit_lognorm=False):
    '''
    Calculates the propability density function of given data up to x=lim
    
    Parameters:
        data: randomly sampled data points of the PDF
        sigma_factor: controls the width of the contributing Gaussian curves
        fit_lognorm: bool that determines wether the generated pdf is fitted to a lognormal form
    Returns:
        x,y: x and y values of calculated pdf
    '''        
    
    if sigma_factor==None:
        print('Careful, no sigma_factor given to function.')
    if lim==None:
        lim = int(max(data))
    
    if fit_lognorm:
        lognorm_fit = stats.lognorm.fit(data, floc=0)
        x = np.linspace(0, lim, 2*lim)
        return x, stats.lognorm.pdf(x, *lognorm_fit)
    
    x = np.linspace(0, lim, 2*lim)
    y = np.array([np.sum(1/(sigma_factor*np.sqrt(2*np.pi*data)) * np.exp(-(x[i]-data)**2/(2*data*sigma_factor**2))) / len(data) for i in range(len(x))])
        
    return x,y


def model_pdf(pdfs, *alphas):
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


def get_pdfs_local(analysis_data_photons, monomer_data_photons, lim=4000, lim_con=2, sigma_factor=None, fit_lognorm=False, method='RA'):
    '''
    Returns probability density functions of analysis data (analysis_data_photons) and probabilty density function of monomer data (monomer_data_photons).
    
    Parameters:
        analysis_data_photons: numpy.ndarray; array of photon numbers
        monomer_data_photons: pandas.DataFrame; photon numbers accessible under the keyword 'mass'
        lim: limit of pdfs on x axis
        lim_con: int; limits the number of convolutions, i.e. the number of oligomeric states to consider
        sigma_factor: controls the width of the Gaussian curves contributing to the PDFs
        fit_lognorm: bool that determines wether the monomer pdf is fitted to a lognormal form
        method: 'RA' or 'MD'; determines wether random addition (RA) or multiplied distributions (MD) assumptions are used for summing brightness values
    Returns:
        pdfs: list of pdf_i (i from 1 to lim_con) with pdf_i = (x_i, y_i)
        pdf_data: pdf from first recovery image of TOCCSL data with pdf_data = (x_data, y_data)
    '''
    pdfs = []
    
    data_rho1 = monomer_data_photons['mass']
    
    data_rhodata = analysis_data_photons

    x1, y1 = get_pdf_local(data_rho1, lim=lim, sigma_factor=sigma_factor, fit_lognorm=fit_lognorm)
    xdata, ydata = get_pdf_local(data_rhodata, lim=lim, sigma_factor=sigma_factor)
    pdf_data = (xdata, ydata)
    pdfs.append((x1, y1))
    
    for i in range(1,lim_con):#, desc='convolutions'):
        if method=='RA':
            y_temp = signal.convolve(pdfs[i-1][1], y1, mode='full', method='auto')/sum(pdfs[i-1][1])
            ###### make all pdfs same size by cutting off at lim
            pdfs.append((x1,y_temp[:len(x1)]))
        elif method=='MD':
            x_temp, y_temp = get_pdf_local((i+1)*data_rho1, lim=lim, sigma_factor=sigma_factor, fit_lognorm=fit_lognorm)
            pdfs.append((x_temp,y_temp))
    
    
    return pdfs, pdf_data


def get_alphas(analysis_data_photons, monomer_data_photons, lim=1000, lim_con=2, n_bootstrap=400, sigma_factor=1.0, fit_lognorm=False, bins=30, savepath=None, donotplot=False):
    '''
    Creates probabilty density functions, fits given brightness data via a maximum likelihood extimator, and returns final oligomeric fraction results for a set of brightness value samples, with associated errors determined by the nonparametric bootstrap. The result is returned for both random addition (RA) and multiplied distributions (MD) assumptions for the summation of fluorophore brightness values.
    
    Parameters:
        analysis_data_photons: numpy.ndarray; array of photon numbers
        monomer_data_photons: pandas.DataFrame; photon numbers should be accessible under the keyword 'mass'
        lim: limit of pdfs on x axis (defaults to 1000)
        lim_con: int; the number of oligomeric states to consider, starting from a monomer (defaults to 2)
        n_bootstrap: number of iterations used for bootstrapping (defaults to 400)
        sigma_factor: controls the width of the Gaussian curves contributing to the PDFs (defaults to 1.0)
        fit_lognorm: bool that determines wether the monomer pdf is fitted to a lognormal form (defaults to False)
        bins: number of bins between 0 and lim that the analysed data is displayed at in the final plot (defaults to 30)
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
        donotplot: if set to True, no plot is generated (defaults to False)
    Returns:
        alphas: array containing the fraction of each oligomeric component, with RA assumptions for index 0, MD assumptions for index 1
        errors: 1 sigma errors corresponding to the values in alphas
        fitted_pd: fitted y data
        kstests: Kolmogorov-Smirnov test statistic and associated p-value of the analysed data and the PDF fits corresponding to the values in alpha
    '''
    full_start = time.time()
    n_samples = len(analysis_data_photons)
    bias = 0.001
    min_ = 0.0001
    # Constraint for fit
    cons=({'type': 'ineq',
    'fun': lambda x: 1.0-np.sum(x)})
    
    if lim==None:
        lim = int(max(analysis_data_photons))
    
    ####### Get results using all data
    ### Generate PDFs
    pdfs_MD, pdf_data_MD = get_pdfs_local(analysis_data_photons, monomer_data_photons, lim=lim, lim_con=lim_con, sigma_factor=sigma_factor, fit_lognorm=fit_lognorm, method='MD')
    
    ### Fit making sure no alpha_i is samller than zero
    alphas_MD = [1]*(lim_con-1) + [-1]
    
    bounds_up = [a-bias for a in alphas_MD[:-1]]
    if alphas_MD[-1]>min_:
        bounds_up = [a if a>0 else a-min_ for a in bounds_up]
    else:
        bounds_up = [a if a>0 else min_ for a in bounds_up]
    # Generate model function for the PDF
    model_pdf_MD = lambda x, alphas: interpolate.interp1d(pdf_data_MD[0], model_pdf(pdfs_MD, *alphas), kind='cubic')(x[x<=lim])
    # Initializing the negative log of the likelihood function to be minimized
    likelihood_estimator = lambda alphas: -1.0*np.sum(np.log(model_pdf_MD(analysis_data_photons, alphas)))
    # Performing the fit
    start = time.time()
    res = optimize.minimize(likelihood_estimator, np.ones(lim_con-1)/lim_con, bounds=tuple(zip([0]*(lim_con-1), bounds_up)), constraints=cons)
    end = time.time()
    if res.success:
        print('Optimization MD: Success after', end-start, 's')
    else:
        print('Optimization MD: Failure after', end-start, 's')
    fit_MD = res.x
    alphas_MD = list(fit_MD) + [1-np.sum(fit_MD)]
    if alphas_MD[-1]<0 and alphas_MD[-1]>-0.000001:
        alphas_MD[-1] = 0
    
    samples = analysis_data_photons[analysis_data_photons<=lim]
    statistic_value = np.max(np.abs(np.arange(len(samples))/len(samples) - np.cumsum(model_pdf_MD(np.sort(samples), fit_MD))/np.sum(model_pdf_MD(samples, fit_MD))))
    p_value = np.exp(-2*len(samples)*statistic_value**2)
    kstest_MD = np.array([statistic_value, p_value])
    
    # Perform bootstrapping
    start = time.time()
    alphas_bootstrap_MD = []
    for i in range(n_bootstrap):
        analysis_data_sample = analysis_data_photons.sample(n_samples, replace=True)
        likelihood_estimator = lambda alphas: -1.0*np.sum(np.log(model_pdf_MD(analysis_data_sample, alphas)))
        res = optimize.minimize(likelihood_estimator, fit_MD, bounds=tuple(zip([0]*(lim_con-1), bounds_up)), constraints=cons)
        fit_temp = res.x
        alphas_temp = list(fit_temp) + [1-np.sum(fit_temp)]
        alphas_bootstrap_MD.append(alphas_temp)
    alphas_bootstrap_MD = np.array(alphas_bootstrap_MD)
    errors_MD = np.zeros_like(alphas_MD)
    for i in range(len(errors_MD)):
        errors_MD[i] = np.std(alphas_bootstrap_MD[:,i])
    end = time.time()
    print('Bootstrapping MD took', end-start, 's')
    
    fitted_pd_MD = model_pdf(pdfs_MD, *fit_MD)
    
    
    ### Generate PDFs
    pdfs_RA, pdf_data_RA = get_pdfs_local(analysis_data_photons, monomer_data_photons, lim=lim, lim_con=lim_con, sigma_factor=sigma_factor, fit_lognorm=fit_lognorm, method='RA')
    
    ### Fit making sure no alpha_i is samller than zero
    alphas_RA = [1]*(lim_con-1) + [-1]
    
    bounds_up = [a-bias for a in alphas_RA[:-1]]
    if alphas_RA[-1]>min_:
        bounds_up = [a if a>0 else a-min_ for a in bounds_up]
    else:
        bounds_up = [a if a>0 else min_ for a in bounds_up]
    # Generate model function for the PDF
    model_pdf_RA = lambda x, alphas: interpolate.interp1d(pdf_data_RA[0], model_pdf(pdfs_RA, *alphas), kind='cubic')(x[x<=lim])
    # Initializing the negative log of the likelihood function to be minimized
    likelihood_estimator = lambda alphas: -1.0*np.sum(np.log(model_pdf_RA(analysis_data_photons, alphas)))
    # Performing the fit
    start = time.time()
    res = optimize.minimize(likelihood_estimator, np.ones(lim_con-1)/lim_con, bounds=tuple(zip([0]*(lim_con-1), bounds_up)), constraints=cons)
    end = time.time()
    if res.success:
        print('Optimization RA: Success after', end-start, 's')
    else:
        print('Optimization RA: Failure after', end-start, 's')
    fit_RA = res.x
    alphas_RA = list(fit_RA) + [1-np.sum(fit_RA)]
    if alphas_RA[-1]<0 and alphas_RA[-1]>-0.000001:
        alphas_RA[-1] = 0
    
    samples = analysis_data_photons[analysis_data_photons<=lim]
    statistic_value = np.max(np.abs(np.arange(len(samples))/len(samples) - np.cumsum(model_pdf_RA(np.sort(samples), fit_RA))/np.sum(model_pdf_RA(samples, fit_RA))))
    p_value = np.exp(-2*len(samples)*statistic_value**2)
    kstest_RA = np.array([statistic_value, p_value])
    
    # Perform bootstrapping
    start = time.time()
    alphas_bootstrap_RA = []
    for i in range(n_bootstrap):
        analysis_data_sample = analysis_data_photons.sample(n_samples, replace=True)
        likelihood_estimator = lambda alphas: -1.0*np.sum(np.log(model_pdf_RA(analysis_data_sample, alphas)))
        res = optimize.minimize(likelihood_estimator, fit_RA, bounds=tuple(zip([0]*(lim_con-1), bounds_up)), constraints=cons)
        fit_temp = res.x
        alphas_temp = list(fit_temp) + [1-np.sum(fit_temp)]
        alphas_bootstrap_RA.append(alphas_temp)
    alphas_bootstrap_RA = np.array(alphas_bootstrap_RA)
    errors_RA = np.zeros_like(alphas_RA)
    for i in range(len(errors_RA)):
        errors_RA[i] = np.std(alphas_bootstrap_RA[:,i])
    end = time.time()
    print('Bootstrapping RA took', end-start, 's')
    
    fitted_pd_RA = model_pdf(pdfs_RA, *fit_RA)
    

    text_MD = []
    for n in range(lim_con):
        text_MD.append(r'$\rho_{}$ MD'.format(n+1) + ': ({}'.format(round(alphas_MD[n]*100, 1-int(floor(log10(abs(errors_MD[n]*100)))))) + ' $\pm$ {})%'.format(round(errors_MD[n]*100, 1-int(floor(log10(abs(errors_MD[n]*100)))))))
    text_RA = []
    for n in range(lim_con):
        text_RA.append(r'$\rho_{}$ RA'.format(n+1) + ': ({}'.format(round(alphas_RA[n]*100, 1-int(floor(log10(abs(errors_RA[n]*100)))))) + ' $\pm$ {})%'.format(round(errors_RA[n]*100, 1-int(floor(log10(abs(errors_RA[n]*100)))))))
    
    alphas = [alphas_MD, alphas_RA]
    errors = [errors_MD, errors_RA]
    fitted_pd = [fitted_pd_MD, fitted_pd_RA]
    kstests = [kstest_MD, kstest_RA]
    
    full_end = time.time()
    print('Completed all after', full_end-full_start, 's\n')
    
    if donotplot:
        return alphas, errors, fitted_pd, kstests
        
    
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(constrained_layout=True)
    
    ax.hist(analysis_data_photons, bins=bins, range=(0, lim), color='skyblue')
    ax.plot(pdf_data_MD[0], len(analysis_data_photons)*(lim/bins)*np.array(fitted_pd_MD), label='Fit MD')
    ax.plot(pdf_data_RA[0], len(analysis_data_photons)*(lim/bins)*np.array(fitted_pd_RA), label='Fit RA')
    for a, pd, t in zip(alphas_MD, pdfs_MD, text_MD):
        ax.plot(pd[0], len(analysis_data_photons)*(lim/bins)*np.array([a*y for y in pd[1]]), label=t)
    for a, pd, t in zip(alphas_RA, pdfs_RA, text_RA):
        ax.plot(pd[0], len(analysis_data_photons)*(lim/bins)*np.array([a*y for y in pd[1]]), label=t)
    ax.legend()
    ax.set_xlabel('Number of photons')
    ax.set_ylabel('PDF')
    
    if savepath != None:
        fig.savefig(savepath)
    plt.rcParams.update({'font.size': 10})
    
    return alphas, errors, fitted_pd, kstests


def plot_fraction_development(filtered_data, monomer_photons, starting_frame, iterations, lim=1000, lim_con=2, n_bootstrap=400, sigma_factor=1.0, fit_lognorm=False,  savepath=None):
    """
    Calculates the oligomeric fraction development using the brightness based analysis over multiple frames of a set of measurements using the get_alphas function and graphically plots the numerical results returned. Also plots the number of localizations in the set of brightness values to be analyzed, corresponding to each starting frame.
    
    Parameters:
        analysis_data_photons: pandas.DataFrame; photon numbers should be accessible under the keyword 'mass'. The location in time, i.e. the corresponding frame, should be stored under the keyword 'frame'
        monomer_data_photons: pandas.DataFrame; photon numbers should be accessible under the keyword 'mass'
        starting_frame: determines from which frame the oligomeric fractions should start being calculated, the first frame having the index 0
        iterations: sets for how many frames succeeding starting_frame the oligomeric fraction is calculated
        lim: limit of pdfs on x axis (defaults to 1000)
        lim_con: int; the number of oligomeric states to consider, starting from a monomer (defaults to 2)
        n_bootstrap: number of iterations used for bootstrapping (defaults to 400)
        sigma_factor: controls the width of the Gaussian curves contributing to the PDFs (defaults to 1.0)
        fit_lognorm: bool that determines wether the monomer pdf is fitted to a lognormal form (defaults to False)
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
    Returns:
        alphas: list containing fraction of each oligomeric component, with RA assumptions for index 0, MD assumptions for index 1. The next index dimension corresponds to the frame, starting from start_frame
        errors: 1 sigma errors corresponding to the values in alphas
        kstests: Kolmogorov-Smirnov test statistic and associated p-value of the analysed data and the PDF fits corresponding to the values in alpha
    """
    alphas_MD = []
    errors_MD = []
    alphas_RA = []
    errors_RA = []
    kstests_MD = []
    kstests_RA = []
    for i in range(iterations):
        analysis_data = filtered_data[filtered_data['frame']==starting_frame+i]['mass']
        alphas, errors, fitted_pd, kstests = get_alphas(analysis_data, monomer_photons, lim=lim, lim_con=lim_con, n_bootstrap=n_bootstrap, sigma_factor=sigma_factor, fit_lognorm=fit_lognorm, donotplot=True)
        alphas_MD.append(alphas[0])
        errors_MD.append(errors[0])
        alphas_RA.append(alphas[1])
        errors_RA.append(errors[1])
        kstests_MD.append(kstests[0])
        kstests_RA.append(kstests[1])
    fig, ax = plt.subplots(constrained_layout=True)
    labels_MD = []
    labels_RA = []
    i = 1
    for alpha in alphas[0]:
        labels_MD.append(r'$\alpha_{}$ MD'.format(i))
        labels_RA.append(r'$\alpha_{}$ RA'.format(i))
        i += 1
    for i in range(len(alphas[0])):
        ax.errorbar(np.arange(iterations)+starting_frame+1, np.array(alphas_MD)[:,i]*100, yerr=np.array(errors_MD)[:,i]*100, marker='o', capsize=5, label=labels_MD[i])
        ax.errorbar(np.arange(iterations)+starting_frame+1, np.array(alphas_RA)[:,i]*100, yerr=np.array(errors_RA)[:,i]*100, marker='o', capsize=5, label=labels_RA[i])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fraction [%]')
    ax.legend()
    if savepath != None:
        fig.savefig(savepath)
    
    alphas = [np.array(alphas_MD), np.array(alphas_RA)]
    errors = [np.array(errors_MD), np.array(errors_RA)]
    kstests = [kstests_MD, kstests_RA]
    
    return alphas, errors, kstests


def pdf_dependence_rois(monomer_folder, roi_center, roi_radii, conversion_factor=1.0, px_size=0.16, sigma_factor=1.0, fit_lognorm=False, monomer_start_frame=0, lim=1000, frames_taken=None, savepath=None):
    """
    Plots in the same figure PDFs calculated using differently sized regions of interest (ROIs), where all ROIs are circular with the same center. The numbers of localizations associated to each PDF are also given in the plot.
    
    Parameters:
        monomer_folder: string; specifies in which folder the localizations of the measurements to be analyzed are stored
        roi_center: sets the central point of all used ROIs
        roi_radii: numpy.ndarray; array of the radii of the ROIs for which the oligomeric fractions are to be calculated, values should be given in micrometers
        conversion_factor: sets the conversion factor between the brightness values extracted from the localizations (usually given in counts) and the number of photons (defaults to 1.0)
        px_size: size of a pixel in micrometers (defaults to 0.16)
        sigma_factor: controls the width of the Gaussian curves contributing to the PDFs (defaults to 1.0)
        fit_lognorm: bool that determines wether the monomer pdf is fitted to a lognormal form (defaults to False)
        monomer_start_frame: sets from which frame signals should start to be taken from the localizations in monomer_folder (defaults to 0)
        lim: limit of pdfs on x axis (defaults to 1000)
        frames_taken: sets how many frames should be taken to gather signals, starting from monomer_start_frame. Defaults to None, in which case all subsequent frames are taken
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
    Returns:
        None
    """
    monomer_pdfs = []
    number_signals = []
    
    for radius in roi_radii:
        axis = radius/px_size
        set_roi = roi.EllipseROI(roi_center, (axis, axis))
        monomer_data = get_sm_dist(monomer_folder, start_frame=0, int_roi=set_roi)
        monomer_data = monomer_data[monomer_data['frame']>=monomer_start_frame]
        if frames_taken != None:
            monomer_data = monomer_data[monomer_data['frame']<monomer_start_frame+frames_taken]
        monomer_pdfs.append(generate_pdf(monomer_data['mass']*conversion_factor, sigma_factor=sigma_factor,  fit_lognorm=fit_lognorm))
        number_signals.append(len(monomer_data['mass']))
    
    plot_vals = np.linspace(0, lim, 200)
        
    fig, ax = plt.subplots(constrained_layout=True)
    for  monomer_pdf, radius, number in zip(monomer_pdfs, roi_radii, number_signals):
        ax.plot(plot_vals, monomer_pdf(plot_vals), label=str(radius)+' $\mu$m (' + str(number) + ' signals)')
    ax.set_xlabel('Number of photons')
    ax.set_ylabel('PDF')
    if len(roi_radii) <= 15:
        ax.legend()
    else:
        ax.legend(fontsize=10*15/len(roi_radii))
    if savepath != None:
        fig.savefig(savepath)


def pdf_dependence_startframe(monomer_folder, roi_center, startframes, conversion_factor=1.0, px_size=0.16, sigma_factor=1.0, fit_lognorm=False, monomer_roi=5.0, lim=1000, frames_taken=None, savepath=None):
    """
    Plots in the same figure PDFs calculated using different starting frames from which localizations are gathered, with identical regions of interest (ROIs). The numbers of localizations associated to each PDF are also given in the plot.
    
    Parameters:
        monomer_folder: string; specifies in which folder the localizations of the measurements to be analyzed are stored
        roi_center: sets the central point of all used ROIs
        startframes: numpy.ndarray; array of the startframes for which the oligomeric fractions are to be calculated, the first frame having the index 0
        conversion_factor: sets the conversion factor between the brightness values extracted from the localizations (usually given in counts) and the number of photons (defaults to 1.0)
        px_size: size of a pixel in micrometers (defaults to 0.16)
        sigma_factor: controls the width of the Gaussian curves contributing to the PDFs (defaults to 1.0)
        fit_lognorm: bool that determines wether the monomer pdf is fitted to a lognormal form (defaults to False)
        monomer_roi: sets the radius of the ROI which is used to gather localizations (defaults to 5.0)
        lim: limit of pdfs on x axis (defaults to 1000)
        frames_taken: sets how many frames should be taken to gather signals, starting from monomer_start_frame. Defaults to None, in which case all subsequent frames are taken
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
    Returns:
        None
    """
    monomer_pdfs = []
    number_signals = []
    
    axis = monomer_roi/px_size
    set_roi = roi.EllipseROI(roi_center, (axis, axis))
    
    for startframe in startframes:
        monomer_data = get_sm_dist(monomer_folder, start_frame=startframe, int_roi=set_roi)
        monomer_data = monomer_data[monomer_data['frame']>=startframe]
        if frames_taken != None:
            monomer_data = monomer_data[monomer_data['frame']<startframe+frames_taken]
        monomer_pdfs.append(generate_pdf(monomer_data['mass']*conversion_factor, sigma_factor=sigma_factor, fit_lognorm=fit_lognorm))
        number_signals.append(len(monomer_data['mass']))
    
    plot_vals = np.linspace(0, lim, 200)
    
    fig, ax = plt.subplots(constrained_layout=True)
    for  monomer_pdf, startframe, number in zip(monomer_pdfs, startframes, number_signals):
        ax.plot(plot_vals, monomer_pdf(plot_vals), label='Frame '+str(startframe)+' (' + str(number) + ' signals)')
    ax.set_xlabel('Number of photons')
    ax.set_ylabel('PDF')
    if len(startframes) <= 15:
        ax.legend()
    else:
        ax.legend(fontsize=10*15/len(startframes))
    if savepath != None:
        fig.savefig(savepath)


def plot_fraction_rois(alphas, errors, labels, roi_radii, savepath=None):
    """
    Nicely plots the analysis results for different region of interest (ROI) sizes, for given results.
    
    Parameters:
        alphas: list of arrays of fraction results to be plotted
        errors: list of 1 sigma errors corresponding to the fraction results given in alphas
        labels: list of labels assigned to each of the arrays given in alphas
        roi_radii: numpy.ndarray; array of ROI radii in micrometers to which each element in the list alphas is plotted
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
    Returns:
        None
    """
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(9,6))
    for alpha, error, label in zip(alphas, errors, labels):
        for i in range(len(alpha[0])):
            ax.errorbar(roi_radii, alpha[:,i]*100, yerr=error[:,i]*100, marker='o', capsize=5, label=label[i], linewidth=2)
    ax.set_xlabel('ROI [$\mu$m]')
    ax.set_ylabel('Fraction [%]')
    #ax.legend()
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.83)
    if savepath != None:
        fig.savefig(savepath, dpi=150)
    plt.rcParams.update({'font.size': 10})

        
def plot_fraction_frames(alphas, errors, labels, frames, savepath=None):
    """
    Nicely plots the analysis results for different analysis frames, for given results.
    
    Parameters:
        alphas: list of arrays of fraction results to be plotted
        errors: list of 1 sigma errors corresponding to the fraction results given in alphas
        labels: list of labels assigned to each of the arrays given in alphas
        roi_radii: numpy.ndarray; array of analysis frames to which each element in the list alphas is plotted
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
    Returns:
        None
    """
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(figsize=(9,6))
    for alpha, error, label in zip(alphas, errors, labels):
        for i in range(len(alpha[0])):
            ax.errorbar(frames, alpha[:,i]*100, yerr=error[:,i]*100, marker='o', capsize=5, label=label[i], linewidth=2)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fraction [%]')
    #ax.legend()
    fig.legend(loc=7)
    fig.tight_layout()
    fig.subplots_adjust(right=0.83)
    if savepath != None:
        fig.savefig(savepath, dpi=150)
    plt.rcParams.update({'font.size': 10})


def fraction_rois_brightness(folder_analysis, monomer_folder, frame, roi_center, roi_radii, monomer_roi=None, conversion_factor=1.0, px_size=0.16, sigma_factor=1.0, fit_lognorm=False, monomer_start_frame=0, frames_taken=None, lim=1000, lim_con=2, n_bootstrap=400, savepath=None):
    """
    Calculates the oligomeric fraction using the brightness based analysis for different sizes of circular regions of interest (ROIs) with the same center using the get_alphas function and graphically plots the numerical results returned. Also plots the number of localizations in the set of brightness values to be analyzed, corresponding to each ROI radius.
    
    Parameters:
        folder_analysis: string; specifies in which folder the localizations of the measurements to be analyzed are stored
        monomer_folder: string; specifies in which folder the localizations of the measurements of monomers are stored
        frame: determines in which frame the oligomeric fractions will calculated, the first frame having the index 0
        roi_center: sets the central point of all used ROIs
        roi_radii: numpy.ndarray; array of the radii of the ROIs for which the oligomeric fractions are to be calculated, values should be given in micrometers
        monomer_roi: sets the radius in micrometers for the ROI to be used for the monomer measurements. Defaults to None, in which case for each calculation the same ROI is used for the monomer as for the analysis values.
        conversion_factor: sets the conversion factor between the brightness values extracted from the localizations (usually given in counts) and the number of photons (defaults to 1)
        px_size: size of a pixel in micrometers (defaults to 0.16)
        sigma_factor: controls the width of the Gaussian curves contributing to the PDFs (defaults to 1.0)
        fit_lognorm: bool that determines wether the monomer pdf is fitted to a lognormal form (defaults to False)
        monomer_start_frame: sets from which frame monomer signals should start to be taken from the localizations in monomer_folder (defaults to 0)
        frames_taken: sets how many frames should be taken to gather monomer signals, starting from monomer_start_frame. Defaults to None, in which case all subsequent frames are taken
        lim: limit of pdfs on x axis (defaults to 1000)
        lim_con: int; the number of oligomeric states to consider, starting from a monomer (defaults to 2)
        n_bootstrap: number of iterations used for bootstrapping (defaults to 400)
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
    Returns:
        alphas: list containing fraction of each oligomeric component, with RA assumptions for index 0, MD assumptions for index 1. The next index dimension corresponds to the ROI radius, with the same order as the radii in roi_radii
        errors: 1 sigma errors corresponding to the values in alphas
        kstests: Kolmogorov-Smirnov test statistic and associated p-value of the analysed data and the PDF fits corresponding to the values in alpha
    """
    alphas_MD = []
    errors_MD = []
    alphas_RA = []
    errors_RA = []
    kstests_MD = []
    kstests_RA = []
    number_monomer = []
    if monomer_roi != None:
        roi_monomer = roi.EllipseROI(roi_center, (monomer_roi/px_size, monomer_roi/px_size))
    for radius in roi_radii:
        axis = radius/px_size
        set_roi = roi.EllipseROI(roi_center, (axis, axis))
        analysis_data = get_sm_dist(folder_analysis, start_frame=0, int_roi=set_roi)
        analysis_data = analysis_data[analysis_data['frame']==frame]['mass']
        if monomer_roi == None:
            monomer_data = get_sm_dist(monomer_folder, start_frame=0, int_roi=roi_monomer)
        else:
            monomer_data = get_sm_dist(monomer_folder, start_frame=0, int_roi=roi_monomer)
        monomer_data = monomer_data[monomer_data['frame']>=monomer_start_frame]
        if frames_taken != None:
            monomer_data = monomer_data[monomer_data['frame']<monomer_start_frame+frames_taken]
        alphas, errors, fitted_pd, kstests = get_alphas(analysis_data*conversion_factor, monomer_data*conversion_factor, lim=lim, lim_con=lim_con, n_bootstrap=400, sigma_factor=sigma_factor, fit_lognorm=fit_lognorm, donotplot=True)
        number_monomer.append(len(analysis_data))
        alphas_MD.append(alphas[0])
        errors_MD.append(errors[0])
        alphas_RA.append(alphas[1])
        errors_RA.append(errors[1])
        kstests_MD.append(kstests[0])
        kstests_RA.append(kstests[1])
    
    labels_MD = []
    labels_RA = []
    i = 1
    for alpha in alphas[0]:
        labels_MD.append(r'$\alpha_{}$ MD'.format(i))
        labels_RA.append(r'$\alpha_{}$ RA'.format(i))
        i += 1
    
    fig, ax = plt.subplots(1, 2, figsize=(14,6), constrained_layout=True)
    for i in range(len(alphas[0])):
        ax[0].errorbar(roi_radii, np.array(alphas_MD)[:,i]*100, yerr=np.array(errors_MD)[:,i]*100, marker='o', capsize=5, label=labels_MD[i])
        ax[0].errorbar(roi_radii, np.array(alphas_RA)[:,i]*100, yerr=np.array(errors_RA)[:,i]*100, marker='o', capsize=5, label=labels_RA[i])
    ax[0].set_xlabel('ROI [$\mu$m]')
    ax[0].set_ylabel('Fraction [%]')
    ax[0].legend()
    ax[1].plot(roi_radii, np.array(number_monomer))
    ax[1].set_xlabel('ROI [$\mu$m]')
    ax[1].set_ylabel('Number of signals')
    if savepath != None:
        fig.savefig(savepath)
    
    alphas = [np.array(alphas_MD), np.array(alphas_RA)]
    errors = [np.array(errors_MD), np.array(errors_RA)]
    kstests = [kstests_MD, kstests_RA]
    
    return alphas, errors, kstests


def get_track_data(analysis_folder, first_frame, max_displacement=4, allowed_gap_size=2, crop_roi=None):
    """
    Finds the tracks from the localizations found in analysis_folder using the trackpy.link function, and returns them.
    
    Parameters:
        analysis_folder: string; specifies in which folder the localizations of the measurements to be analyzed are stored
        first_frame: specifies from which frame on tracks should be searched for, the first frame having the index 0
        max_displacement: the maximum displacement in pixels between two frames that will still be linked to a track (defaults to 4)
        allowed_gap_size: the number of frames a particle is allowed to not be localized and still be linked to a single track (defaults to 2)
        crop_roi: sdt.roi object; initially applied to the localizations before linking them to tracks. Defaults to None, in which case nothing is done and all localizations are used
    Returns:
        trc_data: pandas.DataFrame; dataframe giving the details of all the tracks isolated from the localizations
    """
    trc_data = []
    h5_files = getfiles(extension='h5', folder=analysis_folder)
    for h5 in tqdm(h5_files, desc='files', leave=False):
        loc_data = io.load(h5)
        if crop_roi != None:
            loc_data = crop_roi(loc_data)
        loc_data = loc_data[loc_data['frame']>=first_frame]
        tracked = trackpy.link(loc_data, search_range=max_displacement, memory=allowed_gap_size)
        trc_data.append(tracked)
    remove_empty_panda(trc_data)
    
    return trc_data


def track_length_data(trc_data, analysis_roi, px_size, roi_center, normalized_profile, first_frame):
    """
    Isolates and returns from the dataframe giving the detailed track information, specific information that is of interest in the determination of the oligomeric composition using the bleaching behaviour. In particular, an array of the track lengths, the tracks normalized average laser intensity exposure, and the development of the total intensity of the tracks normalized to the initial intensity, are returned.
    
    Parameters:
        trc_data: pandas.DataFrame; dataframe giving track details
        analysis_roi: radius of the region of interest in which the localization of a track has to be in the frame first_frame to be considered in the analysis
        px_size: size of a pixel in micrometers
        roi_center: center of the circular region of interest in which the localization of a track has to be in the frame first_frame to be considered in the analysis
        normalized_profile: normalized intensity profile of the laser illumination
        first_frame: frame which is chosen as the first frame, i.e. all earlier frames are disregarded. The first frame has the index 0
    Returns:
        track_lengths: array of the lengths of all the tracks considered in the analysis
        probability_scaling: average normalised laser intensity experienced by each track considered in the analysis
        survival_probability: sum of the brightness of each track considered in the analysis in each frame, starting from first_frame
    """
    axis = analysis_roi/px_size
    
    track_lengths = []
    total_intensity = np.zeros(500)
    probability_scaling = []
    
    for tracks in trc_data:
        tracks = tracks[tracks['frame']>=first_frame]
        for particle_num in np.unique(tracks['particle']):
            temp_particle_data = tracks[tracks['particle']==particle_num]
            if first_frame in np.array(temp_particle_data['frame']):
                
                if ((np.array(temp_particle_data['x'])[0]-roi_center[0])**2+(np.array(temp_particle_data['y'])[0]-roi_center[1])**2)>axis**2:
                    continue
                
                temp_particle_data_frames = np.array(temp_particle_data['frame'])
                track_lengths.append(max(temp_particle_data_frames)-first_frame)
                
                temp_positions = tuple([round(temp_particle_data['y']).astype(int), round(temp_particle_data['x']).astype(int)])
                temp_probability_scaling = np.mean(normalized_profile[temp_positions])
                probability_scaling.append(temp_probability_scaling)
                
                corrected_frames = temp_probability_scaling*temp_particle_data_frames - 0.5
                total_intensity[corrected_frames.astype(int)] += temp_particle_data['mass']*temp_probability_scaling
                #total_intensity[corrected_frames.astype(int)] += (np.ceil(corrected_frames)-corrected_frames) * temp_particle_data['mass']*temp_probability_scaling * (corrected_frames>first_frame)
                #total_intensity[corrected_frames.astype(int)+1] += (corrected_frames-np.floor(corrected_frames)) * temp_particle_data['mass']*temp_probability_scaling
        
    
    track_lengths = (np.array(track_lengths).reshape(-1)).astype(float)
    probability_scaling = np.array(probability_scaling)
    survival_probability = total_intensity[first_frame:]/total_intensity[first_frame]
    
    return track_lengths, probability_scaling, survival_probability


def fit_survival_probability(survival_probability, ignore0=False, plotfigure=False, savepath=None, p0=None):
    """
    Performs a double exponential fit to an array using the scipy.curve_fit function, for the purpose of determining the monomer bleaching curve from the intensity decay data, and returns the fit parameters.
    
    Parameters:
        survival_probability: data to which a double exponential curve shall be fitted
        ignore0: bool; determines wether the initial value is disregarded in the fit (defaults to False)
        plotfigure: bool; determines wether the fit results should be plotted (defaults to False)
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
        p0: tuple which determines the initial parameter guess (A, alpha, beta) where A is the fraction of the first exponential species with decay constant alpha, and beta the decay constant of the other species. Defaults to None, in which case a default initial guess is used
    Returns:
        intensity_lam: array of the fit results in the form [A, alpha, beta] where A is the fraction of the first exponential species with decay constant alpha, and beta the decay constant of the other species
        intensity_lam_err: the 1 sigma fit error corresponding to the values given in intensity_lam
    """
    if 0 in survival_probability:
        plot_lim = (survival_probability==0).argmax(axis=0)
        survival_probability = survival_probability[:plot_lim]
    
    lam_fit_func = lambda x, f1, lam1, lam2: f1*np.exp(-lam1*x) + (1.0-f1)*np.exp(-lam2*x)
    
    x_vals = np.arange(len(survival_probability))
    y_vals = survival_probability
    if p0 == None:
        p0 = (0.5, 0.3, 0.07)
    if ignore0:
        x_vals=x_vals[1:]; y_vals=y_vals[1:]
        lam_fit_func = lambda x, I0, f1, lam1, lam2: I0*(f1*np.exp(-lam1*x) + (1.0-f1)*np.exp(-lam2*x))
        intensity_lam, intensity_lam_cov = optimize.curve_fit(lam_fit_func, x_vals, y_vals, p0=(1.0, *p0), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf]))
        intensity_lam_orig = intensity_lam
        intensity_lam = intensity_lam[1:]
        intensity_lam_cov = intensity_lam_cov[1:,1:]
    else:
        intensity_lam, intensity_lam_cov = optimize.curve_fit(lam_fit_func, x_vals, y_vals, p0=p0, bounds=([0.0, 0.0, 0.0], [1.0, np.inf, np.inf]))
        intensity_lam_orig = intensity_lam
    
    result_errors = np.sqrt(np.diag(intensity_lam_cov))
    if result_errors[0]>np.min(np.array([1-intensity_lam[0], intensity_lam[0]])):
        result_errors[0] = np.min(np.array([1-intensity_lam[0], intensity_lam[0]]))
    if result_errors[1]>intensity_lam[1]:
        result_errors[1] = intensity_lam[1]
    if result_errors[2]>intensity_lam[2]:
        result_errors[2] = intensity_lam[2]
    
    if not plotfigure:
        intensity_lam_err = np.sqrt(np.diag(intensity_lam_cov))
        return intensity_lam, intensity_lam_err
    
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(np.arange(len(survival_probability)), survival_probability, label='Data')
    ax.plot(np.arange(len(survival_probability)), lam_fit_func(np.arange(len(survival_probability)), *intensity_lam_orig), label='Fit')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Survival Probability')
    ax.legend()
    
    if savepath != None:
        fig.savefig(savepath)
    
    intensity_lam_err = np.sqrt(np.diag(intensity_lam_cov))
    
    return intensity_lam, intensity_lam_err
    

def get_alphas_bleaching(track_lengths, probability_scaling, lam_fit, lam_fit_err, lim_con=2, n_bootstrap=400, ignore0=False, bins=None, savepath=None, donotplot=False):
    """
    Calculates bleaching curves corresponding to higher order oligomers from the monomer curve, fits given tracklength data via a maximum likelihood extimator, and returns final oligomeric fraction results for a set of brightness value samples, with associated errors determined by the nonparametric bootstrap and variation of the monomer bleaching curve parameter errors.
    
    Parameters:
        track_lengths: array of the lengths of tracks to be analyzed
        probability_scaling: average normalised laser intensity experienced by the tracks to be analyzed
        lam_fit: array of the biexponential fit results of the monomer bleaching curve in the form [A, alpha, beta] where A is the fraction of the first exponential species with decay constant alpha, and beta the decay constant of the other species
        lam_fit_err: the 1 sigma fit error corresponding to the values given in lam_fit
        lim_con: int; the number of oligomeric states to consider, starting from a monomer (defaults to 2)
        n_bootstrap: number of iterations used for bootstrapping (defaults to 400)
        ignore0: bool; determines wether the first frame is disregarded in the analysis (defaults to False)
        bins: the number of bins the track length samples are distributed into. Defaults to None, in which case each length in frames gets its own bin
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
        donotplot: if set to True, no plot is generated (defaults to False)
    Returns:
        alphas: array containing the fraction of each oligomeric component
        errors: 1 sigma errors corresponding to the values in alphas
    """
    lam_fit_func = lambda x: lam_fit[0]*np.exp(-lam_fit[1]*x) + (1.0-lam_fit[0])*np.exp(-lam_fit[2]*x)
    
    def P_oligomer(n, p_scaling, order, ignore0=False):
        result = (1.0-lam_fit_func(p_scaling*(n+1.0)))**order - (1.0-lam_fit_func(p_scaling*n))**order
        if ignore0:
            result /= (1.0 - P_oligomer(np.zeros_like(n), p_scaling, order))
        return result
    
    def P(n, p_scaling, lim_con, alphas):
        result = 0.0
        for i in range(lim_con-1):
            result += alphas[i]*P_oligomer(n, p_scaling, i+1)
        result += (1.0-np.sum(alphas))*P_oligomer(n, p_scaling, lim_con)
        return result
    
    def P_ignoring0(n, p_scaling, lim_con, alphas):
        return np.sign(n)*P(n, p_scaling, lim_con, alphas)/(1.0-P(np.zeros_like(n), p_scaling, lim_con, alphas))
    
    def P_model(n, lim_con, alphas, ignore0=False):
        if isinstance(n, int):
            n = [n]
        if ignore0:
            results = [P_ignoring0(n_temp, 1.0, lim_con, alphas) for n_temp in n]
        else:
            results = [P(n_temp, 1.0, lim_con, alphas) for n_temp in n]
        return np.array(results)
    
    
    probability_scaling_original = probability_scaling.copy()
    track_lengths_original = track_lengths.copy()
    if ignore0:
        probability_scaling = probability_scaling[track_lengths!=0]
        track_lengths = track_lengths[track_lengths!=0]
        likelihood_estimator = lambda x: -1.0*np.sum(np.log(P_ignoring0(track_lengths, probability_scaling, lim_con, x)+1e-25))
    else:
        likelihood_estimator = lambda x: -1.0*np.sum(np.log(P(track_lengths, probability_scaling, lim_con, x)+1e-25))

    cons=({'type': 'ineq',
    'fun': lambda x: 1.0-np.sum(x)})

    # Performing the fit
    start = time.time()
    res = optimize.minimize(likelihood_estimator, np.ones(lim_con-1)/lim_con, bounds=tuple(zip([0]*(lim_con-1), [1]*(lim_con-1))), constraints=cons)
    end = time.time()
    if res.success:
        print('Optimization Success after', end-start, 's')
    else:
        print('Optimization Failure after', end-start, 's')
    fit = res.x
    
    alphas = np.zeros(lim_con)
    alphas[:-1]=fit; alphas[-1]=np.abs(1.0-np.sum(fit))*(np.sum(fit)<1)
    
    
    
    results_bootstrap = []
    for i in range(n_bootstrap):
        idx_sample = np.random.choice(np.arange(len(track_lengths)), size=len(track_lengths))
        sample_tracks = track_lengths[idx_sample]
        sample_pscaling = probability_scaling[idx_sample]
        temp_lam_fit = lam_fit + np.random.normal(size=3)*lam_fit_err
        lam_fit_func = lambda x: temp_lam_fit[0]*np.exp(-temp_lam_fit[1]*x) + (1.0-temp_lam_fit[0])*np.exp(-temp_lam_fit[2]*x)
        temp_likelihood_estimator = lambda x: -1.0*np.sum(np.log(P(sample_tracks, sample_pscaling, lim_con, x)+1e-25))
        temp = optimize.minimize(temp_likelihood_estimator, np.ones(lim_con-1)/lim_con, bounds=tuple(zip([0]*(lim_con-1), [1]*(lim_con-1))), constraints=cons)
        temp_fit = temp.x
        temp_res = np.zeros(lim_con)
        temp_res[:-1]=temp_fit; temp_res[-1]=1.0-np.sum(temp_fit)
        results_bootstrap.append(temp_res)
    results_bootstrap = np.array(results_bootstrap)
    errors = np.zeros_like(alphas)
    for i in range(len(errors)):
        errors[i] = np.std(results_bootstrap[:,i])
    
    if donotplot:
        return alphas, errors
    
    text = []
    for n in range(lim_con):
        text.append(r'P$_{}$'.format(n+1) + ': ({}'.format(round(alphas[n]*100, 1-int(floor(log10(abs(errors[n]*100)))))) + ' $\pm$ {})%'.format(round(errors[n]*100, 1-int(floor(log10(abs(errors[n]*100)))))))
    
    plt.rcParams.update({'font.size': 13})
    fig, ax = plt.subplots(constrained_layout=True)
    
    max_frame = int(max(track_lengths*probability_scaling)+1)
    if bins == None:
        bins = max_frame
    if ignore0:
        plot_positions = np.linspace(0, max_frame, 500)
    else:
        plot_positions = np.linspace(0, max_frame, 500)
    ax.hist(track_lengths_original*probability_scaling_original, bins=bins, range=(-0.5, max_frame-0.5), color='skyblue')
    ax.plot(plot_positions, len(track_lengths_original)*P_model(plot_positions, lim_con, fit, ignore0=False), label='Fit')
    for alpha, t, i in zip(alphas, text, np.arange(lim_con)):
        ax.plot(plot_positions, alpha*len(track_lengths_original)*P_oligomer(plot_positions, 1.0, i+1, ignore0=False), label=t)
        ax.legend()
        ax.set_xlabel('Frames')
        ax.set_ylabel('PDF')
    
    if savepath != None:
        fig.savefig(savepath)
    plt.rcParams.update({'font.size': 10})
    
    return alphas, errors


def fraction_rois_bleaching(folder_analysis, first_frame, normalized_profile, roi_center, roi_radii, px_size=0.16, lim_con=2, n_bootstrap=400, max_displacement=4, allowed_gap_size=2, first_frame_intensityfit=None, savepath=None):
    """
    Calculates the oligomeric fraction using the bleaching based analysis for different sizes of circular regions of interest (ROIs) with the same center using the get_alphas_bleaching function and graphically plots the numerical results returned. Also plots the number of localizations in the set of tracklength values to be analyzed, corresponding to each ROI radius.
    
    Parameters:
        folder_analysis: string; specifies in which folder the localizations of the measurements to be analyzed are stored
        first_frame: determines in which frame the oligomeric fractions will calculated, the first frame having the index 0
        normalized_profile: normalized intensity profile of the laser illumination
        roi_center: sets the central point of all used ROIs
        roi_radii: numpy.ndarray; array of the radii of the ROIs for which the oligomeric fractions are to be calculated, values should be given in micrometers
        px_size: size of a pixel in micrometers (defaults to 0.16)
        lim_con: int; the number of oligomeric states to consider, starting from a monomer (defaults to 2)
        n_bootstrap: number of iterations used for bootstrapping (defaults to 400)
        max_displacement: the maximum displacement in pixels between two frames that will still be linked to a track (defaults to 4)
        allowed_gap_size: the number of frames a particle is allowed to not be localized and still be linked to a single track (defaults to 2)
        first_frame_intensityfit: first frame from which the determination of the monomer bleaching curve shall be carried out. Defaults to None, in which case the value of first_frame is taken
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
    Returns:
        alphas: list containing fraction of each oligomeric component, with standard analysis for index 0, and ignoring tracks of length 0 for index 1. The next index dimension corresponds to the ROI radius, with the same order as the radii in roi_radii
        errors: 1 sigma errors corresponding to the values in alphas
    """
    alphas_w0 = []
    errors_w0 = []
    alphas_n0 = []
    errors_n0 = []
    number_tr_w0 = []
    number_tr_n0 = []
    
    trc_data = get_track_data(folder_analysis, first_frame, max_displacement=max_displacement, allowed_gap_size=allowed_gap_size)
    if (first_frame_intensityfit != None) and (first_frame != first_frame_intensityfit): # Does not work good for later frames
        trc_data_intensityfit = get_track_data(folder_analysis, first_frame_intensityfit, max_displacement=max_displacement, allowed_gap_size=allowed_gap_size)
    else:
        trc_data_intensityfit = trc_data
        first_frame_intensityfit = first_frame
    temp1, temp2, survival_probability = track_length_data(trc_data, roi_radii[-1], px_size, roi_center, normalized_profile, first_frame)
    #lam_fit_w0, lam_fit_err_w0 = fit_survival_probability(survival_probability, ignore0=False, plotfigure=False)
    #lam_fit_n0, lam_fit_err_n0 = fit_survival_probability(survival_probability, ignore0=True, plotfigure=False)
    
    for radius in roi_radii:
        axis = radius/px_size
        set_roi = roi.EllipseROI(roi_center, (axis, axis))
        track_lengths, probability_scaling, survival_probability = track_length_data(trc_data, radius, px_size, roi_center, normalized_profile, first_frame)
        if first_frame != first_frame_intensityfit:
            garbage1, garbage2, survival_probability = track_length_data(trc_data_intensityfit, radius, px_size, roi_center, normalized_profile, first_frame_intensityfit) # Does not work good for later frames
        lam_fit_w0, lam_fit_err_w0 = fit_survival_probability(survival_probability, ignore0=False, plotfigure=False)
        lam_fit_n0, lam_fit_err_n0 = fit_survival_probability(survival_probability, ignore0=True, plotfigure=False, p0=tuple(lam_fit_w0))

        alphas_temp_w0, errors_temp_w0 = get_alphas_bleaching(track_lengths, probability_scaling, lam_fit_w0, lam_fit_err_w0, lim_con=lim_con, n_bootstrap=n_bootstrap, ignore0=False, donotplot=True)
        alphas_temp_n0, errors_temp_n0 = get_alphas_bleaching(track_lengths, probability_scaling, lam_fit_n0, lam_fit_err_n0, lim_con=lim_con, n_bootstrap=n_bootstrap, ignore0=True, donotplot=True)
        print()
        number_tr_w0.append(len(track_lengths))
        number_tr_n0.append(len(track_lengths[track_lengths!=0]))
        alphas_w0.append(alphas_temp_w0)
        errors_w0.append(errors_temp_w0)
        alphas_n0.append(alphas_temp_n0)
        errors_n0.append(errors_temp_n0)
    
    labels_w0 = []
    labels_n0 = []
    i = 1
    for alpha in alphas_w0[0]:
        labels_w0.append(r'$\alpha_{}$ BA'.format(i))
        labels_n0.append(r'$\alpha_{}$ BAn0'.format(i))
        i += 1
    
    fig, ax = plt.subplots(1, 2, figsize=(14,6), constrained_layout=True)
    for i in range(len(alphas_w0[0])):
        ax[0].errorbar(roi_radii, np.array(alphas_w0)[:,i]*100, yerr=np.array(errors_w0)[:,i]*100, marker='o', capsize=5, label=labels_w0[i])
        ax[0].errorbar(roi_radii, np.array(alphas_n0)[:,i]*100, yerr=np.array(errors_n0)[:,i]*100, marker='o', capsize=5, label=labels_n0[i])
    ax[0].set_xlabel('ROI [$\mu$m]')
    ax[0].set_ylabel('Fraction [%]')
    ax[0].legend()
    ax[1].plot(roi_radii, np.array(number_tr_w0), label='all tracks')
    ax[1].plot(roi_radii, np.array(number_tr_n0), label='excl. lengths of 0')
    ax[1].set_xlabel('ROI [$\mu$m]')
    ax[1].set_ylabel('Number of tracks')
    ax[1].legend()
    if savepath != None:
        fig.savefig(savepath)
    
    alphas = [np.array(alphas_w0), np.array(alphas_n0)]
    errors = [np.array(errors_w0), np.array(errors_n0)]
    
    return alphas, errors


def plot_fraction_development_brightness(folder_analysis, normalized_profile, roi_center, roi_radius, starting_frame, iterations, px_size=0.16, lim_con=2, max_displacement=4, allowed_gap_size=2, n_bootstrap=400, first_frame_intensityfit=None, savepath=None):
    """
    Calculates the oligomeric fraction development using the bleaching based analysis over multiple frames of a set of measurements using the get_alphas_bleaching function and graphically plots the numerical results returned. Also plots the number of localizations in the set of tracklength values to be analyzed, corresponding to each starting frame.
    
    Parameters:
        folder_analysis: string; specifies in which folder the localizations of the measurements to be analyzed are stored
        normalized_profile: normalized intensity profile of the laser illumination
        roi_center: sets the central point of the used region of interest (ROI)
        roi_radius: sets the radius in micrometers of the ROI to be used
        starting_frame: determines from which frame the oligomeric fractions should start being calculated, the first frame having the index 0
        iterations: sets for how many frames succeeding starting_frame the oligomeric fraction is calculated
        px_size: size of a pixel in micrometers (defaults to 0.16)
        lim_con: int; the number of oligomeric states to consider, starting from a monomer (defaults to 2)
        max_displacement: the maximum displacement in pixels between two frames that will still be linked to a track (defaults to 4)
        allowed_gap_size: the number of frames a particle is allowed to not be localized and still be linked to a single track (defaults to 2)
        n_bootstrap: number of iterations used for bootstrapping (defaults to 400)
        first_frame_intensityfit: first frame from which the determination of the monomer bleaching curve shall be carried out. Defaults to None, in which case the value of the corresponding first analysis frame is taken
        save_path: string specifying path to save final plot to. Defaults to None, in which case nothing is saved
    Returns:
        alphas: list containing fraction of each oligomeric component, with standard analysis for index 0, and ignoring tracks of length 0 for index 1. The next index dimension corresponds to the ROI radius, with the same order as the radii in roi_radii
        errors: 1 sigma errors corresponding to the values in alphas
    """
    alphas_w0 = []
    errors_w0 = []
    alphas_n0 = []
    errors_n0 = []
    
    follow_frames = True
    
    trc_data = get_track_data(folder_analysis, starting_frame, max_displacement=max_displacement, allowed_gap_size=allowed_gap_size)
    if (first_frame_intensityfit != None)  and (first_frame != first_frame_intensityfit): # Does not work good for later frames
        trc_data_intensityfit = get_track_data(folder_analysis, first_frame_intensityfit, max_displacement=max_displacement, allowed_gap_size=allowed_gap_size)
    else:
        trc_data_intensityfit = trc_data
        first_frame_intensityfit = first_frame
        follow_frames = False
    garbage1, garbage2, survival_probability = track_length_data(trc_data_intensityfit, roi_radius, px_size, roi_center, normalized_profile, first_frame_intensityfit) # Does not work good for later frames
    lam_fit_w0, lam_fit_err_w0 = fit_survival_probability(survival_probability, ignore0=False, plotfigure=False)
    lam_fit_n0, lam_fit_err_n0 = fit_survival_probability(survival_probability, ignore0=True, plotfigure=False, p0=tuple(lam_fit_w0))
    
    
    for i in range(iterations):
        for j in range(len(trc_data)):
            trc_data[j] = trc_data[j][trc_data[j]['frame']>=starting_frame+i]
        
        track_lengths, probability_scaling, survival_probability = track_length_data(trc_data, roi_radius, px_size, roi_center, normalized_profile, starting_frame+i)
        if follow_frames:
            lam_fit_w0, lam_fit_err_w0 = fit_survival_probability(survival_probability, ignore0=False, plotfigure=False)
            lam_fit_n0, lam_fit_err_n0 = fit_survival_probability(survival_probability, ignore0=True, plotfigure=False, p0=tuple(lam_fit_w0))

        alphas_temp_w0, errors_temp_w0 = get_alphas_bleaching(track_lengths, probability_scaling, lam_fit_w0, lam_fit_err_w0, lim_con=lim_con, n_bootstrap=n_bootstrap, ignore0=False, donotplot=True)
        alphas_temp_n0, errors_temp_n0 = get_alphas_bleaching(track_lengths, probability_scaling, lam_fit_n0, lam_fit_err_n0, lim_con=lim_con, n_bootstrap=n_bootstrap, ignore0=True, donotplot=True)
        print()
        alphas_w0.append(alphas_temp_w0)
        errors_w0.append(errors_temp_w0)
        alphas_n0.append(alphas_temp_n0)
        errors_n0.append(errors_temp_n0)
    
    labels_w0 = []
    labels_n0 = []
    i = 1
    for alpha in alphas_w0[0]:
        labels_w0.append(r'$\alpha_{}$ BA'.format(i))
        labels_n0.append(r'$\alpha_{}$ BAn0'.format(i))
        i += 1
    
    fig, ax = plt.subplots(constrained_layout=True)
    for i in range(len(alphas_w0[0])):
        ax.errorbar(np.arange(iterations)+starting_frame+1, np.array(alphas_w0)[:,i]*100, yerr=np.array(errors_w0)[:,i]*100, marker='o', capsize=5, label=labels_w0[i])
        ax.errorbar(np.arange(iterations)+starting_frame+1, np.array(alphas_n0)[:,i]*100, yerr=np.array(errors_n0)[:,i]*100, marker='o', capsize=5, label=labels_n0[i])
    ax.set_xlabel('Frame')
    ax.set_ylabel('Fraction [%]')
    ax.legend()
    if savepath != None:
        fig.savefig(savepath)
    
    alphas = [np.array(alphas_w0), np.array(alphas_n0)]
    errors = [np.array(errors_w0), np.array(errors_n0)]
    
    return alphas, errors


def coordinate_correction(beads_folder, dimensions, split_axis=1, donotplot=False):
    """
    The coordinate transformation between two sets of reference image localizations in different colour channels, recorded next to each other, is calculated and a sdt.multicolor.Registrator object returned with which corrections corresponding to this transformation can be made. The results of the overlay of the reference images and the remaining errors are graphically plotted.
    
    Parameters:
        beads_folder: string; sets folder in which the reference image localizations are saved as h5 files
        dimensions: tuple; gives the size in pixels of the area corresponding to one color channel
        split_axis: axis on which the two color channels are located next to each other (defaults to 1)
        donotplot: if set to True, no plot is generated (defaults to False)
    Returns:
        correction_registrator: sdt.multicolor.Registrator object; Registrator object that corresponds to the determined coordinate transformation
    """
    roi_1 = roi.ROI([0, 0], size=dimensions)
    if split_axis == 1:
        roi_2 = roi.ROI([dimensions[0], 0], size=dimensions)
    if split_axis == 2:
        roi_2 = roi.ROI([0, dimensions[1]], size=dimensions)
    
    h5_files = getfiles(extension='h5', folder=beads_folder)

    bead_loc = [io.load(h5) for h5 in tqdm(h5_files, desc='files', leave=False)]
    beads_r1 = [roi_1(b) for b in bead_loc]
    beads_r2 = [roi_2(b) for b in bead_loc]
    
    correction_registrator = multicolor.Registrator(beads_r1, beads_r2)
    correction_registrator.determine_parameters()
    
    if not donotplot:
        correction_registrator.test()  # Plot results
    
    return correction_registrator


def colocalization_analysis(analysis_folder, analysis_frame, dimensions, correction_registrator, split_axis=1, set_roi=None, d_coloc=1, subsequent_recording=False):
    """
    A colocalization analysis is done for localizations present in a given analysis frame for a set of measurements, a sdt.multicolor.Registrator object for coordinate correction is required as input. Returned are the calculated dimer fraction, assuming a population of monomers and dimers with complete labeling and no partial photobleaching, and the fraction of fluorophores belonging to the first channels wavelength.
    
    Parameters:
        analysis_folder: string; the folder in which the localizations to be analyzed can be found in as h5 files
        analysis_frame: frame number which is to be analyzed, with the first frame having the number 0
        dimensions: tuple; gives the size in pixels of the area corresponding to one color channel
        correction_registrator: sdt.multicolor.Registrator object; Registrator object that corresponds to the relevant coordinate transformation
        split_axis: axis on which the two color channels are located next to each other (defaults to 1)
        set_roi: sdt.roi.ROI object; sets a region of interest within the first color channel, which is then also set in the coordinate corrected data of the second color channel. Defaults to None, in which case the entire recorded area of both channels is used
        d_coloc: the distance in pixels between two colocalizations up to which one considers two localizations to colocalize
        subsequent_recording: bool; can be set to True if the recording of channel 2 is delayed by one frame. Defaults to False, in which case the same frame is taken for both channels
    Returns:
        dimer_fraction: fraction of dimers calculated for the assumption of a population of monomers and dimers with complete labeling and no partial photobleaching
        fraction_labeling_1: fraction of fluorophores that are excited by the first channels wavelength for the assumption of a population of monomers and dimers with complete labeling and no partial photobleaching
    """
    h5_files = getfiles(extension='h5', folder=analysis_folder)
    
    roi_1 = roi.ROI([0, 0], size=dimensions)
    if split_axis == 1:
        roi_2 = roi.ROI([dimensions[0], 0], size=dimensions)
    if split_axis == 2:
        roi_2 = roi.ROI([0, dimensions[1]], size=dimensions)

    loc = [io.load(h5) for h5 in tqdm(h5_files, desc='files', leave=False)]
    
    loc_r1 = [roi_1(loc_temp[loc_temp['frame']==analysis_frame]) for loc_temp in loc]
    if subsequent_recording:
        for loc_temp in loc:
            loc_temp['frame'] -= 1
    loc_r2 = [roi_2(loc_temp[loc_temp['frame']==analysis_frame]) for loc_temp in loc]
    loc_r2 = [correction_registrator(loc_r2_temp, channel=2) for loc_r2_temp in loc_r2]
    
    if set_roi != None:
        loc_r1 = [set_roi(loc_temp) for loc_temp in loc_r1]
        loc_r2 = [set_roi(loc_temp) for loc_temp in loc_r2]
    
    n_1 = 0
    n_2 = 0
    n_coloc = 0
    
    for loc_r1_temp, loc_r2_temp in zip(loc_r1, loc_r2):
        colocs = multicolor.find_colocalizations(loc_r1_temp, loc_r2_temp, max_dist=d_coloc)
        
        n_1 += len(loc_r1_temp)
        n_2 += len(loc_r2_temp)
        n_coloc += len(colocs)
    
    dimer_fraction = 2.0*n_coloc*(n_1+n_2-n_coloc) / ((2.0*n_1-n_coloc)*(2.0*n_2-n_coloc))
    fraction_labeling_1 = (2.0*n_1-n_coloc) / (2.0*(n_1+n_2-n_coloc))
    
    return dimer_fraction, fraction_labeling_1

