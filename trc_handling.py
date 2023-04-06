import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#import pims
#from scipy import ndimage
import trackpy
trackpy.quiet()

from sdt import io, roi #, motion, image, nbui

import scipy.optimize
from sdt.motion import AnomalousDiffusion, BrownianMotion

import random
import seaborn as sns


###### Functions for trajectory filtering:
def filter_tracks(subtable, min_track_length = 10):
    return len(subtable) > min_track_length

def min_track_filter(tracked, min_track_length=10, nonempty=False):
    '''
    Filters particles with tracking lenghts below min_track_length out of tracked data
    Parameters:
        tracked: list of panda Dataframes containing tracked data
        min_track_length: minimal tracking length in steps 
    '''

    trc_filtered = []

    for t in tracked:
        tf = t.groupby("particle").filter(filter_tracks, min_track_length)
        if nonempty == True:
            if len(tf)>0:
                trc_filtered.append(tf)
        else:
            trc_filtered.append(tf)
   
    return trc_filtered

def filter_all_data(data, min_track_length=10, nonempty=False):
    '''
    Filters particles with tracking lenghts below min_track_length out of tracked data
    Parameters:
        tracked: dictionary with lists of panda Dataframes containing tracked data as values
        min_track_length: minimal tracking length in steps 
    '''
    filtered_trc = {}
    for k in data.keys():
        trc_data = data[k]

        filtered_trc[k] = min_track_filter(trc_data, min_track_length=min_track_length, nonempty=nonempty)
        
    return filtered_trc



###### Functions to get (mean) trajectory lengths and number of trajectories for tracked data:
def get_traj_number(cur):
    '''
    Returns number of trajectories for one movie
    Parameters:
        cur: dataframe containing tracked data of movie of interest
    Returns:
        num: number of trajectories
    '''
    
    t = cur['particle']
    particles = np.unique(t)

    num = len(particles)
    
    return num

def get_traj_length(cur, px_size=0.16):
    '''
    returns length of all trajectories in one movie from tracking data
    Parameters:
        cur: dataframe containing tracked data of movie of interest
        px_size: pixel size, set to 0.16 µm by default
    Returns:
        r: list of trajectory lengths
    '''
    t = cur['particle']
    particles = np.unique(t)

    num = len(particles)
    
    r = []

    for p in particles:    
        traj = cur[t==p].sort_values('frame')

        frames = len(traj)

        temp_r = 0

        x = traj['x']
        y = traj['y']

        dx = np.diff(x)
        dy = np.diff(y)

        for diff_x, diff_y in zip(dx, dy):
            temp_r += np.sqrt(diff_x**2+diff_y**2)

        r.append(temp_r*px_size)
    
    return r


def get_mean_traj_length(cur, px_size=0.16):
    '''
    Calculates mean trajectory length for tracked data from one movie
    Parameters:
        cur: dataframe containing tracked data of move of interest
        px_size: pixel size, set to 0.16 µm by default
    Returns:
        R: mean tracking length
    '''
    
    r = get_traj_length(cur, px_size)
    
    if len(r) > 0:
        R = np.nanmean(r)
    else:
        R = 0
    
    return R

def get_traj_steps(cur):
    '''
    returns number of steps of all trajectories in one movie from tracking data
    Parameters:
        cur: dataframe containing tracked data of movie of interest
    Returns:
        steps: list of trajectory lengths [number of steps]
    '''
    t = cur['particle']
    particles = np.unique(t)

    steps = []

    for p in particles:
        steps.append(len(cur[cur['particle']==p]))

    return steps

def get_mean_traj_steps(cur):
    '''
    returns mean number of steps of all trajectories in one movie from tracking data
    Parameters:
        cur: dataframe containing tracked data of move of interest
    Returns:
        steps_mean: mean trajectory length in number of steps
    '''
    steps = get_traj_steps(cur)
    
    if len(steps) > 0:
        steps_mean = np.nanmean(steps)
    else:
        steps_mean = 0
    
    return steps_mean


def get_traj_parameters(data, cell=None, h5_files=None, mask=None):
    '''
    Parameters:
        data: if cell is given: dictionary with cell specification ('c1', 'c2', etc.) as keys and list of tracked data as values
              if cell is not given: list of tracked data
        cell: choose which cell should be used ('c1', 'c2', etc.), has to be given if data is a dictionary
        h5_files: dictionary with cell specification ('c1', 'c2', etc.) as keys and list of h5 filenames as values (can only be given if data is a dictionary)
        mask: optional parameter for determination of number of localisations
    Return:
        num: number or trajectories in each movie
        len: mean trajectory length [µm] for each movie 
        steps: mean trajectory length [steps] for each movie
        loc: number of localisations for each movie
    '''
    if cell != None:
        data = data[cell]
    mov = len(data)

    tlen = []
    tnum = []
    tsteps = []
    tloc = []

    for m in range(mov):
        if h5_files != None:
            loc_data = io.load(h5_files[cell][m])
            if mask == None:
                tloc.append(len(loc_data))
            else:
                tloc.append(len(mask(loc_data)))
        
        cur = data[m]
        tlen.append(get_mean_traj_length(cur))
        tnum.append(get_traj_number(cur))
        tsteps.append(get_mean_traj_steps(cur))
        
    return (tnum, tlen, tsteps, tloc) if h5_files != None else (tnum, tlen, tsteps)


def merge_trajectories(trajs):
    '''
    Parameters:
        trajs: list of panda dataframes containing tracked data
    Returns:
        traj_merged: merged trajectory data
    '''
    
    prev_part = 0
    new_trajs = []

    for t in trajs:
        
        cur = t.sort_values('particle')

        g = np.diff(cur.sort_values('particle').particle) != 0


        part_num = np.count_nonzero(g) + prev_part

        p_lens = []

        c = 1
        for i in g:
            if i == False:
                c += 1
            else:
                p_lens.append(c)
                c = 1
        p_lens.append(c)

        new_p = sum([[i]*p for i,p in zip(range(prev_part, part_num+1), p_lens)], [])

        cur['particle'] = new_p

        new_trajs.append(cur)

        prev_part += part_num+1

    traj_merged = pd.concat(new_trajs)
    
    return traj_merged


###### Functions for splitting trajectories:
def make_new_p(t_sub, start_p):
    '''
    Searches for gaps in indices of t_sub to determine when particle moved away from OFF or ON area.
    In that case a new particle number is assigned so that trajectories don't have any gaps.
    Parameters:
        t_sub: panda dataframe containig the tracked signals
        start_p: starting particle number, which will be raised for every new particle
    Returns:
        new_p: list of new paticle numbers
        start_p: next particle number
    '''
    
    delta = np.diff(t_sub.index) > 1 #np.diff() returns the differences of subsequent values in a list
    delta = np.nonzero(delta)[0]  #np.nozero returns a tuple of n arrays where n is the dimension number of delta (here n=1)
    delta = (delta + 1).tolist()
    
    new_p = np.zeros(len(t_sub))

    start = [0] + delta
    end = delta +[len(t_sub)]
    
    for s, e in zip(start, end):
        new_p[s:e] = start_p
        start_p += 1

    return new_p, start_p


def split_traj(cur, mask_on, mask_off):
    '''
    Splits trajectories stored in cur into ON and OFF areas.
    Parameters:
        cur: panda dataframe containing tracked signals
        mask_on: sdt.roi object definig ON area
        mask_off: sdt.roi object defining OFF area
    Returns:
        t_ON: trajectories on ON area
        t_OFF: trajectories on OFF area
    '''
    
    cur_p = 0
    t_OFF = []
    t_ON = []

    for p, t in cur.groupby('particle'):
        ################ get signals from off and on the pattern
        temp_on = mask_on(t)
        temp_off = mask_off(t, invert=True)

        ################ concatenate on and off signals into one dataframe (doesn't contain in between signals), sort and reset index
        t_sorted = pd.concat([temp_on, temp_off]).sort_values('frame').reset_index(drop=True)

        ################ again assign signals to ON and OFF areas, but now with continously ascending index (change > 1 means diffusing away from OFF or ON area)
        t_on = mask_on(t_sorted)
        t_off = mask_off(t_sorted, invert=True)

        new_p, cur_p = make_new_p(t_off, cur_p)
        t_off['particle'] = new_p
        t_OFF.append(t_off)

        new_p, cur_p = make_new_p(t_on, cur_p)
        t_on['particle'] = new_p
        t_ON.append(t_on)

    t_OFF = pd.concat(t_OFF).reset_index(drop=True)
    t_ON = pd.concat(t_ON).reset_index(drop=True)
    
    return t_ON, t_OFF






############ Functions for plotting
def make_trajectory_plots(data, cell, h5_files, im_size=128):
    '''
    Generates figure containing trajectories for each movie of a specific cell
    Parameters:
        data: dictionary with cell specification ('c1', 'c2', etc.) as keys and list of tracked data as values
        cell: string specifying the cell of interest (e.g. 'c1')
        h5_files: dictionary with cell specification ('c1', 'c2', etc.) as keys and list of h5 filenames as values
        im_size: size of the images in pixel (images have to be sqares!)
    returns:
        fig: generated figure
    '''
    mov = len(data[cell])
    col = 3
    rows = int(np.ceil(mov/col))

    fig, ax = plt.subplots(rows, col, figsize=(25,8*rows), sharex=True, sharey=True)

    fig.suptitle('Cell: ' + cell, weight='bold', fontsize=18, y=0.93)

    for i, ax in enumerate(fig.axes):
        if i < mov:
            file = h5_files[cell][i].split('/')[-1].split('.')[0].replace('_', ' ')[:-1]
            if data[cell][i].empty == False:
                trackpy.plot_traj(data[cell][i], ax=ax)
            ax.set_xlim([0,im_size])
            ax.set_ylim([im_size,0])
            ax.set_title('movie {}'.format(i+1), fontsize=16)
            ax.set_box_aspect(1)
            ax.text(im_size*0.98,im_size*1.1,file,color = 'gray',fontsize =8,alpha = 0.5, ha='right',va='bottom')
        else:
            ax.remove()
            
    return fig
    

def make_summary_plot(data, cell, im_size=128):
    '''
    Plots trajectories from all movies for one cell into one figure
    Parameters:
        data: dictionary with cell specification ('c1', 'c2', etc.) as keys and list of tracked data as values
        cell: choose which cell should be used ('c1', 'c2', etc.)
        im_size: size of initial image in pixel
    Returns:
        fig: figure
    '''
    fig, ax = plt.subplots(figsize=(10,10))
    mov = len(data[cell])
    for m in range(mov):
        if data[cell][m].empty == False:
            trackpy.plot_traj(data[cell][m], ax=ax)
    ax.set_xlim([0,im_size])
    ax.set_ylim([im_size,0])
        
    ax.set_title('Cell: ' + cell, fontsize=20, weight='bold')
    
    return fig

def make_sep_trajectories_plot(trc_all_data, t_ON, t_OFF, mask_ON, mask_OFF, p_img, cell):
    '''
    Plots trajectories for whole image, on and off area in three columns for each movie of a chosen cell
    Parameters:
        t_ON and t_OFF: dictionary with cell specification as values ('c1', 'c2', etc) and list of dataframes with tracked data for on/off data
        mask_ON and mask_OFF: dictionary with cell specfication as values ('c1', 'c2', etc) and mask roi objects for on/off data
        p_img: dictionary with cell specfication as values ('c1', 'c2', etc) and pattern image as value
        cell: chosen cell
    Returns:
        fig
    '''
    mov = max([len(t_ON[cell]), len(t_OFF[cell])])

    fig, ax = plt.subplots(mov, 3, figsize=(20,7*mov), sharex=True, sharey=True)

    img = p_img[cell]
    on = mask_ON[cell](p_img[cell])
    off = mask_OFF[cell](p_img[cell], invert = True)

    for l in range(mov):
        trackpy.plot_traj(trc_all_data[cell][l], superimpose=img, ax=ax[l,0])
        if t_ON[cell][l].empty == False:
            trackpy.plot_traj(t_ON[cell][l], superimpose=on, ax=ax[l,1])
        else:
            ax[l,1].imshow(on, cmap='gray')
        if t_OFF[cell][l].empty == False:
            trackpy.plot_traj(t_OFF[cell][l], superimpose=off, ax=ax[l,2])
        else:
            ax[l,2].imshow(off, cmap='gray')

    ax[0,0].set_title('all trajectories', weight='bold', fontsize=18)
    ax[0,1].set_title('ON pattern', weight='bold', fontsize=18)
    ax[0,2].set_title('OFF pattern', weight='bold', fontsize=18)

    fig.suptitle('Cell: ' + cell, weight='bold', fontsize=24, y=0.9)
    
    return fig

def make_sep_trajectories_sum_plot(trc_all_data, t_ON, t_OFF, p_img, mask_ON, mask_OFF, cells):
    '''
    Plots summary trajectory plots for each cell for all tracked data as well as ON and OFF pattern.
    Parameters:
        trc_all_data, t_ON and t_OFF: dictionary with cell specfication as values ('c1', 'c2', etc) and list of dataframes with tracked data for whole image/on/off data
        p_img: dictionary with cell specfication as values ('c1', 'c2', etc) and pattern image as value
        mask_ON and mask_OFF: dictionary with cell specfication as values ('c1', 'c2', etc) and mask roi objects for on/off data
        cells: list containing cell numbers
    Returns:
        fig
    '''
    cell_num = len(cells)
    fig, ax = plt.subplots(cell_num, 3, figsize=(20,7*cell_num), sharex=True, sharey=True)
    if cell_num > 1:
        ax[0,0].set_title('all trajectories', weight='bold', fontsize=18, y=1.1)
        ax[0,1].set_title('ON pattern', weight='bold', fontsize=18, y=1.1)
        ax[0,2].set_title('OFF pattern', weight='bold', fontsize=18, y=1.1)

        c=0
        for i in cells:
            cell = 'c{}'.format(i)

            mov_on = len(t_ON[cell])
            mov_off = len(t_OFF[cell])

            img = p_img[cell]
            on = mask_ON[cell](p_img[cell])
            off = mask_OFF[cell](p_img[cell], invert = True)

            ax[c,0].imshow(img, cmap='gray')
            ax[c,1].imshow(on, cmap='gray')
            ax[c,2].imshow(on, cmap='gray')

            for m in range(len(trc_all_data[cell])):
                if trc_all_data[cell][m].empty == False:
                    trackpy.plot_traj(trc_all_data[cell][m], ax=ax[c,0])

            for m in range(mov_on):
                if t_ON[cell][m].empty == False:
                    trackpy.plot_traj(t_ON[cell][m], superimpose=on, ax=ax[c,1])

            for m in range(mov_off):
                if t_OFF[cell][m].empty == False:
                    trackpy.plot_traj(t_OFF[cell][m], superimpose=on, ax=ax[c,2])           

            ax[c,0].set_title(cell, loc='left', weight='bold', fontsize=18)
            c += 1
    else:
        ax[0].set_title('all trajectories', weight='bold', fontsize=18, y=1.1)
        ax[1].set_title('ON pattern', weight='bold', fontsize=18, y=1.1)
        ax[2].set_title('OFF pattern', weight='bold', fontsize=18, y=1.1)

        c=0
        for i in cells:
            cell = 'c{}'.format(i)

            mov_on = len(t_ON[cell])
            mov_off = len(t_OFF[cell])

            img = p_img[cell]
            on = mask_ON[cell](p_img[cell])
            off = mask_OFF[cell](p_img[cell], invert = True)

            ax[0].imshow(img, cmap='gray')
            ax[1].imshow(on, cmap='gray')
            ax[2].imshow(on, cmap='gray')

            for m in range(len(trc_all_data[cell])):
                if trc_all_data[cell][m].empty == False:
                    trackpy.plot_traj(trc_all_data[cell][m], ax=ax[0])

            for m in range(mov_on):
                if t_ON[cell][m].empty == False:
                    trackpy.plot_traj(t_ON[cell][m], superimpose=on, ax=ax[1])

            for m in range(mov_off):
                if t_OFF[cell][m].empty == False:
                    trackpy.plot_traj(t_OFF[cell][m], superimpose=on, ax=ax[2])           

            ax[0].set_title(cell, loc='left', weight='bold', fontsize=18)
            c += 1
        
    return fig


from scipy import stats

def get_pdf(data):
    x = np.linspace(min(data), 1, int(abs(min(data)))*100)
    y = stats.gaussian_kde(data).pdf(x)
    return x,y

def bootstrap_pdf_peak(dataset, perc=2/3, immob_thresh_ln=np.log(0.08), iterations=1000):
    data = dataset['D']
    data_len = len(np.log(data))

    dataset['runnumber'] = np.linspace(1, data_len, data_len)

    num = int(np.ceil(perc * len(np.log(data))))

    
    max_vals = []
    immob_perc = []
    for i in range(iterations):

        #### randomly choose indices
        idx = []

        for l in range(num):
            n = np.floor(random.random()*data_len)

            while n in idx:
                n = np.floor(random.random()*data_len)

            idx.append(int(n)) 

        #### select corresponding density and diffusion data
        data_sel = dataset[dataset.runnumber.isin(idx)]['D']
        
        #ln_data = [n for n in np.log(data_sel) if np.isnan(n) == False]
        #x, y = get_pdf(ln_data)
        
        fig,ax =plt.subplots()
        my_kde = sns.kdeplot(data=np.log(data_sel), ax=ax)
        line = my_kde.lines[0]
        x, y = line.get_data()        
        
        max_vals.append(np.exp(x[np.where(y == y.max())][0]))
        
        plt.close(fig)
        
        
        immob_perc.append(len(np.log(data_sel)[np.log(data_sel) < immob_thresh_ln])/len(np.log(data_sel)))
        
    return np.mean(max_vals), np.std(max_vals), np.std(immob_perc)




class BrownianMotionFixedEps(AnomalousDiffusion):
    r"""Fit Brownian motion parameters to MSD values
    Fit a function :math:`\mathit{msd}(t_\text{lag}) = 4 D t_\text{lag} +
    4 \epsilon^2` to
    the tlag-vs.-MSD graph, where :math:`D` is the diffusion coefficient and
    :math:`\epsilon` is the fixed positional accuracy (uncertainty).
    """
    _fit_parameters = ["D"]

    def __init__(self, msd_data, eps, n_lag=2, initial=0.5):
        """Parameters
        ----------
        msd_data : msd_base.MsdData
            MSD data
        n_lag : int or inf, optional
            Maximum number of lag times to use for fitting. Defaults to 2.
        exposure_time : float, optional
            Exposure time. Defaults to 0, i.e. no exposure time correction
        """
        self.eps = eps
        
        def residual(x, lagt, target):
            d, = x
            r = self.theoretical(lagt, d)
            return r - target

        initial = np.atleast_1d(initial)
        self._results = {}
        self._err = {}
        for particle, all_m in msd_data.data.items():
            #print(particle)
            nl = min(n_lag, all_m.shape[0])
            lagt = np.arange(1, nl + 1) / msd_data.frame_rate
            r = []
            for target in all_m[:nl, :].T:
                try:
                    f = scipy.optimize.least_squares(
                        residual, initial,
                        bounds=([0], [np.inf]),
                        kwargs={"lagt": lagt, "target": target})
                    r.append(f.x)
                except ValueError:
                    r.append([np.NaN])
            r = np.array(r)
            self._results[particle] = np.mean(r, axis=0)
            if r.shape[0] > 1:
                # Use corrected sample std as a less biased estimator of the
                # population  std
                self._err[particle] = np.std(r, axis=0, ddof=1)

        self._msd_data = msd_data

    def theoretical(self, t, d):
        r"""Calculate theoretical MSDs for different lag times
        Calculate :math:`msd(t_\text{lag}) = 4 D t_\text{app}^\alpha + 4
        \epsilon^2`, where :math:`t_\text{app}` is the apparent time lag
        which takes into
        account particle motion during exposure; see
        :py:meth:`exposure_time_corr`.
        Parameters
        ----------
        t : array-like or scalar
            Lag times
        d : float
            Diffusion coefficient
        eps : float
            Positional accuracy.
        alpha : float, optional
            Anomalous diffusion exponent. Defaults to 1.
        exposure_time : float, optional
            Exposure time. Defaults to 0.
        squeeze_result : bool, optional
            If `True`, return the result as a scalar type or 1D array if
            possible. Otherwise, always return a 2D array. Defaults to `True`.
        Returns
        -------
        numpy.ndarray or scalar
            Calculated theoretical MSDs
        """
        return AnomalousDiffusion.theoretical(t, d, self.eps, np.ones_like(d), 0)

    def _plot_single(self, data_id, n_lag, name, ax, color):
        d, = self._results[data_id]
        d_err, = self._err.get(data_id, (np.NaN,) * 2)

        x = np.linspace(0, n_lag, 100)
        y = self.theoretical(x, d)

        legend = []
        if name:
            legend.append(name)
        legend.append(self._value_with_error("D", "μm²/s", d, d_err))
        legend = "\n".join(legend)

        ax.plot(x, y, c=color, label=legend)