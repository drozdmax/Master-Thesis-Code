import numpy as np
import math as m
from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output, Layout
import ipywidgets as widgets
from sdt import io, roi
from tqdm.notebook import tnrange, tqdm
from tqdm.contrib import tzip


def sort(pre_density,num_file, group_num):
    '''If data contains more than one data group, they will be sorted and seperated.
    Parameters: 
        pre_density: list of densities in the pre bleach frames for all files
        num_file: number of files
        group_num: number of data groups, intended for sorting
    Return:
        data_sort: data sorted into groups
    '''
    thresh = (max(pre_density)-min(pre_density))/group_num    #variable used to set boundaries of intevall while sorting
    data_sort =[]
    temp_list = []

    for i in range (group_num):
        thresh_l = min(pre_density)+ thresh * i            #boundaries for this group, low and high
        thresh_h = thresh_l + thresh
    
        for j in range(num_file):                          #checks which file should be sorted into this group
            if thresh_l <= pre_density[j] <= thresh_h:     #has to be done this way, because order matters
                temp_list.append(j)
            
        data_sort.append(temp_list)
        temp_list = []
    return data_sort


def evaluate_density(f, rois, areas, rec_frame):
    '''
    Evaluates mean density for all files, at different roi levels
    Parameters:
        f: corrected files
        rois: roi mask for each element of levels
        areas: area of each roi mask
        rec_frame: frame number of first recovery image
    Returns:
        density: mean density per roi level
    '''
    density = np.zeros(len(rois))
    num_file = len(f)
    
    #looping through all files
    for ff,i in zip(tqdm(f, desc='Files'),range(num_file)):
        cur = io.load(ff)
    
        #looping through all levels/ rois
        for rr, area, j in zip(rois, areas, range(len(rois))):
            cur_sel = rr(cur)
            molecule_num = len(cur_sel[cur_sel['frame'] == rec_frame])        
            density[j] += molecule_num/area
        
    density = density/num_file
    return density




def critical_density(FR, R_coloc):
    '''
    Parameter:
        FR: false positive rate
        R_coloc: colocalisation radius in µm
    Return:
        rho: critical density
    '''
    rho = (-1)*(np.log(1-FR))/((R_coloc**2)*m.pi)
    return rho

def False_pos(bound, R_coloc):
    '''
    Converts the boundaries of the density data into boundaries for the False positive rate slider
    Parameter:
        bound: list containing minimum and maximum of the density data
        R_coloc: colocalisation radius in µm
    Return:
        FR_bound: list containg minimum and maximum False positive rate, fitted to the density data
    '''
    FR_bound = []
    for i in bound: FR_bound.append( 1-m.exp((-1)*(R_coloc**2)*m.pi*i))
    return FR_bound


def opt_roi(rho, density, levels):
    '''
    Parameter:
        rho: critical density
        density: list of densities for each ROI
        levels: percentages for which ROI's are being constructed in the laserprofile
    Return:
        levels[position]: optimal percentage of the laserprofile to construct the ROI
    '''
    d = list(density)
    max_den = d.index(max(d))
    for i in range(max_den): d[i] = 10
        
        
    distance = [abs(i - rho) for i in d]
    
    position = distance.index(min(distance)) + 1
    if position==len(levels):
        position = position-1
    return levels[position]   


def get_FR_bounds(gr_recdensities, R_coloc=0.5):
    '''
    returns boundaries of false postive rate
    Parameters:
        gr_recdensities: dict; group identifier as keys, densities in first recovery frame stored in an array as values
        R_coloc: float; assumed colocalisation radius in µm; set to 0.5µm by default 
    Returns:
        FR_bound: list
    '''
    
    temp = np.concatenate(list(gr_recdensities.values()))
    bound = min(temp), max(temp)
    FR_bound = False_pos(bound,R_coloc)

    return FR_bound


def find_bound(density):
    '''
    Finds the global minimun and maximum for all data groups
    Parameters: 
        density: contains the density values sorted in groups
    Returns:
        List of the global minimum and maximum
    '''
    min_ = 10
    max_ = 0
    for i in range(len(density)):
        if density[i].min() < min_: min_ = density[i].min()
        if density[i].max() > max_: max_ = density[i].max()
    return [min_, max_]



def create_slider_FR(step, bound):
    '''
    Creates slider for false positive rate.
    Prarameter:
        step: step size of slider
        bound: data used for boundaries of slider
    Returns slider
    '''
    FR_slider = widgets.FloatSlider(
        value = (sum(bound))/2,
        min=min(bound),
        max=max(bound),
        step=step,
        description='FR:',
        disabled= False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.3f'
        )
    return FR_slider


def creat_group_slider(max_datagroups):
    '''
    Creates slider for # of measurements
    Parameters: 
        max_datagroups: Maximum amount of data groups that can selected
    Returns slider
    '''
    group_slider = widgets.IntSlider(
            min=1,
            max=max_datagroups,
            step=1,
            description='#datagroups:'
            )
    return group_slider