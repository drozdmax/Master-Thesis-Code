import os
from os import listdir

import pickle

import pims

import numpy as np

from tifffile import imsave, imwrite


def make_mean_img(file_paths, save_path=None, profile_frame=0):
    imgs = ()
    for f in file_paths:
        with pims.open(f) as seq:
            imgs += (seq[profile_frame],)
    img = np.mean(np.dstack(imgs), axis=2)
    if save_path is not None:
        imwrite(save_path, img)
    return img

def make_std_img(file_paths, save_path=None, profile_frame=0):
    imgs = ()
    for f in file_paths:
        with pims.open(f) as seq:
            imgs += (seq[profile_frame],)
    img = np.std(np.dstack(imgs), axis=2)
    if save_path is not None:
        imwrite(save_path, img)
    return img


def getfiles(extension = 'SPE' ,folder = os.getcwd(), search=None):
    """
    Returns a list of filepaths of specified type (SPE by default) within given or current(default) folder.
    """
    path = folder + '/{}'
    
    files = [path.format(f) for f in listdir(folder) if f.endswith('.' + extension)]
    
    if search!=None:
        files = [f for f in files if search in f]
    return files

def getfilenames(extension = 'SPE', folder = os.getcwd(), search=None):
    """ 
    Returns a list of filenames of specified type (SPE by default) within given or current(default) folder.
    """
    #path = folder + '/{}'
    
    files = [f for f in listdir(folder) if f.endswith('.' + extension)]
    
    if search!=None:
        files = [f for f in files if search in f]
    return files

def save_interactive_fig(fig, path):
    '''
    Saves figure as png and pyfig file. The latter can be later opened with open_interactive_fig
    Parameters:
        fig: matplotlib figure
        path: should include filename without extension
    '''
    with open(path + '.pyfig', 'wb') as f: 
        pickle.dump(fig, f)
    fig.savefig(path + '.png', bbox_inches='tight')
    
    
def open_interactive_fig(path):
    '''
    Opens figures saved as pyfig file. The latter can be later opened with pickle.
    Parameters:
        path: should include filename without extension
    Returns:
        fig: matplotlib figure
    '''
    with open(path + '.pyfig', 'rb') as f: 
        fig = pickle.load(f)
    return fig


def print_statusline(msg: str):
    '''
    Prints statusline: line is cleared and updated upon calling the function
    Parameters:
        msg: string containing message to be printed
    '''
    last_msg_length = len(print_statusline.last_msg) if hasattr(print_statusline, 'last_msg') else 0
    print(' ' * last_msg_length, end='\r')
    print(msg, end='\r')
    print_statusline.last_msg = msg

def remove_empty_panda(data):
    '''
    Removes empty panda dataframes from (tracked) data
    Parameters:
        data: list of panda dataframes
              OR
              dict with list of panda dataframes as values
    '''
    if type(data) == dict:
        for k in data.keys():
            mov = len(data[k])

            idx = []

            for l in range(mov):
                if data[k][l].empty == True:
                    idx.append(l)

            for n in reversed(range(len(idx))):
                p = idx[n]
                del data[k][p]
                
    elif type(data) == list:
        mov = len(data)

        idx = []

        for l in range(mov):
            if data[l].empty == True:
                idx.append(l)

        for n in reversed(range(len(idx))):
            p = idx[n]
            del data[p]


            
##### Function to remove data:
def remove_data(data, cell, movies, h5_files, SPE_files, mask_bool=False):
    '''
    Removes data from data, h5_files and SPE_files
    Parameters:
        data: dictionary with cells 'c1', 'c2', etc. as keys and corresponding data as values
        cell: string to determine from which cell data should be removed
        movies: list containing which movies should be removed (starting with zero!)
        h5_files: dictionary containing filenames of corresponding h5 files
        SPE_files: dictionary containing filenames of corresponding SPE files
    Returns:
        excluded_files: list of excluded files
    '''
    max_movies = len(data[cell])-1
    for m in movies:
        if m > max_movies:
            raise Exception('Movie numbers out of range!')
    movies = sorted(movies)
    excluded_files = {}
    excl_list = []
    
    if mask_bool == False:
        temp = [f for f in SPE_files[cell] if 'roi' in f or 'ROI' in f]
        roi_file = temp[0]
        roi_pos = SPE_files[cell].index(roi_file)
        if roi_pos < max_movies:
            SPE_files_temp = SPE_files[cell][:roi_pos] + SPE_files[cell][roi_pos+1:]
        else:# roi_pos == max_movies:
            SPE_files_temp = SPE_files[cell][:roi_pos]

    for m in reversed(movies):
        del data[cell][m]
        excl_list.append(h5_files[cell][m])
        del h5_files[cell][m]
        if mask_bool == False:
            del SPE_files_temp[m]
    
    if mask_bool == False:
        SPE_files_temp.append(roi_file)
        SPE_files[cell] = SPE_files_temp
    
    excluded_files[cell] = excl_list
    
    #return data, excluded_files, h5_files, SPE_files
    return excluded_files




def apply_roi(trc_data, r):
    '''
    Applies ROI object to (tracked) data
    Parameters:
        trc_data: list of panda dataframes
                  OR
                  dict with list of panda dataframes as values
        r: std.ROI object
    Returns:
        trc_roi    
    '''
    if type(trc_data) == dict:
        trc_roi = {}

        for k, trc in trc_data.items():
            temp = []
            for m in range(len(trc)):
                temp.append(r(trc[m]))
            trc_roi[k] = temp
            
    elif type(trc_data) == list:
        temp = []
        for m in range(len(trc_data)):
            temp.append(r(trc_data[m]))
        trc_roi = temp
        
    return trc_roi



def assign_dict(data):
    results = {}
    for k, v in data.items():
        temp = {}
        for p, vp in v.items():
            temp[p] = vp['D']
        results[k] = temp

    return results