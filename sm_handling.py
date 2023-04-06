import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims

from sdt import io, roi
from sdt.brightness import from_raw_image

from ipywidgets import interact, interactive, fixed, interact_manual, interactive_output, Layout
import ipywidgets as widgets

from helpers import getfiles


###### Functions for handling single molecule data:
def get_sm_int(sm_folder, stats='median', start_frame=30, int_roi=None):
    '''
    Calculates mean/median single molecule intensity (within ROI) from localised data 
    Parameters:
        sm_folder: string specifying the folder containig single molecule data
        start_frame: first frame to consider for single molecule intensity
        int_roi: sdt.roi ROI for single molecule intensity determination
    Returns:
        sm_int: mean/median single molecule intesity
    '''
    sm_int_data = []

    sm_files = getfiles(extension='h5', folder=sm_folder)

    for f in sm_files:
        sm_int_data.append(io.load(f))
    sm_int_data = pd.concat(sm_int_data)

    sm_int_data = sm_int_data[sm_int_data['frame'] > start_frame] #remove first frame since it's pretty dim
    if int_roi is not None:
        sm_int_data = int_roi(sm_int_data) #only use central ROI (laser profile!)
    
    if stats == 'median':
        sm_int = sm_int_data["mass"].median()
    elif stats == 'mean':
        sm_int = sm_int_data["mass"].mean()
    
    return sm_int


###### Functions for handling reference data:
def get_int_ratio(data_folder, ref_files, ref_num, settings, sm_roi):
    '''
    Calculates intensity ratio from reference images recorded at different laser powers and different illumination times
    Parameters:
        data_folder: string specifying the folder containing all reference files
        ref_files: dictionary with cells 'c1', 'c2', etc. as keys and list of corresponding filenames as values
        ref_num: number of reference samples
        settings: list of the used settings for recording the images in this order:
                    settings = [low_settings=[low_att, low_till],
                                lh_settings=[low_att, high_till],
                                hl_settings=[high_att, low_till],
                                high_settings=[high_att, high_till]]
        sm_roi: sdt.roi object specifying the ROI used for single molecule intensity determination
    Returns:
        int_ratio: list of intensity ratios with elements for each recorded reference - is NaN if too little images were recorded
    '''
        
    low_settings = settings[0]
    lh_settings = settings[1]
    hl_settings = settings[2]
    high_settings = settings[3]

    if len(str(low_settings[0])) == 2:
        ref_int_file_low =  data_folder + "/ref" + "/ref{n}_att{a:03}_ill{i:02}.SPE"
    elif len(str(low_settings[0])) == 3:
        ref_int_file_low =  data_folder + "/ref" + "/ref{n}_att{a:04}_ill{i:02}.SPE"
    if len(str(high_settings[0])) == 2:
        ref_int_file_high =  data_folder + "/ref" + "/ref{n}_att{a:03}_ill{i:02}.SPE"
    elif len(str(high_settings[0])) == 3:
        ref_int_file_high =  data_folder + "/ref" + "/ref{n}_att{a:04}_ill{i:02}.SPE"

    int_ratio = []

    for r in range(1,ref_num+1):
        low_file = ref_int_file_low.format(n=r, a=low_settings[0], i=low_settings[1])
        lh_file = ref_int_file_low.format(n=r, a=lh_settings[0], i=lh_settings[1])
        hl_file = ref_int_file_high.format(n=r, a=hl_settings[0], i=hl_settings[1])
        high_file = ref_int_file_high.format(n=r, a=high_settings[0], i=high_settings[1])

        cur_files = ref_files['ref{}'.format(r)]

        if low_file in cur_files and high_file in cur_files:
            with pims.open(low_file) as img:
                low_img = img[0]
            with pims.open(high_file) as img:
                high_img = img[0]

            low_img = sm_roi(low_img)#int_roi(low_img).astype(float) - cam_bg
            high_img = sm_roi(high_img)#int_roi(high_img).astype(float) - cam_bg

            int_ratio.append(low_img.mean() / high_img.mean())
        
        elif len(cur_files)<3:
            int_ratio.append(np.nan)
        
        elif high_file not in cur_files:
            with pims.open(low_file) as img:
                low_img = img[0]
            with pims.open(lh_file) as img:
                lh_img = img[0]
            with pims.open(hl_file) as img:
                hl_img = img[0]

            low_img = sm_roi(low_img)
            lh_img = sm_roi(lh_img)
            hl_img = sm_roi(hl_img)

            att_ratio = low_img.mean() / hl_img.mean()
            ill_ratio = low_img.mean() / lh_img.mean()

            int_ratio.append(att_ratio*ill_ratio)

        elif low_file not in cur_files:
            with pims.open(high_file) as img:
                high_img = img[0]
            with pims.open(lh_file) as img:
                lh_img = img[0]
            with pims.open(hl_file) as img:
                hl_img = img[0]

            high_img = sm_roi(high_img)
            lh_img = sm_roi(lh_img)
            hl_img = sm_roi(hl_img)

            att_ratio = lh_img.mean() / high_img.mean()
            ill_ratio = hl_img.mean() / high_img.mean()

            int_ratio.append(att_ratio*ill_ratio)
                       
    #int_ratio = np.mean(int_ratio)    
    return int_ratio


############################# FUNCTIONS FOR SM DISTRIBUTION FILTERING


def get_toccsl_dist(toccsl_folder, start_frame=30, set_roi=None, rel_mask=None, data_sort=None):
    '''
    Gets toccsl intensity data (within ROI) from localised data 
    Parameters:
        toccsl_folder: string specifying the folder containig single molecule data
        start_frame: first frame to consider for single molecule intensity
        set_roi: sdt.roi ROI for single molecule intensity determination
        rel_mask: np.array that contains correction factors corresponding to laser profile
        data_sort: list that contains the sequence of the sorted data
    Returns:
        toccsl_int_data: toccsl data
    '''
    toccsl_int_data = []

    toccsl_files = getfiles(extension='h5', folder=toccsl_folder)

    for i in range(len(set_roi)):                            #there is a different roi for each group
        for f in data_sort[i]:                               #croping toccsl files with the right roi, before they get added to toccsl_int_data
            r_data = set_roi[i](io.load(toccsl_files[f]))
            toccsl_int_data.append(r_data)
        
    toccsl_int_data = pd.concat(toccsl_int_data)

    toccsl_int_data = toccsl_int_data[toccsl_int_data['frame'] >= start_frame] #remove first frame since it's pretty dim #Max: Changed > to >=
    
    if rel_mask is not None:
        toccsl_int_data['mass'] = [t/rel_mask[int(y),int(x)] for t,x,y in zip(toccsl_int_data['mass'], toccsl_int_data['x'], toccsl_int_data['y'])]
        toccsl_int_data['signal'] = [t/rel_mask[int(y),int(x)] for t,x,y in zip(toccsl_int_data['signal'], toccsl_int_data['x'], toccsl_int_data['y'])]

    
    toccsl_int_data = toccsl_int_data[toccsl_int_data['bg']>0]
    
    return toccsl_int_data

def get_sm_dist(sm_folder, start_frame=30, int_roi=None, rel_mask=None, radius=None):
    '''
    Gets single molecule intensity data (within ROI) from localised data 
    Parameters:
        sm_folder: string specifying the folder containig single molecule data
        start_frame: first frame to consider for single molecule intensity
        int_roi: sdt.roi ROI for single molecule intensity determination
        rel_mask: np.array that contains correction factors corresponding to laser profile
    Returns:
        sm_int_data: single molecule data
    '''
    sm_int_data = []

    sm_files = getfiles(extension='h5', folder=sm_folder)
    sm_files_raw = getfiles(extension='tif', folder=sm_folder) # Max
    
    ### Max
    if radius == None:
        for f in sm_files:
            sm_int_data.append(io.load(f))
    else:
        for f1, f2 in zip(sm_files, sm_files_raw):
            temp_loc = io.load(f1)
            with io.ImageSequence(f2) as img_seq:
                from_raw_image(temp_loc, img_seq, radius=radius)
            sm_int_data.append(temp_loc)
    ### Max
    sm_int_data = pd.concat(sm_int_data)

    sm_int_data = sm_int_data[sm_int_data['frame'] >= start_frame] #remove first frame since it's pretty dim #Max: Changed > to >=
    
    if rel_mask is not None:
        sm_int_data['mass'] = [t/rel_mask[int(y),int(x)] for t,x,y in zip(sm_int_data['mass'], sm_int_data['x'], sm_int_data['y'])]
        sm_int_data['signal'] = [t/rel_mask[int(y),int(x)] for t,x,y in zip(sm_int_data['signal'], sm_int_data['x'], sm_int_data['y'])]
    
    
    if int_roi is not None:
        sm_int_data = int_roi(sm_int_data) #only use central ROI (laser profile!)
    
    sm_int_data = sm_int_data[sm_int_data['bg']>0]
    
    return sm_int_data


def get_toccsl_dist(toccsl_folder, start_frame=30, set_roi=None, rel_mask=None, data_sort=None):
    '''
    Gets toccsl intensity data (within ROI) from localised data 
    Parameters:
        toccsl_folder: string specifying the folder containig single molecule data
        start_frame: first frame to consider for single molecule intensity
        set_roi: list of sdt.roi objects; ROI for single molecule intensity determination
        rel_mask: np.array that contains correction factors corresponding to laser profile
        data_sort: dictionary with keys g1, g2, ... for every group and a list of the corresponding filepaths as values
    Returns:
        toccsl_int_data: toccsl data
    '''
    if len(data_sort) > 1:
        toccsl_int_data = {}

        toccsl_files = getfiles(extension='h5', folder=toccsl_folder)

        for i in range(len(set_roi)):                            #there is a different roi for each group
            for f in data_sort['g{}'.format(i+1)].keys():                               #croping toccsl files with the right roi, before they get added to toccsl_int_data
                r_data = set_roi[i](io.load(toccsl_folder + '\\' + f[:-3] + 'h5'))
                toccsl_int_data['g{}'.format(i+1)] = r_data[r_data['frame'] >= start_frame]


            if rel_mask is not None:
                toccsl_int_data['g{}'.format(i+1)]['mass'] = [t/rel_mask[int(y),int(x)] for t,x,y in zip(toccsl_int_data['g{}'.format(i+1)]['mass'], toccsl_int_data['g{}'.format(i+1)]['x'], toccsl_int_data['g{}'.format(i+1)]['y'])]
                toccsl_int_data['g{}'.format(i+1)]['signal'] = [t/rel_mask[int(y),int(x)] for t,x,y in zip(toccsl_int_data['g{}'.format(i+1)]['signal'], toccsl_int_data['g{}'.format(i+1)]['x'], toccsl_int_data['g{}'.format(i+1)]['y'])]

    else:
        
        toccsl_int_data = get_sm_dist(toccsl_folder, start_frame=start_frame, int_roi=set_roi[0])
    
    return toccsl_int_data



def filter_range(data, r, attribute = 'size'):
    '''
    Filters data according to the range r (list or tuple with min and max value) of the selected attribute
    '''
    filtered = data
    filtered = filtered[filtered[attribute]<=r[1]]
    filtered = filtered[filtered[attribute]>=r[0]]
    
    return filtered

def filter_max(data, r, attribute = 'mass'):
    '''
    Filters data according to the maximum value r of the selected attribute
    '''
    filtered = data
    filtered = filtered[filtered[attribute]<=r]
    
    return filtered

def filter_min(data, r, attribute = 'mass'):
    '''
    Filters data according to the minimum value r of the selected attribute
    '''
    filtered = data
    filtered = filtered[filtered[attribute]>=r]
    
    return filtered


def filtering(data, size_slider, mass_slider, bg_slider):
    '''
    Filters data according to current slider positions
    '''
    filtered = filter_range(data, r=size_slider.value)
    filtered = filter_range(filtered, r=mass_slider.value, attribute='mass')
    filtered = filter_range(filtered, r=bg_slider.value, attribute='bg')
    
    return filtered


def filtering_sm(sm_int_data, filter_parameters):
    '''
    Filters single molcule data according to set slider positions
    '''
    filtered = filter_range(sm_int_data, r=filter_parameters[0])
    filtered = filter_range(filtered, r=filter_parameters[1], attribute='mass')
    filtered = filter_range(filtered, r=filter_parameters[2], attribute='bg')
    
    return filtered

    
def create_sliders(sm_int_data):#=sm_int_data):
    '''
    Creates sliders for size, mass and background that correspond to sm_int_data
    Returns:
        size_slider
        mass_slider
        bg_slider
    '''
    
    if type(sm_int_data) == dict:
        sm_int_data = pd.concat(list(sm_int_data.values()))
    
    ####################### SIZE
    step = 0.01
    data = sm_int_data['size']
    size_slider = widgets.FloatRangeSlider(
        value=[data.min(), data.max()],
        min=data.min(),
        max=data.max()+step,
        step=step,
        description='Size:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.2f'
    )

    ####################### BRIGHTNESS
    step=0.01
    data = sm_int_data['mass']
    mass_slider = widgets.FloatRangeSlider(
        value=[data.min(), data.max()],
        min=data.min(),
        max=data.max()+step,
        step=step,
        description='Brightness:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        overflow_x='auto'
    )


    ####################### BACKGROUND
    step = 0.01
    data = sm_int_data['bg']
    bg_slider = widgets.FloatRangeSlider(
        value=[data.min(), data.max()],
        min=data.min(),
        max=data.max()+step,
        step=step,
        description='Background:',
        disabled=False,
        continuous_update=True,
        orientation='horizontal',
        readout=True,
        readout_format='.2f',
    )

    #size_slider.observe(update_all, 'value');
    #mass_slider.observe(update_all, 'value');
    #bg_slider.observe(update_all, 'value');
    
    return size_slider, mass_slider, bg_slider



