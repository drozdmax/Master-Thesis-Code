############ Functions for laser profile correction
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np

from scipy import optimize, ndimage

from tifffile import imsave, imwrite

from sdt import roi

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import pims

import os

from tqdm.notebook import tnrange, tqdm
from tqdm.contrib import tzip

from skimage import draw, measure
from cluster import *
from helpers import make_std_img

import warnings


def get_mean_profile(filename, profile_path, profile_frame=0):
    '''
    Determines mean image from passed filenames.
    Parameters:
        filename: string or list of strings specifying path(s) to images
        profile_path: formatable string specifying path to store created image in
    Returns:
        profile_im: mean image
        profile_path: path to stored mean image
    '''
    if type(filename) == list:
        profile_file = [profile_path.format(t) for t in filename]
    else:
        profile_file = profile_path.format(filename)

    if type(profile_file) == list:
        make_mean_img(profile_file, profile_path.format('profile_mean.tif'), profile_frame=profile_frame)
        profile_file = profile_path.format('profile_mean.tif')
        filename = 'profile_mean.tif'


    with pims.open(profile_path.format('profile_mean.tif')) as seq:
        profile_im = seq[0]


    fig, ax = plt.subplots()
    ax.set_title('averaged profile (normalised)', weight='bold')
    im = ax.imshow(profile_im/np.max(profile_im), vmax=1)
    cbar = plt.colorbar(im, ax = ax, aspect=20, pad=0.05)
    cbar.ax.tick_params(labelsize=10) 
    cbar.set_label('normalised intensity', fontsize=14, loc='center', labelpad=10)

    fig.savefig(profile_path.format('profile_mean.png'))
    
    return profile_im, profile_file

def get_std_profile(filename, profile_path, profile_frame=0):
    '''
    Determines standard deviation from passed filenames.
    Parameters:
        filename: string or list of strings specifying path(s) to images
        profile_path: formatable string specifying path to store created image in
    Returns:
        profile_std: standard deviation image
        profile_std_path: path to stored standard deviation image
    '''
    if type(filename) == list:
        profile_file = [profile_path.format(t) for t in filename]
    else:
        profile_file = profile_path.format(filename)

    if type(profile_file) == list:
        make_mean_img(profile_file, profile_path.format('profile_mean.tif'), profile_frame=profile_frame)
        make_std_img(profile_file, profile_path.format('profile_std.tif'), profile_frame=profile_frame)
        profile_std_file = profile_path.format('profile_std.tif')
        filename = 'profile_std.tif'


    with pims.open(profile_path.format('profile_mean.tif')) as seq:
        profile_im = seq[0]
    with pims.open(profile_path.format('profile_std.tif')) as seq:
        profile_std = seq[0]


    fig, ax = plt.subplots()
    ax.set_title('Relative Profile Standard Deviation', weight='bold')
    im = ax.imshow(profile_std/profile_im)
    cbar = plt.colorbar(im, ax = ax, aspect=20, pad=0.05)
    cbar.ax.tick_params(labelsize=10) 
    cbar.set_label('relative standard deviation', fontsize=14, loc='center', labelpad=10)

    fig.savefig(profile_path.format('profile_std.png'))
    
    return profile_std, profile_std_file


##### all things gaussian fitting
def gaussian(height, center_x, center_y, width_x, width_y, bckg):
    '''
    Returns a gaussian function with the given parameters
    Parameters:
        height: amplitude of the gaussian
        center_x, center_y: x and y coordinates of the center
        width_x, width_y: standard deviations in x and y direction
        bckg: background of the gaussian (constant offset)
    '''
    width_x = float(width_x)
    width_y = float(width_y)
    return lambda x,y: abs(bckg) + height*np.exp(
                -(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(data):
    '''
    Returns (height, x, y, width_x, width_y,background) the gaussian parameters of a 2D distribution by calculating its moments
    Parameters:
        data: 2D numpy.array
    Returns:
        height: amplitude of the gaussian
        center_x, center_y: x and y coordinates of the center
        width_x, width_y: standard deviations in x and y direction
    '''
    total = data.sum()
    X, Y = np.indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-x)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-y)**2*row).sum()/row.sum())
    height = data.max()
    return height, x, y, width_x, width_y

def fitgaussian(data):
    '''Returns the gaussian parameters of a 2D distribution found by fitting
    Parameters:
        data: 2D numpy.array
    Returns:
        p = (height, center_x, center_x, width_x, width_y, background)
    '''
    params = moments(data)
    params += (np.abs(np.min(data)),)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                 data)
    p, success = optimize.leastsq(errorfunction, params)
    return p


def get_fit(data, imsize=None):
    '''
    Fits data to a 2D gaussian.
    Parameters:
        data: 2D numpy.array
    Returns:
        gaussian_fit: numpy.array of fitted data
        R2: coefficient of determination of the fit
    '''
    #data=z
    
    if imsize == None:
        shape = data.shape
    else:
        shape = (imsize, imsize)

    plt.matshow(data, cmap=plt.cm.gist_earth_r)

    params = fitgaussian(data)
    fit = gaussian(*params)

    plt.contour(fit(*np.indices(data.shape)), cmap=plt.cm.copper)
    ax = plt.gca()
    (height, x, y, width_x, width_y, background) = params
    
    gaussian_fit = fit(*np.indices(data.shape))
    
    R2 = r2_score(data, gaussian_fit)
    
    gaussian_fit = fit(*np.indices(shape))
    
    plt.text(1.0, 0.05, """
    $R^2$ = %.3f
    height : %.1f
    x : %.1f
    y : %.1f
    width_x : %.1f
    width_y : %.1f
    background : %.1f""" %(R2, height, x, y, width_x, width_y, background),
            fontsize=16, horizontalalignment='left',
            verticalalignment='bottom', transform=ax.transAxes)
        
    return gaussian_fit, R2


def get_fit_mask(data, profile_path, filename, fig_show=True):
    '''
    Returns normalised fit of laser profile and makes several plots, which are saved automatically to the specified profile_path
    Parameters:
        data: laser profile data
        profile_path: path to folder containing laser profile file
        filename: filename of recorded laser profile file
        fig_show: if True 2 extra plots are generated (fit vs. real image and noramalised fit vs. real image)
    '''
    
    z = data
    
    ############## gaussian fitting
    gaussian_fit, R2 = get_fit(z)
    fit_rel = gaussian_fit / np.max(gaussian_fit)

    plt.savefig(profile_path.format(filename[:-4]+'_fit_params.png'), bbox_inches='tight')
    
    
    ############## plot fit vs. real image
    if fig_show == True:
        fig,ax = plt.subplots(1,2,figsize=(10,5))

        ax[0].imshow(gaussian_fit, cmap='hot', vmin=np.min(z), vmax=np.max(z))
        im = ax[1].imshow(z, cmap='hot', vmin=np.min(z), vmax=np.max(z))

        cbar = plt.colorbar(im, ax = ax, fraction=0.0215, pad=0.04)#aspect=20, pad=0.05)
        cbar.ax.tick_params(labelsize=10) 
        cbar.set_label('Intensity [counts]', fontsize=14, loc='center', labelpad=10)

        fig.savefig(profile_path.format(filename[:-4]+'_fit.png'), bbox_inches='tight')

    
    ############## make mask file
    fit_rel = gaussian_fit / np.max(gaussian_fit)
    
    imsave(profile_path.format(filename[:-3]+'tif'), fit_rel)
    
    ############## plot noramalised fit vs. real image
    if fig_show == True:
        fig,ax = plt.subplots(1,2,figsize=(10,5))

        ax[0].imshow(fit_rel, cmap='hot', vmin=0, vmax=1)
        im = ax[1].imshow(z/np.max(z), cmap='hot', vmin=0, vmax=1)

        cbar = plt.colorbar(im, ax = ax, fraction=0.0215, pad=0.04)#aspect=20, pad=0.05)
        cbar.ax.tick_params(labelsize=10) 
        cbar.set_label('Intensity [counts]', fontsize=14, loc='center', labelpad=10)

        fig.savefig(profile_path.format(filename[:-4]+'_norm_fit.png'), bbox_inches='tight')

    return fit_rel

#########cropping image
def crop(im):
    '''
    Parameter:
        im: image of laserprofile
    Returns:
        r_list: 2x2 matrix containing upper_left and bottom_right coordinates used for ROI
    '''
    im = im/np.max(im)
    X = np.array(im < 0) # Max changed 0.4 to 0
    n,m = np.shape(X)

    c = False
    r_list = []
    
    #finding edges to cut out noise
    temp = n # Max
    for i in range(n):
        if c == X[i].all():
    ### Max
            if c:
                temp = i
                continue
            r_list.append(i)
            c = True
    r_list.append(temp)
    c = False
    ### Max
    
    temp = m # Max
    for j in range(m):
        if c == X[:,j].all():
    ### Max
            if c:
                temp = j
                continue
            r_list.append(j)
            c = True
    r_list.append(temp)
    ### Max
            
    r_list[1] = r_list[1] -1
    r_list[3] = r_list[3] -1
    
    return np.reshape(r_list, (2, 2)).T

##### correcting data
def correct_data(files, profile_file, mask_num=None, mask_bool=False, mask_edges=False, fig_bool=False, smoothing_factor=4, thresh=200):
    '''
    Parameters:
        files: list of paths to files to be corrected
        profile_file: path to dummy file
        mask_num: number of frame, after which mask_function will be applied
        mask_bool: specifies if mask_function should be applied
        mask_edges: if true, then only the edges are masked, nothing interior (defaults to False)
        fig_bool: specifies if figure should be created
        smoothing_factor: for gaussian smoothing
        thresh: specifies size of clusters to be removed
    Returns:
        mask_bool: specifies if correction was done
        profile_mask: array containing smoothed profile
    '''
    ###### checking input
    if (mask_bool == True) and (mask_num == None):
        raise ValueError ("You are missing the parameter mask_num")
    elif (mask_bool == False) and (mask_num is not None):
        warnings.warn('If you want to use masking, mask_bool needs to be set to True')
        #raise ValueError ("If you want to use masking, mask_bool needs to be set to True")
    
    with pims.open(profile_file) as seq:
        profile_im = seq[0]
    roi_coordinates = crop(profile_im) # matrix containing upper_left and bottom_down coordinates for roi
    
    ###### create roi object
    r = roi.ROI(roi_coordinates[0], roi_coordinates[1])
    hight = roi_coordinates[1][0] - roi_coordinates[0][0]
    width = roi_coordinates[1][1] - roi_coordinates[0][1]
    
    
    if '/' in profile_file:
        filename = profile_file.split('/')[-1]
    elif '\\' in profile_file:
        filename = profile_file.split('\\')[-1]
    
    if filename ==  '':
        mask = False
    else:
        mask = True

        if '/' in files[0]:
            folder = '/'.join(files[0].split('/')[:-1])
        elif  '\\' in files[0]:
            folder = '\\'.join(files[0].split('\\')[:-1])

        ################### GET DUMMY IMAGE
        with pims.open(profile_file) as seq:
            ####### crop
            img = r(seq[0])

        #################### GAUSSIAN SMOOTHING
        profile_mask = ndimage.gaussian_filter(img/np.max(img), smoothing_factor)

        if fig_bool == True:
            fig, ax = plt.subplots(1, 2, figsize=(10,5))
            ax[0].imshow(img/np.max(img), vmin=0, vmax=1, cmap='jet')
            im = ax[1].imshow(profile_mask, vmin=0, vmax=1, cmap='jet')

            ax[0].set_title('recorded laser profile', weight='bold')
            ax[1].set_title('gaussian smoothing ($\sigma$=' + str(smoothing_factor) + ')', weight='bold')

            cbar = plt.colorbar(im, ax = ax, fraction=0.021, aspect=20, pad=0.05)
            cbar.ax.tick_params(labelsize=10) 
            cbar.set_label('normalised intensity [counts]', fontsize=14, loc='center', labelpad=10)

            fig.savefig(profile_path.format(filename[:-4] +'_gaussian_smoothing.png'), bbox_inches='tight')


        #################### CORRECT AND SAVE IMAGES
        corr_save_path = folder + '/corrected/{}.tif'

        if not os.path.exists(folder + '/corrected'):
            os.makedirs(folder + '/corrected')

        for cur in tqdm(files, desc='Files'):
            with pims.open(cur) as seq:
                
                seq = np.array(seq)
                length = np.shape(seq)[0]
                sequ = []
                
                if mask_num is not None and mask_bool==True:
                    for s in range(0,mask_num):
                        corrected_image = (r(seq[s]))/profile_mask
                        sequ.append(corrected_image)
                        
                    for s in range(mask_num,length):
                        corrected_image = (r(seq[s]))/profile_mask
                        masking = cluster(corrected_image, thresh=thresh, mask_edges=mask_edges)
                        r_masking = roi.MaskROI(masking)
                        im = r_masking(corrected_image, invert=True,  fill_value=0.00001)
                        sequ.append(im)
                else:
                    for s in range(length):
                        corrected_image = (r(seq[s]))/profile_mask
                        sequ.append(corrected_image)
               
                c = np.reshape(sequ, (length, width, hight))
                

                imsave(corr_save_path.format(cur.split('/')[-1][:-4]), c)
                
    return (mask, profile_mask) if mask == True else (mask, [])


##### choosing and creating ROIs
def choose_ROI(fit_rel, data, levels, smoothed=None, smoothing_factor=4):
    '''
    Creates and returns 
    
    Parameters:
        fit_rel: normalised gaussian fit of laser profile
        data: laser profile data
        levels: list of levels for contour lines
        smoothed (optional): smoothed laser profile
        smoothing_factor (): smoothing factor used for smoothing
    Returns:
        fig: figure showing the recorded image, the smoothed image (if smoothed is passed) and the relative fit with contour lines
    '''
    
    if type(smoothed) == None:
        fig, ax = plt.subplots(1,2,figsize=(10,5))

        im = ax[1].imshow(fit_rel, cmap='jet', vmax=1, vmin=np.min(data/np.max(data)))
        c = ax[1].contour(fit_rel, cmap='brg', levels=levels)
        ax[1].clabel(c)
        #ax[0].contour(fit_rel, cmap='brg', levels=levels)
        ax[0].imshow(data/np.max(data), cmap='jet', vmax=1, vmin=np.min(data/np.max(data)))
        #ax[0].contour(data/np.max(data), cmap='brg', levels=levels)
        #ax[2].imshow(data/np.max(data), cmap='hot', vmax=1, vmin=np.min(data/np.max(data)))
        #ax[3].contour(fit_rel, cmap='brg', levels=levels)
        #ax[3].contour(data/np.max(data), cmap='brg', levels=levels)
        #ax[3].set_aspect('equal')
        #ax[3].set_ylim([np.shape(data)[1],0])
        ax[0].set_title('recorded laser profile', weight='bold')
        ax[1].set_title('gaussian fit', weight='bold')
        
        cbar = plt.colorbar(im, ax = ax, fraction=0.0215, aspect=20, pad=0.05)
        cbar.ax.tick_params(labelsize=10) 
        cbar.set_label('normalised intensity [counts]', fontsize=14, loc='center', labelpad=10)
        
    else:
        fig, ax = plt.subplots(1, 3, figsize=(15,5))
        ax[0].imshow(data/np.max(data), vmin=0, vmax=1, cmap='jet')
        ax[1].imshow(smoothed, vmin=0, vmax=1, cmap='jet')
        im = ax[2].imshow(fit_rel, vmin=0, vmax=1, cmap='jet')

        ax[0].set_title('recorded laser profile', weight='bold')
        ax[1].set_title('gaussian smoothing ($\sigma$=' + str(smoothing_factor) + ')', weight='bold')
        ax[2].set_title('gaussian fit', weight='bold')

        #ax[1].contour(smoothed, cmap='brg', levels=[b_level, sm_level])

        c = ax[2].contour(fit_rel, cmap='brg', levels=levels)#[b_level, sm_level])
        ax[2].clabel(c)

        cbar = plt.colorbar(im, ax = ax, fraction=0.015, aspect=20, pad=0.05)
        cbar.ax.tick_params(labelsize=10) 
        cbar.set_label('normalised intensity [counts]', fontsize=14, loc='center', labelpad=10)
    
    return fig


def get_edges_booleans(p, imsize):
    '''
    Checks if point p lies at edges of the image with size of imsize
    '''
    x = p[0]
    y = p[1]

    #1st edge at x=0
    edge1 = False
    #2nd edge at x=imsize
    edge2 = False
    #3rd edge at y=0
    edge3 = False
    #4th edge at y=imsize
    edge4 = False

    if x==0:
        edge1 = True
    elif x==imsize-1:
        edge2 = True
    elif y==0:
        edge3 = True
    elif y==imsize-1:
        edge4 = True
        
    return (edge1,edge2,edge3,edge4)

def get_edges_add(start, end, edges, imsize):
    '''
    Returns corner points to add to contour lines if the end and start point lie on different edges.
    Parameters:
        start, end: start and end point of contour line
        edges = (edge1, edge2, edge3, egde4): tuple of booleans specifiying at which edge the end point lies
                - edge1: x=0
                - edge2: x=imsize
                - edge3: y=0
                - edge4: y=imsize
        imsize: image size
    '''
    edge1, edge2, edge3, edge4 = edges
    x = end[0]
    y = end[1]
    
    edges_add = []
    ################# Edge 1: x = 0
    if edge1 == True:
        if y < imsize/2:
            edges_add.append([0,imsize])
            if np.any(start == [0,imsize]) == False:
                edges_add.append([imsize,imsize])
                if np.any(start == [imsize,imsize]) == False:
                    edges_add.append([imsize,0])
                    if np.any(start == [imsize,0]) == False:
                        edges_add.append([0,0])
        else:
            edges_add.append([0,0])
            if np.any(start == [0,0]) == False:
                edges_add.append([imsize,0])
                if np.any(start == [imsize,0]) == False:
                    edges_add.append([imsize,imsize])
                    if np.any(start == [imsize,imsize]) == False:
                        edges_add.append([0,imsize])
                        
    
    ################# Edge 2: x = imsize
    if edge2 == True:
        if y < imsize/2:
            edges_add.append([imsize,imsize])
            if np.any(start == [imsize,imsize]) == False:
                edges_add.append([0,imsize])
                if np.any(start == [0,imsize]) == False:
                    edges_add.append([0,0])
                    if np.any(start == [0,0]) == False:
                        edges_add.append([imsize,0])
        else:
            edges_add.append([imsize,0])
            if np.any(start == [imsize,0]) == False:
                edges_add.append([0,0])
                if np.any(start == [0,0]) == False:
                    edges_add.append([0,imsize])
                    if np.any(start == [0,imsize]) == False:
                        edges_add.append([imsize,imsize])
    
    
    ################# Edge 3: y = 0
    if edge3 == True:
        if x < imsize/2:
            edges_add.append([imsize,0])
            if np.any(start == [imsize,0]) == False:
                edges_add.append([imsize,imsize])
                if np.any(start == [imsize,imsize]) == False:
                    edges_add.append([0,imsize])
                    if np.any(start == [0,imsize]) == False:
                        edges_add.append([0,0])
        else:
            edges_add.append([0,0])
            if np.any(start == [0,0]) == False:
                edges_add.append([0,imsize])
                if np.any(start == [0,imsize]) == False:
                    edges_add.append([imsize,imsize])
                    if np.any(start == [imsize,imsize]) == False:
                        edges_add.append([imsize,0])
                        
    ################# Edge 4: y = imsize
    if edge4 == True:
        if x < imsize/2:
            edges_add.append([imsize,imsize])
            if np.any(start == [imsize,imsize]) == False:
                edges_add.append([imsize,0])
                if np.any(start == [imsize,0]) == False:
                    edges_add.append([0,0])
                    if np.any(start == [0,0]) == False:
                        edges_add.append([0,imsize])
        else:
            edges_add.append([0,imsize])
            if np.any(start == [0,imsize]) == False:
                edges_add.append([0,0])
                if np.any(start == [0,0]) == False:
                    edges_add.append([imsize,0])
                    if np.any(start == [imsize,0]) == False:
                        edges_add.append([imsize,imsize])
    return edges_add

def make_ROI(fit_rel, data=None, chosen_value=0.95, fig_show = True):
    '''
    Creates std.ROI objects
    Parameters:
        fit_rel: normalised gaussian fit of laser profile
        data (optional): laser profile data
        chosen_value: contour line of fit_rel chosen as border of ROI
        fig_show (optional): if True and if data is passed some plots will be generated
    Returns:
        r: ROI
        r_inv: inverted ROI
        fig: created figure (only if fig_show is set to True)
    '''
    imsize = len(fit_rel)
    #l = imsize*px_size    
    
    if fig_show == True:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(fit_rel, cmap='hot', vmax=1, vmin=np.min(data/np.max(data)))
        c = ax.contour(fit_rel, levels=[chosen_value])
        #cp = c.allsegs[0][0]

    contours = measure.find_contours(fit_rel, chosen_value)
    if len(contours) > 1:
        tup = ()
        for i in range(len(contours)):
            contours[i][:, [1, 0]] = contours[i][:, [0, 1]]
            tup += (contours[i],)
            cp = np.concatenate(tup, axis=0)
    else:
        contours[0][:, [1, 0]] = contours[0][:, [0, 1]]
        cp = contours[0]
        
        
    ########## check if start and end point are at different edges
    start = cp[0]
    end = cp[-1]
    if np.all(start == end):
        diff_edges = False
    else:
        diff_edges = True

    if diff_edges == True:
    ########### check edge for end point
        edges = get_edges_booleans(end, imsize=np.shape(fit_rel)[0])

    ########## define which points need to be added to the contour line
        edges_add = get_edges_add(start, end, edges, imsize=np.shape(fit_rel)[0])

    ###################### add to contour lines
        cp = np.concatenate((cp,np.array(edges_add)))

        
        
    points_contour = cp
    
    r = roi.PathROI(points_contour)
    p_add = [[0,imsize], [imsize,imsize], [imsize,0], [0,0], [0,imsize]]

    temp = np.append(p_add, points_contour)
    temp = np.append(temp, points_contour[0])
    temp = np.split(temp, len(temp)/2)
    r_inv = roi.PathROI(temp)
    
    if data is not None and fig_show == True:
        z = data
        fig, ax = plt.subplots(2,3, figsize=(20,10), constrained_layout=True)
        ax[0,0].imshow(fit_rel, cmap='hot', vmax=1, vmin=np.min(z/np.max(z)))
        ax[0,0].plot(points_contour[:,0], points_contour[:,1])
        im = ax[0,2].imshow(r(fit_rel), cmap='hot', vmax=1, vmin=np.min(z/np.max(z)))
        ax[0,1].imshow(r_inv(fit_rel), cmap='hot', vmax=1, vmin=np.min(z/np.max(z)))
        
        cbar = fig.colorbar(im, ax = ax, aspect=60, pad=0.05)
        cbar.ax.tick_params(labelsize=20) 
        cbar.set_label('normalised intensity', fontsize=25, loc='center', labelpad=10)
        
        ax[1,0].imshow(z/np.max(z), cmap='hot', vmax=1, vmin=np.min(z/np.max(z)))
        ax[1,0].plot(points_contour[:,0], points_contour[:,1])
        im = ax[1,2].imshow(r(z/np.max(z)), cmap='hot', vmax=1, vmin=np.min(z/np.max(z)))
        ax[1,1].imshow(r_inv(z/np.max(z)), cmap='hot', vmax=1, vmin=np.min(z/np.max(z)))
        
        #cbar = fig.colorbar(im, ax = ax[1,2], fraction=0.05, pad=0.1, label='normalised intensity')
        
    return (r, r_inv, fig) if fig_show else (r, r_inv)