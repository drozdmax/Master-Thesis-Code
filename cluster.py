import sys

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


from sdt import roi, image, nbui
#from micro_helpers import pims
import pims
import math
from helpers import *


def remove_cluster(mask, start):
    '''
    Parameters:
        mask: binary matrix originating from adaptive thresh
        start: coordinates of the starting point
    Returns:
        mask: binary matrix with one cluster removed
        cluster: coordinates of pixels in the removed cluster
        size: size of the removed cluster
    '''
    max_x, max_y = np.shape(mask)
    max_x-=1; max_y-=1 # Max: Added line
    cluster = [[start]]
    step = []
    size = 1
    a = 0 
    con = True
    
    #maps out the cluster
    while con:
        for b in range(len(cluster[a])):
            i,j = cluster[a][b]
            mask[i][j] = False
            
            for n in range((i-1), (i+2)):
                if (n<0) or(n== max_x): break # Max: Changed max_y to max_x
                
                for m in range((j-1), (j+2)):
                    if (m<0) or(m == max_y): break
                    
                    if mask[n][m]:
                        step.append([n,m])
                        mask[n][m] = False
                        
        if len(step) == 0: break
        
        size = size + len(step)
        cluster.append(step)
        step = []
        
        a += 1
        if a > 200:
            con = False

    return mask, cluster, size




def add_cluster(mask, cluster):
    '''
    Parameter:
        mask: Binary matrix
        cluster: Big cluster that should be added
    Returns:
        mask: Binary matrix with added cluster
    '''
    
    for a in range(len(cluster)):
        for b in range(len(cluster[a])):
            i,j = cluster[a][b]
            mask[i][j] = True
            
    return mask



def cluster(img, block=15, c=0, smoothf=8, thresh=200, mask_edges=False):
    '''
    Parameter:
        img: image to be corrected
        block: size of area used for thresholding
        c: constant subtracted from mean
        smoothf: for Gaussian smoothing
        thresh: specifies size of clusters to be removed (number of pixels)
        mask_edges: set True if bright edges should be masked, e.g. for TOCCSL
    Returns:
        mask: Binary matrix with only big clusters left
    '''
    #applying adaptive thresholding to find high density areas
    mask = image.adaptive_thresh(img, block, c, smooth=smoothf)
    mask = ndimage.binary_dilation(mask, iterations=2) # Max: Changed binary_closing to binary_dilation
    max_x, max_y = np.shape(mask)
    max_x-=1; max_y-=1 # Max added line

    
    liste = []
    gesamtliste = []
    size = []
    
    #finds starting point in high density area and uses remove_cluster to map out cluster
    for i in range(max_x):
        for j in range(max_y):
            if mask[i][j]:
                start = [i,j]
                mask, liste, s = remove_cluster(mask, start)
                gesamtliste.append(liste)
                size.append(s)
    
    #big clusters are re-added to mask
    temp_marker_bool = False # Max
    for i in range(len(size)): 
        if size[i] > thresh:
            mask = add_cluster(mask, gesamtliste[i])
            continue
        ### Max: Added this part to mask the bright edges without masking bright interior signals. Though masking with previous settings nicely removed the bright edges, often bright molecules I would have liked to keep acted as the seed for a removed cluster.
        if mask_edges:
            for a in range(len(gesamtliste[i])):
                for b in range(len(gesamtliste[i][a])):
                    if (gesamtliste[i][a][b][0]<=1) or (gesamtliste[i][a][b][0]>=max_x-1) or (gesamtliste[i][a][b][1]<=1) or (gesamtliste[i][a][b][1]>=max_y-1):
                        temp_marker_bool = True
            if temp_marker_bool:
                mask = add_cluster(mask, gesamtliste[i])
                temp_marker_bool = False
        ### Max
    
    mask[:1,:] = np.ones_like(mask[:1,:]); mask[-1:,:] = np.ones_like(mask[-1:,:]); mask[:,:1] = np.ones_like(mask[:,:1]); mask[:,-1:] = np.ones_like(mask[:,-1:]) # Max: Added line
    
    return mask






