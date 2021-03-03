#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 14:03:14 2020

@author: jwiesner
"""

import numpy as np
from nilearn.image import new_img_like, iter_img, math_img
from nilearn.masking import apply_mask,unmask
from nilearn.regions import connected_regions
from nilearn import plotting
from sklearn.utils import check_random_state

def region_standardize(stat_img,mask_img,extract_type='connected_components',min_region_size=1,
                       smoothing_fwhm=None,ignore_zeros=True,verbose=False):
    """Standardizes a statistical image region-wise.
    
    Regions in the statistical image will be extracted using nilearn.regions.connected_regions.
    Voxels are then z-standardized according to the region-wise mean and standard deviation.
    
    By default this function is set to be exhaustive (using extract_type = 'connected_components'
    and min_region_size=1) that means every voxel is assigned to one particular region. 
    
    If a region contains of only one voxel, the voxel is standardized according to the mean and 
    standard deviation of the whole brain.
    
    Voxels that have not been assigned to a region are also standardized according to the mean
    and standard deviation of the whole brain.
    
    Parameters
    ----------
    stat_img: Niimg-like object
        A statistical image.
    
    extract_type: str {‘connected_components’, ‘local_regions’}, optional
        See nilearn.masking.connected_regions
        
        Default: 'connected_components'
    
    min_region_size: int, optional
        See nilearn.masking.connected_regions
        
        Default: 1 mm^3
    
    smoothing_fwhm: int, optional
        See nilearn.masking.connected_regions
        
        Default: None
    
    mask_img: Niimg-like object
        A binary image used to mask the data.
    
    ignore_zeros: boolean
        If True, zero-voxels inside the statistical image will be ignored
        when calculating descriptive statistics. This makes sense if the 
        statistical image contains lots of zero voxels (e.g. because
        it is already thresholded).
        
    verbose: boolean
        Print information region sizes and percentage of overall assigned voxels
        
    Returns
    -------
    region_standardized_img: Niimg-like object
        The region standardized version of the original statistical image.
    
    region_sizes: list
        A list showing how many regions where extracted (length of the list)
        and how many voxels are assigned to each region.
    
    See also
    --------
    nilearn.regions.connected_regions
    
    """
    
    # find regions in statistical image
    region_imgs,indices = connected_regions(stat_img,
                                      extract_type=extract_type,
                                      min_region_size=min_region_size,
                                      smoothing_fwhm=smoothing_fwhm)
    
    region_sizes = []
    n_assigned_voxels = 0
    
    stat_img_data = np.asarray(stat_img.dataobj)
    
    # get mean and standard deviation of whole brain
    stat_img_data_masked = apply_mask(stat_img,mask_img)
    
    if ignore_zeros is True:
        stat_img_data_masked = stat_img_data_masked[np.nonzero(stat_img_data_masked)]
    
    whole_brain_mean = stat_img_data_masked.mean()
    whole_brain_std = stat_img_data_masked.std()
    
    region_standardized_stat_img_data = np.zeros_like(stat_img_data)
    region_standardized_indices = np.zeros_like(stat_img_data)
    
    # z-standardize values region-wise
    for idx,region_img in enumerate(iter_img(region_imgs)):
        
        region_img_data = region_img.get_fdata()
        
        # for user infomration get size of every region and count overall number 
        # of voxels that have been assigned to a region
        region_img_size = np.count_nonzero(region_img_data)
        region_sizes.append(region_img_size)
        n_assigned_voxels += region_img_size
        
        # get region indices and corresponding values
        region_img_indices = np.where(region_img_data != 0 )
        stat_img_data_region_values = stat_img_data[region_img_indices]
        
        # save indices
        region_standardized_indices[region_img_indices] = 1
        
        # get the size of each region 
        region_size = len(region_img_indices[0])
        
        # if region consists of more than one voxel calculate the mean and
        # standard deviation from all voxels within that region
        if region_size > 1:
            stat_img_data_region_mean = stat_img_data_region_values.mean()
            stat_img_data_region_std = stat_img_data_region_values.std()
        
        # if region consists of only one voxel, standardize that voxel
        # according to mean and standard deviation of whole brain
        else:
            stat_img_data_region_mean = whole_brain_mean
            stat_img_data_region_std = whole_brain_std

        # z-standardize region values
        stat_img_data_region_values_std = ((stat_img_data_region_values - stat_img_data_region_mean) / stat_img_data_region_std)
        
        # impute z-standardized values into predefined region standardized data array
        region_standardized_stat_img_data[region_img_indices] = stat_img_data_region_values_std
        
    # unassigned voxels are standardized according to whole brain mean and sd
    # get indices of voxels that were not assigned to a region
    # FIXME: Currently function ignores zero-voxels that means zero-voxels
    # within the brain are also ignored. Allow to decide if zero-voxels should
    # be ignored ore not by using ignore_zeros = True/False. For that you
    # have to work with masked data (zero voxels outside brain should
    # always be ignored). Masked data is a 1D array so you have to work with
    # transformations from 1D to 3D.
    unassigned_voxel_indices = np.where((region_standardized_indices != 1) & (stat_img_data != 0))
    unassigned_voxel_values = stat_img_data[unassigned_voxel_indices]
    unassigned_voxel_values_std = ((unassigned_voxel_values - whole_brain_mean) / whole_brain_std)
    region_standardized_stat_img_data[unassigned_voxel_indices] = unassigned_voxel_values_std
    
    # create image from region standardized data
    region_standardized_img = new_img_like(stat_img,region_standardized_stat_img_data)
        
    if verbose == True:

        for region_idx in range(0,len(region_sizes)):
            print('Region {}: {} voxels'.format(region_idx,region_sizes[region_idx]))
        
        percentage_assigned_voxels = n_assigned_voxels / np.count_nonzero(stat_img_data) * 100
        print('{} % of all voxels have been assigned to a region.\n'.format(round(percentage_assigned_voxels,2)))
        
    return region_standardized_img,region_sizes


def scramble_img(img,mask_img,random_state=None):
    '''Shuffles the voxel values within a Niimg-like object
    
    img: Niimg-like object
        Input image where the voxels should be randomly shuffled.
    
    mask_img: Niimg-like object
        A mask which masks the data to brain space
    
    random_state: int, RandomState instance or None, optional, default=None
        If int, random_state is the seed used by the random number generator; 
        If RandomState instance, random_state is the random number generator; 
        If None, the random number generator is the RandomState instance used by np.random. 
        Used when shuffle == True.

    '''
    
    rng = check_random_state(random_state)
    
    # mask image to MNI152 template
    img_data = apply_mask(img,mask_img).flatten()
    n_voxels = img_data.shape[0]
    
    # shuffle data
    img_data_scrambled = rng.choice(img_data,n_voxels,replace=False)
    
    # transform back to image
    img_scrambled = unmask(img_data_scrambled,mask_img)
    
    return img_scrambled

def _get_voxel_volume(voxel_sizes) :  
    result = 1 
    for dim in voxel_sizes:  
        result *= dim
    return result

def cluster_binary_img(binary_img,mask_img,min_region_size='exhaustive'):

    # get voxel resolution in binary_img
    # NOTE: function currently assumes equal width in x,y,z direction
    voxel_sizes = binary_img.header.get_zooms()
    
    # if not specfied by user, cluster exhaustive, i.e. assign each and every
    # voxel to one and only one cluster
    if min_region_size == 'exhaustive':
        min_region_size_ = _get_voxel_volume(voxel_sizes) - 1
    else:
        min_region_size_ = min_region_size
    
    # count overall number of 1s in the binary image
    total_n_voxels = np.count_nonzero(binary_img.get_fdata())
    
    # extract clusters in binary image
    cluster_imgs,indices = connected_regions(maps_img=binary_img,
                                             min_region_size=min_region_size_,
                                             extract_type='connected_components',
                                             smoothing_fwhm=None,
                                             mask_img=mask_img)
    
    
    # Get sizes of clusters (number of voxels that have been assigned to each region)
    # As a sanity check + for user information get size of every region and 
    # count overall number of voxels that have been assigned to that region
    cluster_sizes = []
    total_n_voxels_assigned = 0
    
    for idx,cluster_img in enumerate(iter_img(cluster_imgs)):
        cluster_img_data = cluster_img.get_fdata()    
        cluster_img_size = np.count_nonzero(cluster_img_data)
        cluster_sizes.append(cluster_img_size)
        total_n_voxels_assigned += cluster_img_size
    
    if total_n_voxels_assigned != total_n_voxels:
        raise ValueError('Number of voxels in output clustered image is different from total number of voxels in input binary image ')
        
    # Collapse the extracted cluster images to one cluster atlas image
    cluster_imgs_labeled = []
    
    for idx,cluster_img in enumerate(iter_img(cluster_imgs),start=1):
        cluster_img_labeled = math_img(f"np.where(cluster_img == 1,{idx},cluster_img)",cluster_img=cluster_img)
        cluster_imgs_labeled.append(cluster_img_labeled)
        
    cluster_img_atlas = math_img("np.sum(imgs,axis=3)",imgs=cluster_imgs_labeled)
    
    # plot the cluster atlas image 
    plotting.plot_roi(cluster_img_atlas,
                      title='Clustered Binary Image',
                      cut_coords=[0,0,0],
                      draw_cross=False)
    
    return cluster_sizes

def swap_img_data(img,mask_img,absolute_values=True):
    '''Swap data in a nifti image. User can choose to do swapping by taking
    the real or absolute values in the image into account'''
    
    img_data = apply_mask(img,mask_img)
    
    if absolute_values == True:
        map_dict = dict(zip(sorted(set(img_data),key=abs),sorted(set(img_data),key=abs,reverse=True)))
    else:
        map_dict = dict(zip(sorted(set(img_data)),sorted(set(img_data),reverse=True)))
    
    img_data_swapped = np.array([map_dict[x] for x in img_data])
    img_swapped = unmask(img_data_swapped,mask_img)
    
    return img_swapped

if __name__ == '__main__':
    pass





