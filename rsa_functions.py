# -*- coding: utf-8 -*-
"""
RSA_functions.py
"""

BIDS_DIR = ''
TASKS = ['friend', 'number']
CONTRAST_NAMES = range(10)
N_CONTRASTS = len(CONTRAST_NAMES)
TR = 0.75
N_TRS = 514
PARCELLATION_FNAME = ''

from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr, norm, zscore
import os
import shutil
import sys
from datetime import datetime
import copy
import glob
import pandas as pd
import nibabel as nib
import numpy as np
#from funcs import *

def load_nii( imgpath ):
    """
    :param imgpath: (str)
                    Nifti image to read in.
                    Absolute path or relative path from current directory.
    :return: (numpy 3D or 4D array)
                    The values of each voxel in a 3D or 4D array based on the
                    nifti's dimensions.
    """
    img4D = nib.load(imgpath, mmap=False).get_data()
    return img4D

def save_nii( data, refnii, filename ):
    """
    :param data: (numpy 3D or 4D array)
                    The data from a nifti image.
    :param refnii: (nibabel Nifti1Image object)
                    A nifti image to use as reference for header and affine settings
    :param filename: (str)
                    The full path to save the nifti object to.
    """
    out_img = nib.Nifti1Image(data, refnii.affine, refnii.header)
    out_img.to_filename(filename)
    print(str(datetime.now()) + ": File %s saved." % filename)

def make_RDM( data_array ):
    """
    Finds the absolute value between every pair of values in data_array and
    organizes these differences into a representational dissimilarity matrix (RDM).

    :param data_array: (numpy 1D array)
                    A vector of values: 1 element per row/column
    :return: (numpy array)
                    The lower triangle of the symmetric RDM
    """
    # find number of nodes
    n = len(data_array)
    if n != N_NODES:
        print("WARNING: number nodes entered ("+ str(n) +") is different than N_NODES ("+str(N_NODES)+"):")
        print(data_array)
    # create empty matrix
    mat = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            mat[i,j] = abs(data_array[i] - data_array[j])
    # make squareform (upper diagonal elements)
    tri = squareform(mat)
    return tri

def get_node_mapping( sub ):
    """
    :param sub: (str)
    :return: (dictionary)
    """
    events_file = glob.glob(BIDS_DIR+sub+"/func/"+sub+"*events.tsv")[0]
    events_df = pd.read_csv(events_file, delimiter='\t')
    node_mapping = {}
    for n in range(N_NODES):
        # find first row that matches this node
        r = events_df.loc[events_df['node']==n].index[0]
        # find corresponding stimulus file name
        stim = events_df.loc[r,'stim_file']
        # remove '.png'
        stim = stim.split('.')[0]
        # save to mapping
        node_mapping[stim] = n
    # order node numbers to match CFD measures
    node_order = []
    cfd_targets = pd.read_csv(CFD_FNAME)['Target']
    for t in cfd_targets:
        node_order.append(node_mapping[t])
    return node_order

def get_model_RDM_dict( node_mapping, meas_name_array,
                        df = None, fname=CFD_FNAME,
                        compress=False, out_key='sum' ):
    """
    :param node_mapping: (dictionary, keys=img (str) and values=int)
    :param meas_name_array: (array)
    :param df: (pandas dataframe)
    :param fname: (str)
    :param compress: (boolean)
    :param out_key: (str)
    :return: (dictionary)
    """
    if df is None:
        df = pd.read_csv(fname)
    # copy chicago face database data frame
    df_sub = copy.deepcopy(df)
    # add to CFD measures data frame and sort
    df_sub['Node'] = node_order
    df_sub.sort_values(by='Node', inplace=True)
    # social features
    rdm_dict = {}
    for i in meas_name_array:
        # extract column
        v = df_sub[i]
        if compress:
            df_sub['z'+i] = zscore(v)
        else:
            # make RDMs
            rdm_dict[i] = make_RDM(v)
    if compress:
        v = df_sub[meas_name_array].sum(axis=1)
        rdm_dict[out_key] = make_RDM(v)
    return rdm_dict

def run_rsa_reg(neural_v, model_mat):
    """
    :param neural_v: (numpy array)
    :param model_mat: (numpy matrix)
    :return: (numpy array)
    """
    # orthogonalize
    model_mat = zscore(model_mat)
    model_mat,R = np.linalg.qr(model_mat)
    # column of ones (constant)
    X = np.hstack((model_mat,
                 np.ones((model_mat.shape[0],1))))
    # Convert neural DSM to column vector
    neural_v = neural_v.reshape(-1,1)
    # Compute betas and constant
    betas = np.linalg.lstsq(X, neural_v)[0]
    # Determine model (if interested in R)
    for k in range(len(betas)-1):
        if k==0:
            model = X[:,k].reshape(-1,1)*betas[k]
        else:
            model = model + X[:,k].reshape(-1,1)*betas[k]
    # Get multiple correlation coefficient (R)
    R = pearsonr(neural_v.flatten(),model.flatten())[0]
    out = betas[:-1] + [R]
    return np.array(out)
