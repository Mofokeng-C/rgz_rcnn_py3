#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 12 July 2018 by chen.wu@icrar.org
import os
import os.path as osp
import math
import warnings
import csv
from collections import defaultdict

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#import ephem

#getfits_exec = '/Users/Chen/Downloads/wcstools-3.9.5/bin/getfits'
#cutout_cmd = '{0} -sv -o %s -d %s %s %s %s J2000 %d %d'.format(getfits_exec)

#subimg_exec = '/Users/Chen/proj/Montage_v3.3/Montage/mSubimage -d' #degree
montage_path = '/Users/chen/Downloads/Montage/bin'
subimg_exec = '%s/mSubimage' % montage_path
regrid_exec = '%s/mProject' % montage_path
imgtbl_exec = '%s/mImgtbl' % montage_path
coadd_exec = '%s/mAdd' % montage_path
subimg_cmd = '{0} %s %s %.4f %.4f %.4f %.4f'.format(subimg_exec)
splitimg_cmd = '{0} -p %s %s %d %d %d %d'.format(subimg_exec)
"""
e.g.
/Users/Chen/proj/Montage_v3.3/Montage/mSubimage -d
/Users/Chen/proj/rgz-ml/data/EMU_GAMA23/gama_linmos_corrected.fits
/tmp/gama_linmos_corrected_clipped.fits 345.3774 -32.499 6.7488 6.1177
"""

def clip_nan(d, file, fname, work_dir='/tmp'):
    h = d.shape[0]
    w = d.shape[1]
    #print(w, h)
    # up and down
    x1, x2, y1, y2 = None, None, None, None
    for i in range(h):
        if (y1 is None and np.sum(np.isnan(d[i, :])) < w):
            y1 = i
        if (y2 is None and np.sum(np.isnan(d[h - i - 1, :])) < w):
            y2 = h - i - 1
        if (y1 is not None and y2 is not None):
            break

    # left and right
    for j in range(w):
        if (x1 is None and np.sum(np.isnan(d[:, j])) < h):
            x1 = j
        if (x2 is None and np.sum(np.isnan(d[:, w - j - 1])) < h):
            x2 = w - j - 1
        if (x1 is not None and x2 is not None):
            break
    #print(x1, x2, y1, y2)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    #print(cx, cy)
    fhead = file[0].header
    warnings.simplefilter("ignore")
    w = pywcs.WCS(fhead, naxis=2)
    warnings.simplefilter("default")
    ra, dec = w.wcs_pix2world([[cx, cy]], 0)[0]
    #print(ra, dec)
    #ra0 = str(ephem.hours(ra * math.pi / 180)).split('.')[0]
    #dec0 = str(ephem.degrees(dec * math.pi / 180)).split('.')[0]
    width = abs((x2 - x1) * fhead['CDELT1'])
    height = abs((y2 - y1) * fhead['CDELT2'])
    fid = osp.basename(fname).replace('.fits', '_clipped.fits')
    #cmd = cutout_cmd % (fid, work_dir, fname, ra0, dec0, width, height)
    clipped_fname = osp.join(work_dir, fid)
    cmd = subimg_cmd % (fname, clipped_fname, ra, dec, width, height)
    print(cmd)
    return clipped_fname

def clip_file(fname):
    """
    remove all the NaN cells surrounding the image
    """
    file = pyfits.open(fname)
    d = file[0].data
    dim = len(d.shape)
    if (dim > 2):
        if (d.shape[-3] > 1):
            raise Exception("cannot deal with cubes yet")
        else:
            d = np.reshape(d, [d.shape[-2], d.shape[-1]])
    clip_nan(d, file, fname)
    file.close()

def split_file(fname, width_ratio, height_ratio, halo_ratio=50,
               show_split_scheme=False, work_dir='/tmp', 
               equal_aspect=False):
    """
    width_ratio = current_width / new_width, integer
    height_ratio = current_height / new_height, integer
    halo in pixel
    """
    file = pyfits.open(fname)
    d = file[0].data
    #fhead = file[0].header
    h = d.shape[-2] #y
    w = d.shape[-1] #x
    print((h, w))
    if (equal_aspect):
        size = int(min(h / height_ratio, w / width_ratio))
        halo_h, halo_w = (size / halo_ratio,) * 2
        height_ratio, width_ratio = h / size, w / size # update the ratio just in case
        ny = np.arange(height_ratio) * size
        nx = np.arange(width_ratio) * size
        if (ny[-1] + size + halo_h < h):
            ny = np.hstack([ny, h - size])
        if (nx[-1] + size + halo_w < w):
            nx = np.hstack([nx, w - size])
        new_w, new_h = (size,) * 2
    else:
        new_h = int(h / height_ratio)
        halo_h = new_h / halo_ratio
        ny = np.arange(height_ratio) * new_h
        new_w = int(w / width_ratio)
        halo_w = new_w / halo_ratio
        nx = np.arange(width_ratio) * new_w
    print((new_h, new_w))
    print(ny)
    print(nx)

    if (show_split_scheme):
        _, ax = plt.subplots(1)
        ax.imshow(np.reshape(d, [d.shape[-2], d.shape[-1]]))

    for i, x in enumerate(nx):
        for j, y in enumerate(ny):
            x1 = max(x - halo_w, 0)
            y1 = max(y - halo_w, 0)
            wd = new_w
            hd = new_h
            
            x2_c = x1 + wd + halo_w
            x2 = min(x2_c, w - 1)
            x1 -= max(0, x2_c - (w - 1))
                
            y2_c = y1 + hd + halo_h
            y2 = min(y2_c, h - 1)
            y1 -= max(0, y2_c - (h - 1))
            fid = osp.basename(fname).replace('.fits', '%d-%d.fits' % (i, j))
            out_fname = osp.join(work_dir, fid)
            print((splitimg_cmd % (fname, out_fname, x1, y1, (x2 - x1), (y2 - y1))))
            if (show_split_scheme):
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                        linewidth=0.5, edgecolor='r',
                                        facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
    if (show_split_scheme):
        plt.tight_layout()
        plt.show()
        plt.savefig('test.pdf')

def vo_get(split_fits_dir):
    """
    1. use VO query to get the image list close to the centre of a given EMU fits image
    wget -O ir_vo_list.csv 
    "https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+348.425+31.151111+0.0027777
    &RESPONSEFORMAT=CSV"
    2. go through the ir_vo_list.csv, and download the W1 band image (should be just one)
    """
    
    cmd = 'wget -O %s.csv '\
    '"https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+%.4f+%.4f+%.6f'\
    '&RESPONSEFORMAT=CSV"'
    for fn in os.listdir(split_fits_dir):
        #print(split_fits_dir)
        fname = osp.join(split_fits_dir, fn)
        if (not fname.endswith('.fits') or fname.find('wise') > -1):
            continue
        #print(fname)
        file = pyfits.open(fname)
        d = file[0].data
        h = d.shape[-2] #y
        w = d.shape[-1] #x

        cx = int(w / 2)
        cy = int(h / 2)
        #print(cx, cy)
        fhead = file[0].header
        warnings.simplefilter("ignore")
        w = pywcs.WCS(fhead, naxis=2)
        warnings.simplefilter("default")
        ra, dec = w.wcs_pix2world([[cx, cy]], 0)[0]
        radius = max(fhead['CDELT1'] * cx, fhead['CDELT2'] * cy)
        print((cmd % (osp.splitext(fname)[0], ra, dec, radius)))

def download_wise(download_dir):
    """
    """
    mapping = defaultdict(list)
    for fn in os.listdir(download_dir):
        fname = osp.join(download_dir, fn)
        if (not fname.endswith('.csv')):
            continue
        with open(fname, 'r') as votable:
            reader = csv.DictReader(votable)
            for row in reader:
                if ('W1' == row['energy_bandpassname'] and 'image/fits' == row['access_format']):
                    url = row['access_url']
                    mapping[osp.basename(fname).replace('.csv', '.fits')].append(url.split('/')[-1])
                    print(('wget %s' % (url)))
    
    with open(osp.join(download_dir, 'mapping_neighbour.txt'), 'w') as fout:
        for k, v in list(mapping.items()):
            fout.write('%s,%s' % (k, ','.join(v)))
            fout.write(os.linesep)
                
def prepare_coadd(split_fits_dir):
    """
    load the mapping_neighbour into a dict
    create a folder f for each EMU fits split
    create symbolic links inside f pointing back to all related wise IR fits files
    create the imgages.tbl for each EMU fits split
    do the co-adding
    """
    wise_dict = dict()
    with open(osp.join(split_fits_dir, 'mapping_neighbour.txt'), 'r') as fin:
        mylist = fin.read().splitlines()
        for line in mylist:
            ll = line.split(',')
            wise_dict[ll[0]] = ll[1:]

    for fn in os.listdir(split_fits_dir):
        fname = osp.join(split_fits_dir, fn)
        if (fname.endswith('.fits') and fname.find('wise') == -1):
            dirn = fname.replace('.fits', '_dir')
            if (not osp.exists(dirn)):
                os.mkdir(dirn)
            ir_list = wise_dict[fn]
            for ir_fn in ir_list:
                dst = osp.join(split_fits_dir, dirn, ir_fn)
                if (osp.exists(dst)):
                    #print("skip creating %s" % dst)
                    continue
                src = osp.join(split_fits_dir, 'wise_ir', ir_fn)
                os.symlink(src, dst)
            tbl_fn = fname.replace('.fits', '.tbl')
            if (not osp.exists(tbl_fn)):
                cmd = '%s %s %s' % (imgtbl_exec, dirn, tbl_fn)
                print(cmd)
            outfile = fname.replace('.fits', '_wise_coadd.fits')
            if (not osp.exists(outfile)):
                hdr_tpl = fname.replace('.fits', '_tmp.hdr')
                cmd = '%s %s %s %s' % (coadd_exec, tbl_fn, hdr_tpl, outfile)
                print(cmd)

def regrid(split_fits_dir):
    """ 
    3. cutout the image to have a slightly larger angular size than the EMU image
    4. reproject the cutout IR image onto the same grid as the EMU radio image
    """
    for fn in os.listdir(split_fits_dir):
        fname = osp.join(split_fits_dir, fn)
        if (fname.endswith('.fits') and fname.find('wise') == -1):
            file = pyfits.open(fname)
            head = file[0].header.copy()
            hdr_tpl = fname.replace('.fits', '_tmp.hdr')
            #print(dir(head))
            head.totextfile(hdr_tpl, clobber=True)
            infile = fname.replace('.fits', '_wise_whole.fits') # wise_whole
            outfile = fname.replace('.fits', '_wise_regrid.fits') # wise_regrid
            cmd = '%s %s %s %s' % (regrid_exec, infile, outfile, hdr_tpl)
            print(cmd)

if __name__ == '__main__':
    #root_dir = '/Users/Chen/proj/rgz-ml/data/EMU_GAMA23'
    root_dir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/EMU_GAMA23'
    # fname = osp.join(root_dir, 'gama_linmos_corrected.fits')
    # clip_file(fname)

    #fname = osp.join(root_dir, 'gama_linmos_corrected_clipped.fits')
    #split_file(fname, 6, 6, show_split_scheme=False, equal_aspect=True)
    #vo_get(osp.join(root_dir, 'split_fits/1deg'))
    #download_wise(osp.join(root_dir, 'split_fits/1deg'))
    prepare_coadd(osp.join(root_dir, 'split_fits/1deg'))
    #regrid(osp.join(root_dir, 'split_fits/1deg'))
