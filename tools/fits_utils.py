#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 12 July 2018 by chen.wu@icrar.org
import os, errno
import os.path as osp
import math
import warnings
import csv
import subprocess
import re
import glob
import math
#import pyvo
from collections import defaultdict
import montage_wrapper as montage
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.io.votable import parse
from astropy.stats import sigma_clipped_stats

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fuse_radio_ir import fuse
from functions import vo_get_host_pos
#import ephem

#getfits_exec = '/Users/Chen/Downloads/wcstools-3.9.5/bin/getfits'
#cutout_cmd = '{0} -sv -o %s -d %s %s %s %s J2000 %d %d'.format(getfits_exec)

#= '/Users/Chen/proj/Montage_v3.3/Montage/mSubimage -d' #degree
#montage_path = '/Users/chen/Downloads/Montage/bin'
#subimg_exec = '%s/mSubimage' % montage_path
#regrid_exec = '%s/mProject' % montage_path
#imgtbl_exec = '%s/mImgtbl' % montage_path
#coadd_exec = '%s/mAdd' % montage_path
#subimg_cmd = '{0} %s %s %.4f %.4f %.4f %.4f'.format(subimg_exec)
#splitimg_cmd = '{0} -p %s %s %d %d %d %d'.format(subimg_exec)
"""
e.g.
/Users/Chen/proj/Montage_v3.3/Montage/mSubimage -d
/Users/Chen/proj/rgz-ml/data/gmrt_GAMA23/gama_linmos_corrected.fits
/tmp/gama_linmos_corrected_clipped.fits 345.3774 -32.499 6.7488 6.1177
"""
def run_command(cmd):
    """given shell command, returns outcome of call function used to run a command"""
    return subprocess.call(cmd)

def force_symlink(file1, file2):
    try:
        os.symlink(file1, file2)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(file2)
            os.symlink(file1, file2)


def generate_cutouts(fits_file,cat_file,work_dir):
    """
    This function takes fits file, its corresponding catalog and the working directory.
    
    returns cutouts of the respective position fro, the catalog
    """
    
    f = pyfits.open(fits_file)
    fhead = f[0].header
    f.close()
    warnings.simplefilter("ignore")
    w = pywcs.WCS(fhead, naxis=2)
    warnings.simplefilter("default")
    
    cat = parse(cat_file)
    table = cat.get_first_table()
    name = table.array['Source_id']
    r = table.array['RA_deg']
    d = table.array['Dec_deg']
    flux = table.array['Total_flux']
    
    #iterate given N positions
    N_total = len(name) #for all the datasets contained in the catalog
    print("Total number of sources in the catalog %f"%N_total)
    for i in range(N_total):
        #print('--', name[i],r[i],d[i])
        
        ra = r[i]
        dec = d[i]
        
        npix = 3.0/abs(fhead['CDELT1']*60) # each cutout must be of size 3 arcmin.
                                           # so diving that by pixel scale [degrees per pixel] of the fits file to get number of pixels.
        
        #width = abs(180 * ) #width of the output square cutout
        width = abs(int(npix)*fhead['CDELT1'])
        #print(width)
        
        #print(width)
        #fid = osp.basename(file).replace('.fits', '%.2f-%.2f.fits'%(ra,dec))
        fid = osp.basename(fits_file).replace('.fits', '_%s.fits'%(name[i]))
        cutout_fname = osp.join(work_dir, fid) #file name for the output file
        
        #This bit makes the cutouts given the position and width of the image
        if (osp.exists(cutout_fname)):
            #skip a cutout that is aleady available
            continue
            
        try:
            montage.mSubimage(fits_file,cutout_fname,ra,dec,width)
            
                
        except montage.status.MontageError:
            continue    
    
    
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
    fid = osp.basename(fname).replace('.FITS', '_clipped.fits')
    #cmd = cutout_cmd % (fid, work_dir, fname, ra0, dec0, width, height)
    clipped_fname = osp.join(work_dir, fid)
    #cmd = subimg_cmd % (fname, clipped_fname, ra, dec, width, height)
    montage.mSubimage(fname,clipped_fname,ra,dec,width)
    #print(clipped_fname)
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
    file.close()
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
            #print((splitimg_cmd % (fname, out_fname, x1, y1, (x2 - x1), (y2 - y1))))
            montage.mSubimage_pix(fname,out_fname,x1,y1,(x2 - x1))
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

def vo_get(split_fits_dir, download_dir):
    """
    1. use VO query to get the image list close to the centre of a given Radio position
    wget -O ir_vo_list.csv 
    "https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+348.425+31.151111+0.0027777
    &RESPONSEFORMAT=CSV"
    or
    wget -O out.tbl "https://irsa.ipac.caltech.edu/SCS?table=allwise_p3as_psd&RA=180.0&DEC=0.0&SR=0.05&format=csv"
    
    wget -O 1.tgz “http://unwise.me/cutout_fits?version=neo3&ra=41&dec=10&size=100&bands=1”
    
    2. go through the ir_vo_list.csv, and download the W1 band image (should be just one)
    """
    
    #cmd = 'wget -O %s.csv "https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+%.4f+%.4f+%.6f&RESPONSEFORMAT=CSV"'
    #cmd = ['wget', '-O', 'wise%.2f_%.2f.tgz','http://unwise.me/cutout_fits?version=neo3&ra=%.2f&dec=%.2f&size=%s&bands=1']
    for fn in os.listdir(split_fits_dir):
        #print(split_fits_dir)
        fname = osp.join(split_fits_dir, fn)
        if (not fname.endswith('.fits') or fname.find('wise') > -1):
            continue
        #print(fname)
        file = pyfits.open(fname)
        d = file[0].data
        #file.close()
        h = d.shape[-2] #y
        w = d.shape[-1] #x
        

        cx = int(w / 2)
        cy = int(h / 2)
        #print(cx, cy)
        fhead = file[0].header
        file.close()
        warnings.simplefilter("ignore")
        w = pywcs.WCS(fhead, naxis=2)
        warnings.simplefilter("default")
        #ra, dec = w.wcs_pix2world([[cx, cy]], 0)[0]
        ra, dec = w.wcs_pix2world([[cx, cy]], 0)[0]
        #radius = max(fhead['CDELT1'] * cx, fhead['CDELT2'] * cy)
        radius = max(fhead['CDELT1'] * cx, fhead['CDELT2'] * cy)
        qstring = "https://irsa.ipac.caltech.edu/SIA?COLLECTION=wise_allwise&POS=circle+%.4f+%.4f+%.6f&RESPONSEFORMAT=CSV"%(ra, dec, radius)
        #https://irsa.ipac.caltech.edu/TAP/sync?"#COLLECTION=wise_allwise&"
        #arg = #"QUERY=SELECT+ra,dec+FROM+allwise_p3as_psd+WHERE+CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',%.4f,%.4f,%.5f))=1&format=csv"%(x, y,radius)#
        f_out = "%s.csv"%(osp.join(download_dir, osp.splitext(osp.basename(fname))[0]))
        
        #cmd = ["wget","-O",f_out,url+arg]
        cmd = 'wget -O %s %s'%(f_out,qstring)
        run_command(cmd.split())
        
        
        

def download_wise(download_dir):
    """
    """
    mapping = defaultdict(list)
    for fn in os.listdir(download_dir):
        fname = osp.join(download_dir, fn)
        if (not fname.endswith('.csv')):
            continue
        with open(fname, 'rt') as votable:
            reader = csv.DictReader(votable)
            match = 0
            for row in reader:
                csv_base = osp.basename(fname)
                if ('W1' == row['energy_bandpassname'] and 'image/fits' == row['access_format']):
                    url = row['access_url']
                    wise_suffix = '_%d_wise.fits' % match
                    mapping[csv_base.replace('.csv', '.fits')].append(csv_base.replace('.csv', wise_suffix))
                    
                    out_file = osp.join(download_dir,"%s"%fn.replace('.csv', wise_suffix))
                    if osp.exists(out_file):
                        continue
                    #cmd = ["wget","-O",out_file,url]
                    cmd = 'wget -O %s %s'%(out_file,url)
                    run_command(cmd.split())
                    
                    #print('wget -O %s %s' % (fn.replace('.csv', wise_suffix), url))
                    match += 1
                    
    
    with open(osp.join(download_dir, 'mapping_neighbour.txt'), 'wt') as fout:
        for k, v in list(mapping.items()):
            l = "%s,%s" % (k, ','.join(v))
            fout.write(l)
            fout.write(os.linesep)
                
def prepare_coadd(split_fits_dir):
    """
    load the mapping_neighbour into a dict
    create a folder f for each gmrt fits split
    create symbolic links inside f pointing back to all related wise IR fits files
    create the imgages.tbl for each gmrt fits split
    do the co-adding
    
    NOTE - this requires (original and derived) wise fits and original radio fits mingled 
    into the same directory
    """
    
    wise_dict = dict()
    with open(osp.join(split_fits_dir, 'mapping_neighbour.txt'), 'r') as fin:
        mylist = fin.read().splitlines()
        for line in mylist:
            ll = line.split(',')
            wise_dict[ll[0]] = ll[1:]
            
    with open(osp.join(split_fits_dir,"MontageError_coadd.txt"), "wt") as f:
        
        for fn in os.listdir(split_fits_dir):
            if (not fn in wise_dict):
                #print(fn)
                continue
            fname = osp.join(split_fits_dir, fn)
            if (fname.endswith('.fits') and fname.find('wise') == -1):
                dirn = fname.replace('.fits', '_dir')
                if (not osp.exists(dirn)):
                    os.mkdir(dirn)
                    #run_command(["mkdir",dirn])
                ir_list = wise_dict[fn]
                for ir_fn in ir_list:
                    # co-add works on projected images
                    ir_fn = ir_fn.replace('_wise.fits', '_wise_regrid.fits')
                    dst = osp.join(split_fits_dir, dirn, ir_fn)
                    #print(dst)
                    if (osp.exists(dst)):
                        #print("skip creating %s" % dst)
                        continue
                    src = osp.join(split_fits_dir, ir_fn)
                    force_symlink(src,dst)
                    #os.symlink(src, dst)
                tbl_fn = fname.replace('.fits', '.tbl')
                #print(dirn)
                if (not osp.exists(tbl_fn)):
                    #cmd = '%s %s %s' % (imgtbl_exec, dirn, tbl_fn)
                    try:
                        rtn = montage.mImgtbl(dirn, tbl_fn)
                        #print(rtn)
                        
                    except montage.status.MontageError:
                        rtn = montage.mAdd(tbl_fn,hdr_tpl,outfile)
                        f.write("Imgtbl: %s, %s\n"%(str(rtn),tbl_fn))
                        continue
                    #print(cmd)
                # coadd requires the area file co-located in the same directory as input fits
                for ir_fn in ir_list:
                    ir_fn = ir_fn.replace('_wise.fits', '_wise_regrid_area.fits')
                    dst = osp.join(split_fits_dir, dirn, ir_fn)
                    if (osp.exists(dst)):
                        #print("skip creating %s" % dst)
                        continue
                    src = osp.join(split_fits_dir, ir_fn)
                    force_symlink(src,dst)

                outfile = fname.replace('.fits', '_wise_coadd.fits')
                if (not osp.exists(outfile) and len(os.listdir(dirn))>2): 
                    hdr_tpl = fname.replace('.fits', '_gmrt.hdr')   
                    #cmd = '%s %s %s %s' % (coadd_exec, tbl_fn, hdr_tpl, outfile)
                    try:
                        rtn = montage.mAdd(tbl_fn,hdr_tpl,outfile)
                        print("Coadding to ",outfile)
 
                    except montage.status.MontageError:
                        rtn = montage.mAdd(tbl_fn,hdr_tpl,outfile)
                        f.write("mAdd: %s, %s\n"%(str(rtn),hdr_tpl))
                        continue

def regrid(gmrt_fits_dir, ir_fits_dir):
    """ 
    re-project the cutout IR image onto the same grid as the gmrt radio image
    """\
    
    with open(osp.join(ir_fits_dir,"Montage_error.txt"), "wt") as f:
        
        for fn in os.listdir(ir_fits_dir):
            fname = osp.join(ir_fits_dir, fn)
            if (fname.endswith('_wise.fits')):
                gmrt_fits = osp.join(gmrt_fits_dir, re.sub('_[0-9]_wise', '', fn))
                hdr_tpl = osp.join(ir_fits_dir, osp.basename(gmrt_fits).replace('.fits', '_gmrt.hdr'))
                if (not osp.exists(hdr_tpl)):
                    file = pyfits.open(gmrt_fits)
                    head = file[0].header.copy()
                    head.totextfile(hdr_tpl, overwrite=True)
                outfile = fname.replace('_wise', '_wise_regrid') # wise_regrid

                if (not osp.exists(outfile)):
                    rtn = ""
                    try:
                        print("Projecting image to %s"%outfile)
                        rtn = montage.mProject(fname,outfile,hdr_tpl)

                    except montage.status.MontageError:
                        f.write("mProject: %s - %s\n"%(str(rtn),outfile))
                        continue

    # for fn in os.listdir(gmrt_fits_dir):
    #     fname = osp.join(gmrt_fits_dir, fn)
    #     if (fname.endswith('.fits') and fname.find('wise') == -1):
    #         file = pyfits.open(fname)
    #         head = file[0].header.copy()
    #         hdr_tpl = fname.replace('.fits', '_tmp.hdr')
    #         #print(dir(head))
    #         head.totextfile(hdr_tpl, clobber=True)
    #         infile = fname.replace('.fits', '_wise_whole.fits') # wise_whole
    #         outfile = fname.replace('.fits', '_wise_regrid.fits') # wise_regrid
    #         cmd = '%s %s %s %s' % (regrid_exec, infile, outfile, hdr_tpl)
    #         print(cmd)
            
def fits2png_D4(fits_dir, png_dir):
    """
    Convert fits to png files based on the D4 method
    """

    """
    export DISPLAY=:1
    Xvfb :1 -screen 0 1024x768x16 &
    """
    
    
    # Because there is quiet a number of images in oone directory,
    # separating them by using glob function to capture their names was very efficient.
    
    for fits in glob.glob(osp.join(fits_dir,"*[0-9].fits")): # Capture only fits files for D1 maps
        
        mean_filled = fits.replace('.fits','_filled.fits')
        if not osp.exists(mean_filled):
            #if fits.endswith('[0-9].fits'):

        #    png = fits.replace('.fits','_logminmax.png')
            png = fits.replace('.fits','_infrared.png')
            #png = osp.join(png_dir,osp.basename(png))
            if (not osp.exists(png)):
                print(fits)
                fits_0 = re.sub("infrared.png","0_wise_regrid.fits",png)
                fits_1 = re.sub("infrared.png","1_wise_regrid.fits",png)

                if (osp.exists(fits_0) and not osp.exists(fits_0.replace('0_wise_regrid.fits','wise_coadd.fits'))):

                    f = mean_clip(fits_0)
                    if f!="":
                        cmd = 'ds9 %s -cmap gist_heat -cmap value 0.30 0 -scale log -scale mode minmax -export %s -exit'%(f,png)
                        run_command(cmd.split())
                        print(fits_0)
                    else:
                        continue
                    #continue

                elif (osp.exists(fits_1) and not osp.exists(fits_1.replace('1_wise_regrid.fits','wise_coadd.fits'))):

                    f = mean_clip(fits_1)
                    if f!="":
                        cmd = 'ds9 %s -cmap gist_heat -cmap value 0.30 0 -scale log -scale mode minmax -export %s -exit'%(f,png)
                        run_command(cmd.split())
                        print(fits_1)
                    else:
                        continue
                    #continue


                elif (osp.exists(png.replace('infrared.png','wise_coadd.fits'))):

                    f = mean_clip(png.replace('infrared.png','wise_coadd.fits'))
                    if f!="":
                        cmd = 'ds9 %s -cmap gist_heat -cmap value 0.30 0 -scale log -scale mode minmax -export %s -exit'%(f,png)
                        run_command(cmd.split())
                        print(png.replace('infrared.png','wise_coadd.fits'))

                    else:
                        continue
                    #print(png.replace('infrared.png','wise_coadd.fits'))
                    
            else:
                continue

def mean_clip(fits_file):
    
    output_fname = fits_file.replace('.fits','_filled.fits')
    
    try:
        with fits.open(fits_file) as file:
            hduwise = file[0]
            wcswise = WCS(hduwise.header, naxis=2)
            #fits_data = hduradio.data
            mean, _, _ = sigma_clipped_stats(hduwise.data, sigma=3.0, maxiters=5)

            hduwise.data[np.isnan(hduwise.data)] = mean

            hduwise.header.remove('CRPIX3')
            hduwise.header.remove('CRVAL3')
            hduwise.header.remove('CDELT3')
            #hduwise.header.remove('CUNIT3')
            hduwise.header.remove('CTYPE3')
            hduwise.header.remove('NAXIS3')
            hduwise.header.remove('NAXIS4')

            hduwise.writeto(output_fname)

            return output_fname
    except OSError:
        return ""
    
                
                
def fits2png_D1(fits_dir, png_dir):
    """
    Convert fits to png files based on the D1 method
    """
    #cmd_tpl = '%s -cmap Cool'\
    #    ' -zoom to fit -scale log -scale mode minmax -export %s -exit'
    #from sh import Command
    #ds9 = Command(ds9_path)
    #fits = '/Users/chen/gitrepos/ml/rgz_rcnn/data/gmrt_GAMA23/split_fits/30arcmin/gama_linmos_corrected_clipped0-0.fits'
    #png = '/Users/chen/gitrepos/ml/rgz_rcnn/data/gmrt_GAMA23/split_png/30arcmin/gama_linmos_corrected_clipped0-0.png'
    
    # Run this command in the terminal before using the function:
    """
    export DISPLAY=:1
    Xvfb :1 -screen 0 1024x768x16 &
    """
    
    
    # Because there is quiet a number of images in oone directory, i thought separating them by using glob function to capture their names was very efficient.
    
    for fits in glob.glob(osp.join(fits_dir,"*[0-9].fits")): # Capture only fits files for D1 maps
        #if fits.endswith('[0-9].fits'):
        
        png = fits.replace('.fits','_logminmax.png')
        #png = osp.join(png_dir,osp.basename(png))
        #png = osp.join(png_dir,osp.basename(png))
        if (not osp.exists(png)):
            cmd = 'ds9 %s -cmap cool -scale log -scale mode minmax -export %s -exit'%(fits,png)
            run_command(cmd.split())
            print(png)

if __name__ == '__main__':
    #root_dir = '/Users/Chen/proj/rgz-ml/data/gmrt_GAMA23'
    #root_dir = '/Users/chen/gitrepos/ml/rgz_rcnn/data/gmrt_GAMA23'
    root_dir = '/idia/users/cmofokeng/data/gmrt'
    #root_dir = '/users/cmofokeng/rgz_rcnn/'
    #fname = osp.join(root_dir, 'gmrt_en1w610.FITS')
    #clip_file(fname)
    #fname_tmp = '/tmp/gmrt_en1w610_clipped.fits'
    #run_command(["mv",fname_tmp,root_dir])
    
    
    #fname = osp.join(root_dir, 'gmrt_en1w610_clipped.fits')
    
    #generate_cutouts(fname,osp.join(root_dir,'en1w610-5sg9-clean-offset.vot'),osp.join(root_dir, 'split_fits/1deg'))

    
    #split_file(fname, 6, 6, show_split_scheme=True, equal_aspect=True)
    #vo_get(osp.join(root_dir, 'split_fits/1deg'), osp.join(root_dir, 'split_fits/wise_1deg'))
    #vo_get(osp.join(root_dir, 'test'), osp.join(root_dir, 'test'), emu_type='E1')
    #vo_get(osp.join(root_dir, 'split_fits/1deg'), osp.join(root_dir, 'split_fits/wise_1deg'))
    #vo_get_host_pos(osp.join(root_dir, 'split_fits/test_1deg'), osp.join(root_dir, 'split_fits/position'))
    #download_wise(osp.join(root_dir, 'split_fits/wise_1deg'))
    #download_wise(osp.join(root_dir, 'split_fits/test_1deg'))
    #download_wise(osp.join(root_dir, 'test'))
    #regrid(osp.join(root_dir, 'fits'), osp.join(root_dir, 'ir'))
    #regrid(osp.join(root_dir, 'split_fits/1deg'), osp.join(root_dir, 'split_fits/wise_1deg'))
    #regrid(osp.join(root_dir, 'split_fits/test_split'), osp.join(root_dir, 'split_fits/test_split'))
    #regrid(osp.join(root_dir, 'test'), osp.join(root_dir, 'test'))
    #prepare_coadd(osp.join(root_dir, 'split_fits/1deg'))
    #prepare_coadd(osp.join(root_dir, 'split_fits/test_1deg'))
    #prepare_coadd(osp.join(root_dir, 'split_fits/test_split'))
    #prepare_coadd(osp.join(root_dir, 'output_gmrt/15arcmin'))
    #fits2png(osp.join(root_dir, 'split_fits_1deg_960MHz'), osp.join(root_dir, 'split_png_1deg_960MHz'))
    #fits2png(osp.join(root_dir, 'fits'), osp.join(root_dir, 'fits'))
    #fits2png(osp.join(root_dir, 'split_fits/1deg'), osp.join(root_dir, 'split_fits/1deg'))
    #fits2png(osp.join(root_dir, 'split_fits/test_1deg'), osp.join(root_dir, 'split_fits/test_1deg'))
    #fits2png_D4(osp.join(root_dir, 'split_fits/test_1deg'), osp.join(root_dir, 'split_fits/test_1deg'))
    fits2png_D4('~/rgz_rcnn/output_gmrt/15arcmin', '~/rgz_rcnn/output_gmrt/15arcmin')
    #fits2png_D1(osp.join(root_dir, 'split_fits/test_1deg'), osp.join(root_dir, 'split_fits/test_1deg'))
    #fits2png(osp.join(root_dir, 'test'), osp.join(root_dir, 'test'))
    
