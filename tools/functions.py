import sys
import os
import re
import math
import csv
import os.path as osp
import glob
import warnings
import subprocess
from astroquery.irsa import Irsa
import astropy.units as u

sys.path.append('/users/cmofokeng/rgz_rcnn/lib/nms')
sys.path.append('/users/cmofokeng/rgz_rcnn/tools/')

import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import numpy as np

from astropy.io import ascii
from scipy.ndimage.measurements import center_of_mass
from astropy.stats import sigma_clip
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.table import Table
from photutils import detect_threshold, detect_sources, source_properties
from astropy.stats import sigma_clipped_stats,sigma_clip

from fuse_radio_ir import fuse

import cv2


MAX_REJECT = 0.5
MIN_NPIXELS = 5
GOOD_PIXEL = 0
BAD_PIXEL = 1
KREJ = 2.5
MAX_ITERATIONS = 5

def zscale (image, nsamples=1000, contrast=0.25, bpmask=None, zmask=None):
    """Implement IRAF zscale algorithm
    nsamples=1000 and contrast=0.25 are the IRAF display task defaults
    bpmask and zmask not implemented yet
    image is a 2-d numpy array
    returns (z1, z2)
    """

    # Sample the image
    samples = zsc_sample (image, nsamples, bpmask, zmask)
    npix = len(samples)
    samples.sort()
    zmin = samples[0]
    zmax = samples[-1]
    # For a zero-indexed array
    center_pixel = (npix - 1) // 2
    if npix%2 == 1:
        median = samples[center_pixel]
    else:
        median = 0.5 * (samples[center_pixel] + samples[center_pixel + 1])

    #
    # Fit a line to the sorted array of samples
    minpix = max(MIN_NPIXELS, int(npix * MAX_REJECT))
    ngrow = max (1, int (npix * 0.01))
    ngoodpix, zstart, zslope = zsc_fit_line (samples, npix, KREJ, ngrow,
                                             MAX_ITERATIONS)

    if ngoodpix < minpix:
        z1 = zmin
        z2 = zmax
    else:
        if contrast > 0: zslope = zslope / contrast
        z1 = max (zmin, median - (center_pixel - 1) * zslope)
        z2 = min (zmax, median + (npix - center_pixel) * zslope)
    return z1, z2

def zsc_sample (image, maxpix, bpmask=None, zmask=None):
    
    # Figure out which pixels to use for the zscale algorithm
    # Returns the 1-d array samples
    # Don't worry about the bad pixel mask or zmask for the moment
    # Sample in a square grid, and return the first maxpix in the sample
    nc = image.shape[0]
    nl = image.shape[1]
    stride = max (1.0, math.sqrt((nc - 1) * (nl - 1) / float(maxpix)))
    stride = int (stride)
    samples = image[::stride,::stride].flatten()
    return samples[:maxpix]
    
def zsc_fit_line (samples, npix, krej, ngrow, maxiter):

    #
    # First re-map indices from -1.0 to 1.0
    xscale = 2.0 / (npix - 1)
    xnorm = np.arange(npix)
    xnorm = xnorm * xscale - 1.0

    ngoodpix = npix
    minpix = max (MIN_NPIXELS, int (npix*MAX_REJECT))
    last_ngoodpix = npix + 1

    # This is the mask used in k-sigma clipping.  0 is good, 1 is bad
    badpix = np.zeros(npix, dtype="int32")

    #
    #  Iterate

    for niter in range(maxiter):

        if (ngoodpix >= last_ngoodpix) or (ngoodpix < minpix):
            break
        
        # Accumulate sums to calculate straight line fit
        goodpixels = np.where(badpix == GOOD_PIXEL)
        sumx = xnorm[goodpixels].sum()
        sumxx = (xnorm[goodpixels]*xnorm[goodpixels]).sum()
        sumxy = (xnorm[goodpixels]*samples[goodpixels]).sum()
        sumy = samples[goodpixels].sum()
        sum = len(goodpixels[0])

        delta = sum * sumxx - sumx * sumx
        # Slope and intercept
        intercept = (sumxx * sumy - sumx * sumxy) / delta
        slope = (sum * sumxy - sumx * sumy) / delta
        
        # Subtract fitted line from the data array
        fitted = xnorm*slope + intercept
        flat = samples - fitted

        # Compute the k-sigma rejection threshold
        ngoodpix, mean, sigma = zsc_compute_sigma (flat, badpix, npix)

        threshold = sigma * krej

        # Detect and reject pixels further than k*sigma from the fitted line
        lcut = -threshold
        hcut = threshold
        below = np.where(flat < lcut)
        above = np.where(flat > hcut)

        badpix[below] = BAD_PIXEL
        badpix[above] = BAD_PIXEL
        
        # Convolve with a kernel of length ngrow
        kernel = np.ones(ngrow,dtype="int32")
        badpix = np.convolve(badpix, kernel, mode='same')

        ngoodpix = len(np.where(badpix == GOOD_PIXEL)[0])
        
        niter += 1

    # Transform the line coefficients back to the X range [0:npix-1]
    zstart = intercept - slope
    zslope = slope * xscale

    return ngoodpix, zstart, zslope

def zsc_compute_sigma (flat, badpix, npix):

    # Compute the rms deviation from the mean of a flattened array.
    # Ignore rejected pixels

    # Accumulate sum and sum of squares
    goodpixels = np.where(badpix == GOOD_PIXEL)
    sumz = flat[goodpixels].sum()
    sumsq = (flat[goodpixels]*flat[goodpixels]).sum()
    ngoodpix = len(goodpixels[0])
    if ngoodpix == 0:
        mean = None
        sigma = None
    elif ngoodpix == 1:
        mean = sumz
        sigma = None
    else:
        mean = sumz / ngoodpix
        temp = sumsq / (ngoodpix - 1) - sumz*sumz / (ngoodpix * (ngoodpix - 1))
        if temp < 0:
            sigma = 0.0
        else:
            sigma = math.sqrt (temp)

    return ngoodpix, mean, sigma


def ds9_coords(lc_x,lc_y,uc_x,uc_y,wcs):
    
    lb = SkyCoord(lc_x,lc_y, frame='icrs', unit ='deg')
    ub = SkyCoord(uc_x,uc_y, frame='icrs', unit ='deg')
    
    lb = lb.to_pixel(wcs)
    ub = ub.to_pixel(wcs)
    
    width = (ub[0] - lb[0])
    height = (ub[1] - lb[1])
    
    y1_flip = hosts_overlay(0,lb[1],181/2,181/2,-np.pi)
    y2_flip = hosts_overlay(0,ub[1],181/2,181/2,-np.pi)
    
    #bbox1 = SkyCoord.from_pixel(lower_point[0]-1,y2_flip[1],wcsradio)
    #bbox2 = SkyCoord.from_pixel(lower_point[1]-1,y1_flip[1],wcsradio)


#     x1 = lc_x
#     y1 = lc_y


#     x2 = uc_x
#     y2 = uc_y

    ave_x = (lb[0]+ub[0])/2
    ave_y = (y2_flip[1]+y1_flip[1])/2
    
    poss = SkyCoord.from_pixel(ave_x-1,ave_y,wcsradio)
    poss =  [float(x) for x in poss.to_string().split()]
    
    return poss[0],poss[1],height,width

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
        #if fits.endswith('[0-9].fits'):
        
    #    png = fits.replace('.fits','_logminmax.png')
        png = fits.replace('.fits','_infrared.png')
        #png = osp.join(png_dir,osp.basename(png))
        if (not osp.exists(png)):
            
            fits_0 = re.sub("infrared.png","0_wise_regrid.fits",png)
            fits_1 = re.sub("infrared.png","1_wise_regrid.fits",png)
            
            if (osp.exists(fits_0) and not osp.exists(fits_0.replace('0_wise_regrid.fits','wise_coadd.fits'))):
                cmd = 'ds9 %s -cmap gist_heat -cmap value 0.30 0 -scale log -scale mode minmax -export %s -exit'%(fits_0,png)
                run_command(cmd.split())
                print(fits_0)
                #continue
                
            elif (osp.exists(fits_1) and not osp.exists(fits_1.replace('1_wise_regrid.fits','wise_coadd.fits'))):
                cmd = 'ds9 %s -cmap gist_heat -cmap value 0.30 0 -scale log -scale mode minmax -export %s -exit'%(fits_1,png)
                run_command(cmd.split())
                print(fits_1)
                #continue
                
                
            elif (osp.exists(png.replace('infrared.png','wise_coadd.fits'))):
                cmd = 'ds9 %s -cmap gist_heat -cmap value 0.30 0 -scale log -scale mode minmax -export %s -exit'%(png.replace('infrared.png','wise_coadd.fits'),png)
                run_command(cmd.split())
                print(png.replace('infrared.png','wise_coadd.fits'))

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
                
def vo_get_host_pos(split_fits_dir, download_dir):
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
        ras, decs = w.wcs_pix2world([[cx, cy]], 0)[0]
        #radius = max(fhead['CDELT1'] * cx, fhead['CDELT2'] * cy)
        radius = max(fhead['CDELT1'] * cx, fhead['CDELT2'] * cy)
        pos = "%.6f %.6f"%(ras,decs)
        table = Irsa.query_region(pos, catalog="allwise_p3as_psd", spatial="Box", width=3.00*u.arcmin, selcols="source_id,ra,dec")
        #qstring = "https://irsa.ipac.caltech.edu/SCS?table=allwise_p3as_psd&RA=%.5f&DEC=%.5f&SR=%.6f&format=csv"%(ra,dec,radius)
        #string =  "https://irsa.ipac.caltech.edu/TAP/sync?"
        #query = "QUERY=SELECT+source_id,ra,dec+FROM+allwise_p3as_psd+WHERE+CONTAINS(POINT('J2000',ra,dec),BOX('J2000',%.4f,%.4f,%4f,%.4f))=1&format=csv"%(ras,decs,radius,radius)
        
        #arg = "QUERY=SELECT+ra,dec+FROM+allwise_p3as_psd+WHERE+CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{0:.4f},{0:.4f},{0:.6f}))=1&format=csv".format(ras,decs,radius)
        
        #qstring = "https://irsa.ipac.caltech.edu/SCS?table=allwise_p3as_psd&RA=%.4f&DEC=%.4f&SR=%.6f&format=csv"%(ras,decs,radius)
        #query = qstring+arg
        f_out = "%s.csv"%(osp.join(download_dir, osp.splitext(osp.basename(fname))[0]))
        ascii.write(table,output=f_out,format='csv',overwrite=True)
        print("Writing to %s\n"%f_out)
        
        
        #print(ra,dec)
        #cmd = ["wget","-o",f_out,qstring]
        #qstring = string+query
        #cmd = "wget -O %s %s"%(f_out,qstring)
        #cmd = ["curl", "-o",f_out,qstring]
        #run_command(cmd.split())
        #print("Saved hosts position to %s"%f_out)
        
        
def hosts_overlay(x,y,xo,yo,theta): 
    """
    This function rotates the given position around the given central position and angle.
    Returns an array of rotated positions
    
    
    x,y - coordinates of the point to be rotated
    
    xo,yo - central coordinates to be used as a reference point
    
    theta - rotation angle about the axis, in radians
    
    Returns:
    
    aray[xr,yr] - rotated point
    """
    xr=np.cos(theta)*(x-xo)-np.sin(theta)*(y-yo)   + xo
    yr=np.sin(theta)*(x-xo)+np.cos(theta)*(y-yo)  + yo
    
    return np.array([xr,yr])
        
def deg2world(poss_array,wcs_info):
    """
    This function takes the array of coordinates in pixels
    it converts that to Skycoordinates given the WCS information.
    
    poss_array - array of coordinates in pixels
    
    wcs_info - WCS information from the fits file.
    
    Returns:
    
    poss_array - coordinates in Sky coordinates
    """
    coords_string = []
    data = poss_array
    #print(poss_array)
    poss = data[np.logical_not(pd.isnull(data))]
    rows = len(poss)/2
    
    poss = poss.reshape((int(rows),2))
    
    keep = []
    
    for j in range(len(poss)):
        if poss[j].size!=0:
            keep.append(j)
    
    data_poss = poss[np.array(keep)]
    #print(data_poss)
    coords_world = SkyCoord.from_pixel(data_poss[:,0],poss[:,1],wcs_info)
    coords_string = coords_world.to_string(decimal=True,precision=4)
    
    coords = []
    #print(type(coords_string))
    if isinstance(coords_string,np.ndarray):
        for i in range(len(coords_string)):
            string = coords_string[i]
            for c in string:
                #print(c)
                coords.append(c.split())
            #coords.append()
    if isinstance(coords_string,list):
        for l in coords_string:
            coords.append(l.split())
            #coords.append()    
    
    return np.array(coords)

def host_inbbox(position,x1,y1,x2,y2):
    """
    This function takes an array of coordinates and
    checks whether they are within a given bounding box.
    
    position - array of coordinates.
    # the following depend on the position of the origin chosen in matplotlib
    # for this can, origin top-left
    
    x1,y1 - top-left point of the bounding box
    
    x2,y2 - bottom-right point of the bounding box
    
    Returns:
    
    bool - True if source is within the box, else False
    """
    
    ra_cond = np.logical_and(position[0]>x1,position[0]<(x1 + (x2 - x1)))
    dec_cond = np.logical_and(position[1]>y1,position[1]<(y1 + (y2 - y1)))
    
    return np.logical_and(ra_cond,dec_cond)

def source_inbbox(position,x1,y1,x2,y2):
    """
    This function takes an array of coordinates and
    checks whether they are within a given bounding box this time return the index.
    
    position - array of coordinates.
    # the following depend on the position of the origin chosen in matplotlib
    # for this can, origin top-left
    
    x1,y1 - top-left point of the bounding box
    
    x2,y2 - bottom-right point of the bounding box
    
    Returns:
    
    int index - index of the source found to be within the box of the given array.
    """
    
    ra_cond = np.logical_and(position[0]>x1,position[0]<(x1 + (x2 - x1)))
    dec_cond = np.logical_and(position[1]>y1,position[1]<(y1 + (y2 - y1)))
    
    i_true = np.where(np.logical_and(ra_cond,dec_cond)==True)[0]
    
    return i_true
    
def plot_com_data(predicted_sources,background_img,fits_data,ax,position_in_pix,host_id,source_positon_pix):
    """
    This function takes an array of predictions from CLARAN, image data to be overlaid with preditions,
    axes, position of IR and radio sources in pixels. Returns central positions computed using the bounding
    box and consequently the properties of the estimated IR hosts.
    
    predicted_sources - datafram containing predictions from CLARAN.
    background_img - image data to be overlaid with predictions
    ax - axes already instantiated 
    position_in_pix - array of positions of sources in pixels
    host_id - array of IDs of hosts
    source_position_pix - central position of the cutout in pix
    
    Returns:
    
    RC - radio centre, center-of-mass of the pixel data within the bounding box 
    poss_host -  position of the host determined by the given central coordinates of the catalogue (gmrt)
    poss_rc - position of the host determined by the RC 
    ir_host_ids - IR host IDs estimated by central position of the FOV 
    ir_host_ids_rc - IR host IDs estimated by RC 
    found_host - bool, True if host was found, False otherwise.
    
    """
    CP = []
    RC = []
    
    ir_host_ids = []
    ir_host_ids_rc = []
    
    poss_host = []
    poss_host_gmrt = []
    poss_host_rc = []
    
    poss_host_pix = []
    poss_centre_pix = []
    poss_rc = []
    
    found_host = False
    found_center = False
    pred_sources = predicted_sources
    im = background_img
    position_pix = position_in_pix
    source_poss_pix = source_positon_pix
    
    # Stats of the FITS file
    mean, median, std = sigma_clipped_stats(fits_data)
    
    # boundary values of the 3-sigma clip
    lower_thresh = median-3*std
    upper_thresh = median+3*std

    
    # Check if ClaRAN predicted source
    if(len(pred_sources)!=0):

        for i in range(len(pred_sources)):
            
            # predicted classes
            cl = pred_sources['class'][i]

            # Probability score
            prob_score = pred_sources['scores'][i]

            # Bounding box coordinates from ClaRAN were scaled up to a given scale, in this case 600
            bbox = np.array([pred_sources['x1'][i],pred_sources['y1'][i],pred_sources['x2'][i],pred_sources['y2'][i]])

            # So they need to be down-scaled
            bbox /= (float(600)/float(im.shape[0]))
            
            # dimenstions of the bounding box
            width = int(bbox[2]-bbox[0])
            height = int(bbox[3]-bbox[1])
            
            # rotate the bounding box coordinates to match the FITS file format
            y1_flip = hosts_overlay(0,bbox[1],im.shape[0]/2,im.shape[0]/2,np.pi)
            y2_flip = hosts_overlay(0,bbox[3],im.shape[0]/2,im.shape[0]/2,np.pi)
            
            # indexing, FITS is 1-based index, whereas Python is 0-based.
            ymin_index = y2_flip[1]-1
            xmin_index = bbox[0]-1
            
            # get the region of interest from the cutout
            fits_roi = fits_data[int(ymin_index):int(y1_flip[1]),int(xmin_index):int(bbox[2])]
            
            # clip all the values below the upper boundary values
            # there's a high peak at small flux values, it is the ‘background’ signal, 
            # the low peak at high flux values is where ‘source’ signal appears. Thus clipping using the upper boundary, 
            # we recover source signal only
            fits_roi[fits_roi<upper_thresh] = 0.0
            
            #calculate the center of mass
            #cp_fits_roi[cp_fits_roi < 0] = 0
            com_fits_radio = np.array(center_of_mass(fits_roi)) #+ np.array([bbox[1],xmin_index])
            #com_fits_radio =np.array(center_of_mass(cp_fits_roi)) #+ np.array([bbox[1],xmin_index])
            
            # convert the coordinates to that of matplotlib, noting that FITS is column-major format, 
            # therefore x is represented by y and vice versa.
            cp_fits = np.array([bbox[3],xmin_index]) + np.array([-com_fits_radio[0],com_fits_radio[1]])
            
            rc = np.zeros(2)
            
            if np.isnan(com_fits_radio[0]): # RuntimeWarning: 
                                            # invalid value encountered in double_scalars for dir in range(input.ndim)]
                                            # this happens when the region has an average value of zero
                        
                # therefore we assign central coordinates of the bounding box
                rc = np.array([bbox[3],xmin_index]) + np.array([-width/2,height/2])
            else:
                rc = np.array([cp_fits[1],cp_fits[0]])
            
            
            # Add rectangle bounding box on the axes
            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                bbox[2] - bbox[0],
                bbox[3] - bbox[1], fill=False,
                edgecolor='blue', linewidth=1.0)
            )

            ax.text(bbox[0], bbox[1] - 2,
                        '{:s} {:.2f}'.format(cl, prob_score),
                        bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                        fontsize=16, color='white')
            

            # Get indices of IR host in this FOV
            #print(position_pix[0])
            ind = source_inbbox(position_pix,bbox[0],bbox[1],bbox[2],bbox[3])
            
            # host properties of sources within the bounding box
            host_name = host_id[ind]
            host_ra = position_pix[0][ind]
            host_dec = position_pix[1][ind]

            poss_source_dis = np.zeros(len(host_ra))
            poss_rc_dis = np.zeros(len(host_ra))

            # check whether there are host within the box
            if (len(poss_source_dis)!=0):
                
                found_host = True
                for i in range(len(host_ra)):
                    pos = [host_ra[i],host_dec[i]]

                    min_dis_source_rc = ((rc[0]-pos[0])**2 + (rc[1]-pos[1])**2)**2

                    poss_rc_dis[i] = min_dis_source_rc

                # get the index of the closest source and appeand its properties
                i_rc = np.where(poss_rc_dis==min(poss_rc_dis))[0]

                ir_host_ids_rc.append(host_name[i_rc])
                #print(ir_host_ids_gc)
                poss_rc.append([host_ra[i_rc],host_dec[i_rc]])
            
            # Get indices of IR host in this FOV
            gmrt_host = source_inbbox(source_poss_pix,bbox[0],bbox[1],bbox[2],bbox[3])
            #found_host_center = host_found
            
            # check whether there are host within the box
            if (len(poss_source_dis)!=0):
                for i in gmrt_host:
                    gmrt_ra = source_poss_pix[i]
                    gmrt_dec = source_poss_pix[i]
                    pos_gmrt = [gmrt_ra,gmrt_dec]
                
                    #if (len(poss_source_dis)!=0):

                    for j in range(len(host_ra)):
                        pos = [host_ra[j],host_dec[j]]
                        min_dis_source = ((pos_gmrt[0]-pos[0])**2 + (pos_gmrt[1]-pos[1])**2)**2
                        poss_source_dis[j] = min_dis_source
                            
                            
                    i_host = np.where(poss_source_dis==min(poss_source_dis))[0][0]
                    ir_host_ids.append(host_name[i_host])
                    poss_host.append([host_ra[i_host],host_dec[i_host]])

            RC.append(rc)
            
    return RC, poss_host, poss_rc, ir_host_ids, ir_host_ids_rc,found_host



def overlay_centroids_data(pred_pos,bg_png,fits_f,hosts_file,radio_catalog,img_type):
    """
    This fuction overlays the given position of the IR hosts given the
    output from ClaRAN and use the image to get the estimate of geometric centre and center-of-mass of pixels.
    
    pred_pos - a csv file containg output from ClaRAN; predicted class,
                                                        probability value,
                                                        bounding box coordinates - x1, y1, x2, y2.
                where bottom-left corner coordinates are x1,y1 and top right corner coordinates are x2,y2
                
    fits_file - FITS image of the radio source
    bg_png - PNG of radio/IR image used on ClaRAN
    hosts_file - a csv file containg position (RA,Dec) of the IR source from the catalog 
    
    Returns - PNG image showing ClARAN output overlaid with IR hosts position
    """
    # Read ClaRAN output from csv file
    sources = ascii.read(pred_pos)
    #sources = np.array([re.sub("'","",sources.replace("b",""))
    
    # Read the IR image
    im = cv2.imread(bg_png)
    
    # Output filename
    Output_overlay = ""
    
    sid = 0
    cd = osp.basename(pred_pos)
    ClaRAN_file = pred_pos
    
    # Get source ID from file name. Need to be changed to be more general!
    if (img_type=="D1"):
        Output_overlay = pred_pos.replace(".csv","_host_overlay.png")
        cd = cd.replace("gmrt_en1w610_clipped_","")
        cd = cd.replace("_D1.csv","")
        sid = int(cd)
        
    elif (img_type=="D4"):
        Output_overlay = pred_pos.replace(".csv","_D4_host_overlay.png")
        cd = cd.replace("gmrt_en1w610_clipped_","")
        cd = cd.replace(".csv","")
        sid = int(cd)   
    
    # Read position of IR hosts
    data = ascii.read(hosts_file)
    data = data.to_pandas()
        
    hosts_ids = data["source_id"].get_values() # Get data from pandas datafram as array
    #hosts_ids = np.array([re.sub("'","",s.replace("b","")) for s in hosts_ids])
    ra = data["ra"].get_values()
    dec = data["dec"].get_values()
    
    #print(hosts_ids)
    # Transforming the coordinates to standard coordinates
    position = SkyCoord(ra,dec, frame='icrs', unit ='deg')

    # Get header information from FITS file to transform pixel scales
    with fits.open(fits_f) as file:
        hduradio = file[0]
        wcsradio = WCS(hduradio.header, naxis=2)
        fits_data = hduradio.data
        
        xscale = abs(hduradio.header['CDELT1'])
        yscale = abs(hduradio.header['CDELT2'])
            
        xpix = hduradio.header['NAXIS1']
        ypix = hduradio.header['NAXIS2']
            
        xsize = xscale*xpix
        ysize = yscale*ypix
        
    #threshold = detect_threshold(fits_data, snr=3)
    
    #radio_data = detect_sources(fits_data, threshold, npixels=5)
    #radio_data = sigma_clip(fits_data, sigma=3.0, maxiters=5) # this function return the original 2D array and the mask, where clipped pixel are True.
                                                              # thus pixels the pixel densities above 3-sigma are True in this array 
    
    #radio_data[radio_data.mask==False] = 0 # assign a value of zero to pixels that are False, thus to be ignored when calculating the center-of-mass 
        
    # Transform coordinates to pixel
    position_pix = position.to_pixel(wcsradio)
    
    in_img = np.logical_and(position_pix[0]<181, position_pix[1]<181)
    position_pix = np.array([position_pix[0][in_img],position_pix[1][in_img]])
    
    y_pix = hosts_overlay(0,position_pix[1],im.shape[0]/2,im.shape[0]/2,-np.pi) # invert y coordinates in case of IR image.
    position_pix = np.array([position_pix[0],y_pix[1]])
    
    
    source_ra = 0.0
    source_dec = 0.0
    source = ""
    t = parse_single_table(radio_catalog)
    catalog_data = t.to_table(use_names_over_ids=True)
    
    # Loop throught the catalog for information
    for i in range(len(catalog_data['RA_deg'])):
        source_id = catalog_data['Source_id'][i]
        #print(type(source_id),type(sid))
        if source_id == sid: 
            source = str(source_id)
            source_ra = catalog_data['RA_deg'][i]
            source_dec = catalog_data['Dec_deg'][i]
            


    source_poss = SkyCoord(source_ra,source_dec, frame='icrs', unit ='deg')
    source_poss_string = source_poss.to_string("decimal")
    
    
    source_poss_pix = source_poss.to_pixel(wcsradio)
    source_poss_pix_coords = source_poss_pix #
    
    # Plot the results
    fig = plt.figure(figsize=(10,10),dpi=150)
    #fig.set_size_inches(600 / my_dpi, 600 / my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_xlim([0, im.shape[0]])
    ax.set_ylim([im.shape[0],0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_aspect('equal')
    #im = cv2.resize(im, (600, 600))
    im = im[:, :, (2, 1, 0)]
    
    #print(sources)
    #print(source_poss_pix)
    if(len(sources)!=0):
        
        if (img_type=="D1"):
            position_pix = np.array([position_pix[0],position_pix[1]])

        #else:
        #    position_pix = np.array([position_pix[0],y_pix[1]])

        #CM,GC,poss_host,poss_centre_pix,poss_centre_geo,ir_host_ids, ir_host_ids_pc, ir_host_ids_gc,found_host = plot_bbox_data(sources,im,ax,position_pix,hosts_ids,source_poss_pix)
        
        RC,pybdsf_host,rc_host,ir_host_ids_pybdsf, ir_host_ids_rc,found_host = plot_com_data(sources,im,fits_data,ax,position_pix,hosts_ids,source_poss_pix)
        
        #print(CM)
        #CM = np.array(CM)
        #cm = deg2world(CM,wcsradio)  #[l.split() for l in cm_string]
        #print()


        RC = np.array(RC)
        #rc = filter(lambda v: v==v, GC)
        #print(rc)
        rc = deg2world(RC,wcsradio)  #[l.split() for l in gc_string]

        pybdsf_hosts = np.array(pybdsf_host)


        #centre_pix = np.array(poss_centre_pix)
        #keep = []
    
        #for j in range(len(poss_centre_geo)):
        #    if poss_centre_geo[j].size!=0:
        #        keep.append(j)

        #poss = poss[np.array(keep)]
        
        rc_hosts = np.array(rc_host)


        #print(poss_host)
        source_ID = "Source_ID: " + source
        ax.annotate(source_ID, xy=(0.45,0.97), xycoords='axes fraction', color='w', fontsize=14)

        coords = "RA, Dec: {:.4f} {:.4f} [degrees] (center)".format(source_ra,source_dec)
        ax.annotate(coords, xy=(0.45,0.95),xycoords='axes fraction', color='w', fontsize=14)


        #angular_scale = xsize
        show_img_size = im.shape[0]

        scal = "scale: %.2f' x %.2f' (%.2fx%.2f pix)"%((xsize)*60,(xsize)*60,show_img_size,show_img_size)
        ax.annotate(scal, xy=(0.02,0.97), xycoords='axes fraction', color='w', fontsize=14)

        patch_contour = fuse(fits_f, im, None, sigma_level=4, mask_ir=False,
                                 get_path_patch_only=True)

        ax.scatter(RC[:,0],RC[:,1],s=220,c='blue',marker='+',label='RC')
        ax.scatter(float(source_poss_pix_coords[0]),float(source_poss_pix_coords[0]),s=220,c='lime',marker='+',label='PyBDSF')    
        #ax.scatter(CM[:,0],CM[:,1],s=200,c='lime',marker='+',label='Pixel centroid (PC)')
        


        if (bg_png.endswith("_logminmax.png")):
            ax.imshow(im)
            ax.add_patch(patch_contour)

            ax.scatter(position_pix[0],position_pix[1],s=220,c='black',marker='x',label='IR Source')
            if found_host:
                rc_hosts_coords = deg2world(rc_hosts,wcsradio)
                ax.scatter(rc_hosts[:,0],rc_hosts[:,1],s=220,c='blue',marker='o',alpha=0.45,label='IR host (RC)')

                #centre_pix_coords = deg2world(centre_pix,wcsradio)
                #ax.scatter(centre_pix[:,0],centre_pix[:,1],s=200,c='lime',marker='o',alpha=0.35,label='potential IR host (PC)')

                #print(len(hosts))
                if (len(pybdsf_hosts)!=0):
    #                 print(hosts)
                    pybdsf_host_coords = deg2world(pybdsf_hosts,wcsradio)
                    ax.scatter(pybdsf_hosts[:,0],pybdsf_hosts[:,1],s=220,c='lime',marker='v',alpha=0.45,label='IR host (PyBDSF)')
                    #print([float(source_poss_string.split()[0]),float(source_poss_string.split()[1])])
                    #print(ir_host_ids_pc)
                    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", fontsize=20, borderaxespad=0)
                    #fig.savefig(Output_overlay)
                    fig.savefig(Output_overlay,bbox_inches='tight')
                    ax.cla()
                    return source,[float(source_poss_string.split()[0]),float(source_poss_string.split()[1])],rc,ir_host_ids_pybdsf,pybdsf_host_coords,ir_host_ids_rc,rc_hosts_coords
                else:
                    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", fontsize=20, borderaxespad=0)
                    fig.savefig(Output_overlay,bbox_inches='tight')
                    ax.cla()
                    return source,[float(source_poss_string.split()[0]),float(source_poss_string.split()[1])],rc,['-'],np.array([['-1.0','-1.0']]),['-'],np.array([['-1.0','-1.0']])
            else:
                ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", fontsize=20, borderaxespad=0)
                fig.savefig(Output_overlay,bbox_inches='tight')
                ax.cla()
                return source,[float(source_poss_string.split()[0]),float(source_poss_string.split()[1])],rc,['-'],np.array([['-1.0','-1.0']]),['-'],np.array([['-1.0','-1.0']])

        elif (bg_png.endswith("_infrared.png") and type(patch_contour)!=None):
            #ax.imshow(np.flipud(im), origin='lower') # because the pixel coordinates start at (0,0) whereas
                                                     # FITS images starts at (1,1)
            #ax.imshow(im, origin='lower')
            #ax.yaxis_inverted()
            ax.set_ylim([im.shape[0],0])
            ax.add_patch(patch_contour)
            ax.imshow(im, origin='upper')
            #print(type(patch_contour))
            ax.scatter(position_pix[0],position_pix[1],s=220,c='black',marker='x',label='IR Source')
            if found_host:
                rc_hosts_coords = deg2world(rc_hosts,wcsradio)
                ax.scatter(rc_hosts[:,0],rc_hosts[:,1],s=220,c='blue',marker='o',alpha=0.45,label='IR host (RC)')

                #centre_pix_coords = deg2world(centre_pix,wcsradio)
                #ax.scatter(centre_pix[:,0],centre_pix[:,1],s=200,c='lime',marker='o',alpha=0.35,label='potential IR host (PC)')

                #print(len(hosts))
                if (len(pybdsf_hosts)!=0):
    #                 print(hosts)
                    pybdsf_host_coords = deg2world(pybdsf_hosts,wcsradio)
                    ax.scatter(pybdsf_hosts[:,0],pybdsf_hosts[:,1],s=220,c='lime',marker='v',alpha=0.45,label='IR host (PyBDSF)')
                    #print([float(source_poss_string.split()[0]),float(source_poss_string.split()[1])])
                    #print(ir_host_ids_pc)
                    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", fontsize=20, borderaxespad=0)
                    fig.savefig(Output_overlay,bbox_inches='tight')
                    ax.cla()
                    return source,[float(source_poss_string.split()[0]),float(source_poss_string.split()[1])],rc,ir_host_ids_pybdsf,pybdsf_host_coords,ir_host_ids_rc,rc_hosts_coords
                else:
                    ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", fontsize=20, borderaxespad=0)
                    fig.savefig(Output_overlay,bbox_inches='tight')
                    ax.cla()
                    return source,[float(source_poss_string.split()[0]),float(source_poss_string.split()[1])],rc,['-'],np.array([['-1.0','-1.0']]),['-'],np.array([['-1.0','-1.0']])

            else:
                ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", fontsize=20, borderaxespad=0)
                fig.savefig(Output_overlay,bbox_inches='tight')
                ax.cla()
                return source,[float(source_poss_string.split()[0]),float(source_poss_string.split()[1])],rc,['-'],np.array([['-1.0','-1.0']]),['-'],np.array([['-1.0','-1.0']])
        else:
            print("error...")
            # Do not do anything
    plt.show();
            
                
def catalog_data(pred_png_dir,ir_png_dir,fits_dir,host_pos_dir,radio_catalog,img_type):
    
    suffix = ""
    output_catalog = ""
    output_suffix = ""
   
    if img_type=="D1":
        #print(img_type)
        suffix = "*_logminmax_pred.png"
        output_catalog = "ClaRAN_D1_final_catalog_v4.csv"
        csv_ext = "_D1.csv"
        output_suffix = "_logminmax.png"
        
    elif img_type=="D4":
        suffix = "*_infraredctmask_pred.png"
        output_catalog = "ClaRAN_D4_final_catalog_v5.csv"
        csv_ext = ".csv"
        output_suffix = "_infrared.png"
        
    else:
        sys.exit(1)
    
    ClaRAN_output_files = glob.glob(osp.join(pred_png_dir,suffix))
    output_catalog = osp.join(pred_png_dir,output_catalog)

    #output_f = osp.join(pred_png_dir,"ClaRAN_catalog_D4.asc")
    with open(output_catalog, "w") as f:
        f.write("#Source_ID,PyBDSF_RA,PyBDSF_Dec,Class,Scores,x1,y1,x2,y2,RC_RA,RC_Dec,PyBDSF_host,PyBDSF_host_RA,PyBDSF_host_Dec,RC_host,RC_host_RA,RC_host_Dec,file_path\n")
    
        for file in ClaRAN_output_files:
            sfx = suffix.replace("*","")
            #print(file)

            # Read the IR image
            #print(suffix.replace("*",""))
            img = osp.basename(file.replace(sfx,output_suffix))
            ir_img = osp.join(ir_png_dir,img)#
            im = cv2.imread(ir_img)
            
            hosts_file = osp.basename(file.replace(sfx,".csv"))
            hosts_file = osp.join(host_pos_dir,hosts_file)
            
            

            output_csv = file.replace(sfx,csv_ext)
            #nlines = os.system('cat %s | wc -l'%output_csv)
            #output_csv = osp.join(ir_png_dir,output_csv)
            if not osp.exists(output_csv):
                #print("File does not exist...")
                continue
                
               
          
            print("File exists...")
            ClaRAN_prediction_file = output_csv
            #print(output_csv)

            fits_name = osp.basename(file.replace(sfx,".fits"))
            fits_f = osp.join(fits_dir,fits_name)


            with fits.open(fits_f) as file:
                hduradio = file[0]
                wcsradio = WCS(hduradio.header, naxis=2)
                #print(wcsradio)
            print(output_csv)
            try:
                pred_sources = ascii.read(output_csv)
                pred_sources = pred_sources.to_pandas()
                #print(output_csv)#type(output_csv),type(ir_img),type(radio_catalog),type(img_type))
                if (os.stat(hosts_file).st_size!=0 and len(pred_sources)!=0):
                    #print(type(pred_sources))
                    source_name,source_pos,geometric_centre,host_id_gmrt,hosts_position_gmrt,host_id_geo,centre_geo_position = overlay_centroids_data(output_csv,ir_img,fits_f,hosts_file,radio_catalog,img_type)

                    #print(output_csv)
                    #  pred_sources = ascii.read(output_csv)



                    for i in range(len(pred_sources)):

                        cl = pred_sources['class'].iloc[i]
                        #pred_class.append(cl)


                        prob_score = pred_sources['scores'].iloc[i]
                        #pred_class.append(prob_score)


                        # Bounding box coordinates from ClaRAN were scaled up to a given scale, in this case 600
                        bbox = np.array([pred_sources['x1'].iloc[i],pred_sources['y1'].iloc[i],pred_sources['x2'].iloc[i],pred_sources['y2'].iloc[i]])

                        # So they need to be down-scaled
                        bbox /= (float(600)/float(im.shape[0]))

                        bbox_coords_world1 = SkyCoord.from_pixel(bbox[0],bbox[1],wcsradio)
                        bbox_coords1_string = bbox_coords_world1.to_string(decimal=True,precision=4)
                        bbox_coords1 = bbox_coords1_string.split()

                        bbox_coords_world2 = SkyCoord.from_pixel(bbox[2],bbox[3],wcsradio)
                        bbox_coords2_string = bbox_coords_world2.to_string(decimal=True,precision=4)
                        bbox_coords2 = bbox_coords2_string.split()



                        #print(source_name,centre_pix_position)
                        fname = "gmrt_en1w610_clipped_%s_%s"%(source_name,suffix.replace("*_",""))
                        fname = osp.join(pred_png_dir,fname)
                        print("Saving data to catalog...\n")
                        try:
                            f.write("{},{:0.4f},{:0.4f},{},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{},{:0.4f},{:0.4f},{},{:0.4f},{:0.4f},{}\n".format(source_name,
                                                                                                                     source_pos[0],
                                                                                                                     source_pos[1],
                                                                                                                     cl,
                                                                                                                     prob_score,
                                                                                                                     float(bbox_coords1[0]),
                                                                                                                     float(bbox_coords1[1]),
                                                                                                                     float(bbox_coords2[0]),
                                                                                                                     float(bbox_coords2[1]),
                                                                                                                     float(geometric_centre[i][0]),
                                                                                                                     float(geometric_centre[i][1]),
                                                                                                                     host_id_gmrt[0],
                                                                                                                     float(hosts_position_gmrt[0][0]),
                                                                                                                     float(hosts_position_gmrt[0][1]),
                                                                                                                     host_id_geo[i][0],
                                                                                                                     float(centre_geo_position[i][0]),
                                                                                                                     float(centre_geo_position[i][1]),fname))

                        except IndexError as error:
                            f.write("{},{:0.4f},{:0.4f},{},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{:0.4f},{},{:0.4f},{:0.4f},{},{:0.4f},{:0.4f},{}\n".format(source_name,
                                                                                                                     source_pos[0],
                                                                                                                     source_pos[1],
                                                                                                                     cl,
                                                                                                                     prob_score,
                                                                                                                     float(bbox_coords1[0]),
                                                                                                                     float(bbox_coords1[1]),
                                                                                                                     float(bbox_coords2[0]),
                                                                                                                     float(bbox_coords2[1]),
                                                                                                                     float(geometric_centre[i][0]),
                                                                                                                     float(geometric_centre[i][1]),
                                                                                                                     '-',
                                                                                                                     float(hosts_position_gmrt[0][0]),
                                                                                                                     float(hosts_position_gmrt[0][1]),
                                                                                                                     '-',
                                                                                                                     float(-1.0),
                                                                                                                     float(-1.0),fname))

                else:
                    print("Unable to guess table format...")
                    continue
            except csv.Error:
                continue


                
def vis_data(pred_png_dir,ir_png_dir,fits_dir,host_pos_dir,radio_catalog,img_type):
    
    suffix = ""
    output_catalog = ""
    output_suffix = ""
   
    if img_type=="D1":
        print(img_type)
        suffix = "*_logminmax_pred.png"
        output_catalog = "ClaRAN_catalog_D1.asc"
        csv = "_D1.csv"
        output_suffix = "_logminmax.png"
        
    elif img_type=="D4":
        suffix = "*_infraredctmask_pred.png"
        output_catalog = "ClaRAN_catalog_D4.asc"
        csv = ".csv"
        output_suffix = "_infrared.png"
        
    else:
        sys.exit(1)
    
    ClaRAN_output_files = glob.glob(osp.join(pred_png_dir,suffix))
    output_catalog = osp.join(pred_png_dir,output_catalog)
#     t = parse_single_table(radio_catalog)
#     data = t.to_table(use_names_over_ids=True)
    
    for file in ClaRAN_output_files:
        #print(file)

        # Read the IR image
        img = osp.basename(file.replace(suffix.replace("*",""),output_suffix))
        ir_img = osp.join(ir_png_dir,img)#
        im = cv2.imread(ir_img)
            
        hosts_file = osp.basename(file.replace(suffix.replace("*",""),".csv"))
        hosts_file = osp.join(host_pos_dir,hosts_file)

        output_csv = file.replace(suffix.replace("*",""),csv)
        if not osp.exists(output_csv):
            #print("File does not exist...")
            continue
                
        print("File exists...")
        ClaRAN_prediction_file = output_csv
        #print(output_csv)
            
        fits_name = osp.basename(file.replace(suffix.replace("*",""),".fits"))
        fits_f = osp.join(fits_dir,fits_name)
            
                                        
        with fits.open(fits_f) as file:
            hduradio = file[0]
            wcsradio = WCS(hduradio.header, naxis=2)
            #print(wcsradio)
            pred_sources = ascii.read(output_csv)
        if (not os.stat(hosts_file).st_size==0 and len(pred_sources)!=0):
                #print(type(pred_sources))
                #source_name,source_pos,com_pix,geometric_centre,host_id_gmrt,hosts_position_gmrt,host_id_pix,centre_pix_position,host_id_geo,centre_geo_position = overlay_centroids_data(output_csv,ir_img,fits_f,hosts_file,radio_catalog,img_type)
                
                overlay_centroids_data(output_csv,ir_img,fits_f,hosts_file,radio_catalog,img_type)


        else:
            print("Unable to guess table format...")
            continue

    
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def unique_catalog(claran_catalog,original_img_size,output_name):
    
    """Produce a dataframe contating uninque detections by applying NMS alogrithm on the detected sources"""
    
    
    radius = 2.7777777854E-4*(original_img_size) # This is the product CDELT value from FITS Header and image size of the original input cutout
    
    
    data = ascii.read(claran_catalog) # Read the output catalog from CLARAN
    
    df = data.to_pandas() # Convinient for tasks to be performed
    
    match = 0
    nms_keep = []
    dets = []
    fov = []

    for i in range(len(df)):
        fov = []

        ra = df.iloc[i]["RA_deg"]
        dec = df.iloc[i]["Dec_deg"]

        for j in range(len(df)):
            row = df.iloc[j]

            x = row["RA_deg"] - ra
            y = row["Dec_deg"] - dec

            distance = np.sqrt(x*x + y*y) # calculate the distance

            if distance<radius: # take only cataloged sources that are in the same FOV
                match+=1
                fov.append(j) 

        if match>1: # in some FOVs only one source is found, skip it.
            dets = []
            fov = np.array(fov)
            same_fov_data = df.iloc[fov] # retrieve data that belongs to the same FOVs

            for j in range(len(same_fov_data)): 
                dets.append([same_fov_data["x1"].iloc[j], same_fov_data["y1"].iloc[j], same_fov_data["x2"].iloc[j], same_fov_data["y2"].iloc[j], same_fov_data["scores"].iloc[j]])

            keep = np.array(py_cpu_nms(np.array(dets),0.5)) #Use ClaRAN Python NMS Baseline to get indices of entries to keep
            for ix in keep:
                nms_keep.append(fov[ix]) # append indices to a list
                
    nms_keep = np.unique(np.array(nms_keep)) # Take only non-repeating index values
    keep_data = df.iloc[nms_keep]
    keep_data_sorted = keep_data.sort_values(by="Source_id")
    print("Saving catalog to %s"%output_name)
    #keep_data_sorted.to_csv(output_path,float_format='0.4f',index=False)
    tb = Table.from_pandas(keep_data_sorted)
    ascii.write(tb,output_name,format='csv')
    
def ds9_bbox_reg(lc_x,lc_y,uc_x,uc_y,wcs):
    
    lb = SkyCoord(lc_x,lc_y, frame='icrs', unit ='deg')
    ub = SkyCoord(uc_x,uc_y, frame='icrs', unit ='deg')
    
    lb = lb.to_pixel(wcs)
    ub = ub.to_pixel(wcs)
    
    width = (ub[0] - lb[0])
    height = (ub[1] - lb[1])
    
    y1_flip = hosts_overlay(0,lb[1],181/2,181/2,-np.pi)
    y2_flip = hosts_overlay(0,ub[1],181/2,181/2,-np.pi)
    
    #bbox1 = SkyCoord.from_pixel(lower_point[0]-1,y2_flip[1],wcsradio)
    #bbox2 = SkyCoord.from_pixel(lower_point[1]-1,y1_flip[1],wcsradio)


#     x1 = lc_x
#     y1 = lc_y


#     x2 = uc_x
#     y2 = uc_y

    ave_x = (lb[0]+ub[0])/2
    ave_y = (y2_flip[1]+y1_flip[1])/2
    
    poss = SkyCoord.from_pixel(ave_x-1,ave_y,wcs)
    poss =  [float(x) for x in poss.to_string().split()]
    
    return poss[0],poss[1],height,width