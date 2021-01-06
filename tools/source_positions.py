
import sys
import os, errno
import os.path as osp
import glob
import re
import logging
handler = logging.FileHandler(filename='important_log.log', mode='a')
sys.path.insert(0,'/users/cmofokeng/rgz_rcnn/tools/')

import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib as mpl
import numpy as np
from astropy.io import ascii
from astropy.table import unique
from astropy.io.votable import parse_single_table
#from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground

from astropy.coordinates import SkyCoord
from scipy.ndimage import center_of_mass,mean
from astropy.stats import SigmaClip,sigma_clipped_stats
import matplotlib.patches as patches
import preprocess_toolset as ppt
from photutils import centroid_com, centroids
from astropy.wcs import WCS
from astropy.io import fits
from fuse_radio_ir import fuse, _get_contour
import make_contours
from matplotlib.path import Path
from functions import *
from photutils import detect_threshold, detect_sources, source_properties

import cv2

def overlay_centroids_unique(df):
    """
    This fuction overlays the given position of the IR hosts given the
    output from ClaRAN and use the image to get the estimate of geometric centre and center-of-mass of pixels.
    
    pred_pos - a csv file containg output from ClaRAN; predicted class,probability value,bounding box coordinates - x1, y1, x2, y2.
                where bottom-left corner coordinates are x1,y1 and top right corner coordinates are x2,y2
    fits_file - FITS image of the radio source
    bg_png - PNG of radio/IR image used on ClaRAN
    hosts_file - a csv file containg position (RA,Dec) of the IR source from the catalog 
    
    Returns - PNG image showing ClaRAN output overlaid with IR hosts position
    """
    
    # Read the background image
    pred_file = osp.join("/idia/users/cmofokeng/data/gmrt/output_gmrt/","gmrt_en1w610_clipped_"+str(df["Source_ID"].iloc[0])+"_infraredctmask_pred.png")
    img = osp.join("/idia/users/cmofokeng/data/gmrt/split_fits/test_1deg",osp.basename(pred_file).replace("_infraredctmask_pred.png","_infrared.png"))

    im = cv2.imread(img)

    # Output filename
    Output_overlay = osp.join("/idia/users/cmofokeng/data/gmrt/output_gmrt/",osp.basename(img).replace("_infrared.png","_D4_hosts.png"))

    # Read position of IR hosts
    #data = np.loadtxt(hosts_file,delimiter=',',skiprows=1)
    host_file = osp.join("/idia/users/cmofokeng/data/gmrt/split_fits/position/",osp.basename(img).replace("_infrared.png",".csv"))
    # Read position of IR hosts
    data = ascii.read(hosts_file)
    data = data.to_pandas()
        
    hosts_ids = data["source_id"].to_numpy() # Get data from pandas dataframe as array
    #hosts_ids = np.array([re.sub("'","",s.replace("b","")) for s in hosts_ids])
    ra = data["ra"].to_numpy()
    dec = data["dec"].to_numpy()

    # Transforming the coordinates to standard coordinates
    position = SkyCoord(ra,dec, frame='fk5', unit ='deg')
    #center = SkyCoord(RAs,Decs, frame='icrs', unit ='deg')

    # Get header information from FITS file to transform pixel scales
    #wcsradio = WCS(fits_f, naxis=2)
    fits_f = osp.join("/idia/users/cmofokeng/data/gmrt/split_fits/test_1deg",osp.basename(host_file).replace(".csv",".fits"))
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


    # Transform coordinates to pixel
    position_pix = position.to_pixel(wcsradio);
    #print(position_pix)

    y_pix = hosts_overlay(0,position_pix[1],im.shape[0]/2,im.shape[0]/2,-np.pi)
    position_pix = np.array([position_pix[0],y_pix[1]])

    #print(len(position_pix))
    in_img = np.logical_and(position_pix[0]<181.00, position_pix[1]<181.00)

    #print(in_img)

    #sources_in_img = source_inbbox(position_pix,0.0,)
    position_pix = np.array([position_pix[0][in_img],position_pix[1][in_img]])

    # Plot the results
    fig = plt.figure(figsize=(10,10),dpi=150)
    #fig.set_size_inches(600 / my_dpi, 600 / my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_xlim([0, im.shape[0]])
    #ax.set_ylim([0,im.shape[0]])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_aspect('equal')
    #im = cv2.resize(im, (600, 600))
    im = im[:, :, (2, 1, 0)]    

    ir_host_ids = []
    ir_host_ids_rc = []
    ir_host_ids_pybdsf = []

    poss_host = []
    rc_poss = []
    pybdsf_poss = []

    poss_host_pix = []
    poss_centre_pybdsf = []
    poss_centre_rc = []

    found_host = False
    found_host_center = False

    pred_sources = df
    #ix = pred_df.index.values
    source_ra = pred_sources['PyBDSF_RA'].iloc[0]
    source_dec = pred_sources['PyBDSF_Dec'].iloc[0]
    for i in range(len(pred_sources)):
        source = pred_sources['Source_ID'].iloc[i]

        cl = pred_sources['Class'].iloc[i]

        prob_score = pred_sources['Scores'].iloc[i]

        pybdsf = SkyCoord(pred_sources['PyBDSF_RA'].iloc[i],pred_sources['PyBDSF_Dec'].iloc[i], frame='fk5', unit ='deg')
        pybdsf_pix = pybdsf.to_pixel(wcsradio)
        pybdsf_y = hosts_overlay(0,pybdsf_pix[1],im.shape[0]/2,im.shape[0]/2,-np.pi)
        pybdsf_pix = np.array([pybdsf_pix[0],pybdsf_y[1]])


        rc = SkyCoord(pred_sources['RC_RA'].iloc[i],pred_sources['RC_Dec'].iloc[i], frame='fk5', unit ='deg')
        rc_pix = rc.to_pixel(wcsradio)
        rc_y = hosts_overlay(0,rc_pix[1],im.shape[0]/2,im.shape[0]/2,-np.pi)
        rc_pix = np.array([rc_pix[0],rc_y[1]])
        #idx,d2d,_ = rc.match_to_catalog_sky(position)
        #print(idx,d2d)

        # Bounding box coordinates from ClaRAN were scaled up to a given scale, in this case 600

        ul = SkyCoord(pred_sources['x1'].iloc[i],pred_sources['y1'].iloc[i], frame='fk5', unit ='deg')

        lr = SkyCoord(pred_sources['x2'].iloc[i],pred_sources['y2'].iloc[i], frame='fk5', unit ='deg')


        upper_left = ul.to_pixel(wcsradio)
        lower_right = lr.to_pixel(wcsradio)


        ax.add_patch(
            plt.Rectangle((lower_right[0], lower_right[1]),
            upper_left[0] - lower_right[0],
            upper_left[1] - lower_right[1], fill=False,
            edgecolor='blue', linewidth=1.0)
            )

        ax.text(upper_left[0], upper_left[1] - 2,
                '{:s} {:.2f}'.format(cl, prob_score),
                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                fontsize=16, color='white')

        ind = source_inbbox(position_pix,upper_left[0],upper_left[1],lower_right[0],lower_right[1])
        #print(rc.match_to_catalog)
        host_name = hosts_ids[ind]
        sky_poss = position[ind]
        #print(idx,d2d.arcsec)
        host_ra = position_pix[0][ind]
        host_dec = position_pix[1][ind]
        flip_sky_poss = SkyCoord.from_pixel(host_ra,host_dec,wcsradio)
        #print(SkyCoord.from_pixel(host_ra,host_dec,wcsradio))

        poss_pybdsf_dis = np.zeros(len(host_ra))

        poss_rc_dis = np.zeros(len(host_ra))

        if (len(poss_rc_dis)!=0):
            found_host = True
                #for i in range(len(host_ra)):
            #poss_rc_dis = SkyCoord.from_pixel(rc_pix[0],rc_pix[1],wcsradio).separation(flip_sky_poss).arcsec#rc.separation(sky_poss).arcsec
            poss_rc_dis = rc.separation(flip_sky_poss).arcsec

            i_rc = np.where(poss_rc_dis==min(poss_rc_dis))[0]
                
            ir_host_ids_rc.append(host_name[i_rc])
                    #print(ir_host_ids_gc)
            poss_centre_rc.append([host_ra[i_rc],host_dec[i_rc]])
            rc_poss.append([flip_sky_poss.ra.deg[i_rc],flip_sky_poss.dec.deg[i_rc]])
                
                
        else:
            ir_host_ids_rc.append(["-"])
                    #print(ir_host_ids_gc)
            poss_centre_rc.append([float("nan"),float("nan")])
            rc_poss.append([float("nan"),float("nan")])

        pybdsf_host = source_inbbox(pybdsf_pix,upper_left[0],upper_left[1],lower_right[0],lower_right[1])

        if len(pybdsf_host)!=0 and (len(poss_rc_dis)!=0):
            found_host_center = True
            #print(pybdsf_host)  
            poss_source_dis = np.zeros(len(pybdsf_pix))

            if (len(poss_source_dis)!=0):
                #for j in range(len(pybdsf_poss_pix)):
                #poss_pybdsf_dis = SkyCoord.from_pixel(pybdsf_pix[0],pybdsf_pix[1],wcsradio).separation(flip_sky_poss).arcsec
                poss_pybdsf_dis = pybdsf.separation(flip_sky_poss).arcsec

#                             for k in range(len(pybdsf_poss_pix)):

                i_pybdsf = np.where(poss_pybdsf_dis==min(poss_pybdsf_dis))[0] 

                ir_host_ids_pybdsf.append(host_name[i_pybdsf])

                poss_centre_pybdsf.append([host_ra[i_pybdsf],host_dec[i_pybdsf]])
                pybdsf_poss.append([sky_poss.ra.deg[i_pybdsf],sky_poss.dec.deg[i_pybdsf]])
        else:
                
            ir_host_ids_pybdsf.append(["-"])
                    #print(ir_host_ids_gc)
            poss_centre_pybdsf.append([float("nan"),float("nan")])
            pybdsf_poss.append([float("nan"),float("nan")])

        rc = rc.to_pixel(wcsradio)
        pybdsf = pybdsf.to_pixel(wcsradio)

        rc_host = np.array(poss_centre_rc)
        pybdsf_host = np.array(poss_centre_pybdsf)
            #print(source_pix)


            #print(poss_host)
        source_ID = "Source_ID: " + str(source)
        ax.annotate(source_ID, xy=(0.45,0.97), xycoords='axes fraction', color='w', fontsize=14)

        coords = "RA, Dec: {:.4f} {:.4f} [degrees] (center)".format(source_ra,source_dec)
        ax.annotate(coords, xy=(0.45,0.95),xycoords='axes fraction', color='w', fontsize=14)


        scale = 1.0
        show_img_size = im.shape[0]

        scal = "scale: %.2f' x %.2f' (%.2fx%.2f pix)"%((xsize/scale)*60,(ysize/scale)*60,show_img_size,show_img_size)
        ax.annotate(scal, xy=(0.02,0.97), xycoords='axes fraction', color='w', fontsize=14)

        patch_contour = fuse(fits_f, im, None, sigma_level=4, mask_ir=False,
                                     get_path_patch_only=True)

        ax.scatter(rc[0],rc[1],s=260,c='blue',marker='+',label='RC')
        ax.scatter(pybdsf[0],pybdsf[1],s=260,c='lime',marker='+',label='PyBDSF')

        ax.set_ylim([im.shape[0],0])
        ax.add_patch(patch_contour)
        ax.imshow(im, origin='upper')
        ax.scatter(position_pix[0],position_pix[1],s=260,c='black',marker='x',label='IR host')
        if found_host:
                #rc_host_coords = deg2world(rc_host,wcsradio)
            ax.scatter(rc_host[:,0],rc_host[:,1],s=260,c='blue',marker='o',alpha=0.35,label='IR host (RC)')

        if found_host_center:
            ax.scatter(pybdsf_host[:,0],pybdsf_host[:,1],s=260,c='lime',marker='v',alpha=0.35,label='IR host (PyBDSF)')

    ax.legend()
    handles, labels = ax.get_legend_handles_labels()  
    lgd = dict(zip(labels, handles))
    ax.legend(lgd.values(), lgd.keys(),bbox_to_anchor=(1.04,0.5), loc="center left", fontsize=22, borderaxespad=0)
    plt.savefig(Output_overlay, bbox_inches='tight')
    plt.close()
    #plt.show()
    print("Saving to {}".format(Output_overlay))
    return ir_host_ids_pybdsf,pybdsf_poss,ir_host_ids_rc,rc_poss

if __name__=='__main__':
    
    df_unique = pd.read_csv("/idia/users/cmofokeng/data/gmrt/output_gmrt/ClaRAN_D4_edge_catalogue_v3.csv")
    data_cut = df_unique
    
    ids = df_unique["Source_ID"].values
    unique_ids = np.unique(ids)
    #for i in range(len(data_cut)):
    for i in unique_ids:
    
        #row = data_cut.iloc[i]

        data_df = data_cut[data_cut["Source_ID"]==i]

        img_file = osp.join("/idia/users/cmofokeng/data/gmrt/output_gmrt/","gmrt_en1w610_clipped_" + str(i) + "_infraredctmask_pred.png")#data_df["file_path"].iloc[0]
        fits_f = osp.join("/idia/users/cmofokeng/data/gmrt/split_fits/test_1deg",osp.basename(img_file).replace("_infraredctmask_pred.png",".fits"))
        with fits.open(fits_f) as file:
            hduradio = file[0]
            wcsradio = WCS(hduradio.header, naxis=2)

        ix = data_df.index.values
        #print(ix)

        props = overlay_centroids_unique(data_df)
        #print()
        for j in range(len(props[0])):
            rc = SkyCoord(data_df["RC_RA"].iloc[j],data_df["RC_Dec"].iloc[j],frame='fk5',unit='deg')
            pybdsf = SkyCoord(data_df["PyBDSF_RA"].iloc[j],data_df["PyBDSF_Dec"].iloc[j],frame='fk5',unit='deg')

            pybdsf_host_id = props[0][j][0]
            rc_host_id = props[2][j][0]

            ind = ix[j]
            if not np.isnan(props[3][j][0]):

                rc_host = SkyCoord(props[3][j][0],props[3][j][1],frame='fk5',unit='deg')
                #host = rc_host.to_string("decimal").split()
                #print(rc_host.ra.deg)

                rc_sep = rc_host.separation(rc).arcsec

                data_cut.loc[ind,"RC_host"] = rc_host_id
                data_cut.loc[ind,"RC_host_RA"] = np.around(rc_host.ra.deg,4)
                data_cut.loc[ind,"RC_host_Dec"] = np.around(rc_host.dec.deg,4)
                data_cut.loc[ind,"RC_separation"] = np.around(rc_sep,4)
            else:
                data_cut.loc[ind,"RC_host"] = "-"
                data_cut.loc[ind,"RC_host_RA"] = float("nan")
                data_cut.loc[ind,"RC_host_Dec"] = float("nan")
                data_cut.loc[ind,"RC_separation"] = float("nan")

            if not np.isnan(props[1][j][0]):
                pybdsf_host = SkyCoord(props[1][j][0],props[1][j][1],frame='fk5',unit='deg')
                pybdsf_sep = pybdsf_host.separation(rc_host).arcsec

                data_cut.loc[ind,"PyBDSF_host"] = pybdsf_host_id
                data_cut.loc[ind,"PyBDSF_host_RA"] = np.around(pybdsf_host.ra.deg,4)
                data_cut.loc[ind,"PyBDSF_host_Dec"] = np.around(pybdsf_host.dec.deg,4)
                data_cut.loc[ind,"PyBDSF_sep"] =np.around(pybdsf_sep,4)

            else:
                data_cut.loc[ind,"PyBDSF_host"] = "-"
                data_cut.loc[ind,"PyBDSF_host_RA"] = float("nan")
                data_cut.loc[ind,"PyBDSF_host_Dec"] = float("nan")
                data_cut.loc[ind,"PyBDSF_sep"] = float("nan")
                
    data_cut.to_csv("/idia/users/cmofokeng/data/gmrt/output_gmrt/claran_pybdsf_allwise_catalogue.csv",index=False)  