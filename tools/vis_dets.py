
import os.path as osp
import sys
sys.path.insert(0,'/users/cmofokeng/rgz_rcnn/tools/')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.wcs import WCS
from astropy.io import fits
from fuse_radio_ir import fuse, _get_contour
import make_contours
from matplotlib.path import Path
from astropy.coordinates import SkyCoord
import cv2




def visualize_unique(full_df,cut_df,img_type):
    
    suffix = ""
    output_suffix = ""
    if img_type=="D4":
        suffix = "_infrared.png"
        output_suffix = "_infraredctmask_pred.png"
    else:
        suffix = "_logminmax.png"
        output_suffix = "_logminmax_pred.png"
    
    data_full = full_df
    data_cut = cut_df
    
    for i in range(len(data_cut)):
        
        row = data_cut.iloc[i]
        
        data = data_cut[data_cut['Source_ID']==row['Source_ID']]
        data_df = data_full[data_full["Source_ID"]==row['Source_ID']]
        
        df_diff = pd.concat([data_df,data], sort=False).drop_duplicates(subset=['Class', 'x1', 'x2'],keep=False)
        
        file_path = "gmrt_en1w610_clipped_" + str(row["Source_ID"]) + suffix

        img = cv2.imread(osp.join("/idia/users/cmofokeng/data/gmrt/split_fits/test_1deg/",file_path))
        
        fits_file = osp.join("/idia/users/cmofokeng/data/gmrt/split_fits/test_1deg/",file_path.replace(suffix,".fits"))
        
        with fits.open(fits_file) as file:
            hduradio = file[0]
            wcsradio = WCS(hduradio.header, naxis=2)
            fits_data = hduradio.data

            xscale = abs(hduradio.header['CDELT1'])
            yscale = abs(hduradio.header['CDELT2'])

            xpix = hduradio.header['NAXIS1']
            ypix = hduradio.header['NAXIS2']

            xsize = xscale*xpix
            ysize = yscale*ypix
            
        patch_contour = fuse(fits_file, img, None, sigma_level=4, mask_ir=False,
                             get_path_patch_only=True)
        
        fig = plt.figure(figsize=(10,10),dpi=150)
            #fig.set_size_inches(600 / my_dpi, 600 / my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_xlim([0, img.shape[0]])
            #ax.set_ylim([0,im.shape[0]])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.set_aspect('equal')
            #im = cv2.resize(im, (600, 600))
        im = img[:, :, (2, 1, 0)]
        ax.imshow(im,origin='upper')
        ax.add_patch(patch_contour)

        #from matplotlib.patches import Rectangle
        for i in range(len(data)):

            row = data.iloc[i]

            ul = SkyCoord(row['x1'],row['y1'], frame='fk5', unit ='deg')
            #ul = SkyCoord(data['x1'],data['y1'], frame='icrs', unit ='deg')
            lr = SkyCoord(row['x2'],row['y2'], frame='fk5', unit ='deg')

            upper_left = ul.to_pixel(wcsradio)
            lower_right = lr.to_pixel(wcsradio)

            #print(sep1.arcsecond,sep2.arcsecond)
            w = upper_left[0] - lower_right[0]
            h = upper_left[1] - lower_right[1]

            ax.add_patch(plt.Rectangle((lower_right[0], lower_right[1]), w, h, edgecolor='blue', facecolor='none'))
            ax.text(upper_left[0], upper_left[1] - 2,
                                '{:s} {:.2f}'.format(row['Class'], row['Scores']),
                                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                                fontsize=16, color='white')

            source_ID = "Source_ID: " + str(row['Source_ID'])
            ax.annotate(source_ID, xy=(0.45,0.97), xycoords='axes fraction', color='w', fontsize=14)

            coords = "RA, Dec: {:.4f} {:.4f} [degrees] (center)".format(row['PyBDSF_RA'],row['PyBDSF_Dec'])
            ax.annotate(coords, xy=(0.45,0.95),xycoords='axes fraction', color='w', fontsize=14)


            scale = 1.0
            show_img_size = img.shape[0]

            scal = "scale: %.2f' x %.2f' (%.2fx%.2f pix)"%((xsize/scale)*60,(ysize/scale)*60,show_img_size,show_img_size)
            ax.annotate(scal, xy=(0.02,0.97), xycoords='axes fraction', color='w', fontsize=14)


        for i in range(len(df_diff)):

            row = df_diff.iloc[i]

            # store coordinates of the bounding box as an instance of skycoord
            ul = SkyCoord(row['x1'],row['y1'], frame='fk5', unit ='deg')
            #ul = SkyCoord(data['x1'],data['y1'], frame='icrs', unit ='deg')
            lr = SkyCoord(row['x2'],row['y2'], frame='fk5', unit ='deg')

            upper_left = ul.to_pixel(wcsradio)
            lower_right = lr.to_pixel(wcsradio)

            #print(sep1.arcsecond,sep2.arcsecond)
            w = upper_left[0] - lower_right[0]
            h = upper_left[1] - lower_right[1]
            ax.add_patch(plt.Rectangle((lower_right[0], lower_right[1]), w, h, edgecolor='black', facecolor='none', linestyle='--'))
            ax.text(upper_left[0], upper_left[1] - 2,
                                '{:s} {:.2f}'.format(row['Class'], row['Scores']),
                                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                                fontsize=16, color='white')

            source_ID = "Source_ID: " + str(row['Source_ID'])
            ax.annotate(source_ID, xy=(0.45,0.97), xycoords='axes fraction', color='w', fontsize=14)

            coords = "RA, Dec: {:.4f} {:.4f} [degrees] (center)".format(row['PyBDSF_RA'],row['PyBDSF_Dec'])
            ax.annotate(coords, xy=(0.45,0.95),xycoords='axes fraction', color='w', fontsize=14)


            scale = 1.0
            show_img_size = img.shape[0]

            scal = "scale: %.2f' x %.2f' (%.2fx%.2f pix)"%((xsize/scale)*60,(ysize/scale)*60,show_img_size,show_img_size)
            ax.annotate(scal, xy=(0.02,0.97), xycoords='axes fraction', color='w', fontsize=14)

        output_fname = osp.join("/idia/users/cmofokeng/data/gmrt/output_gmrt/",file_path.replace(suffix,output_suffix))
        print("Saving file to {}...\n".format(output_fname))
        fig.savefig(output_fname)
        plt.close(fig)
        
if __name__ == '__main__':
    
    # input catalog (full and filtered)
    data_full = pd.read_csv("/idia/users/cmofokeng/data/gmrt/output_gmrt/ClaRAN_D4_final_catalogue_v5.csv") # full catalog containing all predictions
    data_cut = pd.read_csv("/idia/users/cmofokeng/data/gmrt/output_gmrt/ClaRAN_D4_edge_catalogue_v3.csv") # catalog post removal of overlapping and edge detections
    
    visualize_unique(data_full,data_cut,"D4")
