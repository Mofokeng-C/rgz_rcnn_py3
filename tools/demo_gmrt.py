#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia, 2017
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the MIT license
#
#    Created on 15 March 2018 by chen.wu@icrar.org

import os, sys, time
import os.path as osp
import argparse
import json
import csv
import pandas as pd
from itertools import cycle
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table, Column
from astropy.io.votable import parse_single_table
from astropy.io import ascii

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from astropy.io import fits
import tensorflow as tf

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect, remove_embedded
from fast_rcnn.nms_wrapper import nms
from networks.factory import get_network
from utils.timer import Timer
from download_data import get_rgz_root
from fuse_radio_ir import fuse

CLASSES =  ('__background__', # always index 0
                            '1_1', '1_2', '1_3', '2_2', '2_3', '3_3')

colors_ = cycle(['blue', 'blue', 'blue'])

#def write_detections(dict_writer,class_name,score,box):
#    dict_writer.writerow({'class': class_name,'scores': score, 'boxes': box})
#    return 0
    


def vis_detections(im, class_name, dets,ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    #print(inds)
    if len(inds) == 0:
        # get a box with a highest score
        # try:
        #     max_score = np.max(dets[:, -1])
        #     inds = np.where(dets[:, -1] == max_score)[0][0:1]
        # except Exception as exp:
        #     print('inds == 0, but %s' % str(exp))
        return len(inds)

    
#     with open(output_f, "w") as f:
#         f.write("class,scores,x1,y1,x2,y2\n")
    for i in inds:
        #cl = class_name[i]
        
        bbox = dets[i, :4]

        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1], fill=False,
            edgecolor=next(colors_), linewidth=1.0)
            )
                #cns = class_name.split('_')
                #class_name = '%sC%sP' % (cns[0], cns[1])
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.2f}'.format(class_name.replace('_', 'C_') + 'P', score),
                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                fontsize=12, color='white')
                    #bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                    #fontsize=25, color='black')

            #classes = np.array(['1C_1P','1C_2P','1C_3P','2C_2P','2C_3P','3C_3P'])
#         pred_class = '{:s}'.format(class_name.replace('_', 'C_') + 'P')
            
            #string = "{}\t{0:.4f}\t{0:.4f}\t{0:.4f}\t{0:.4f}\t{0:.4f}\n".format(pred_class,score,bbox[0],bbox[1],bbox[2],bbox[3])

            #f.write(string)
#             f.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f\n"%(pred_class,score,bbox[0],bbox[1],bbox[2],bbox[3]))
        

    plt.axis('off')
    #plt.tight_layout()
    plt.draw()
    return len(inds)

def vis_detections_new(outfile,im, class_name, dets, ax, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -2] >= thresh)[0]
    if len(inds) == 0:
        thresh = np.max(dets[:, -2])
        inds = np.where(dets[:, -2] >= thresh)[0]
        
    with open(outfile, 'w') as csvfile:
        fieldnames = ['scores', 'boxes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            
            writer.writerow({'scores': score, 'boxes': bbox})

            ax.add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor=next(colors_), linewidth=1.0)
            )
            #cns = class_name.split('_')
            #class_name = '%sC%sP' % (cns[0], cns[1])
            ax.text(bbox[0], bbox[1] - 2,
                    '{:s} {:.2f}'.format(class_name.replace('_', 'C_') + 'P', score),
                    bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                    fontsize=10, color='white')
                    #bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                    #fontsize=25, color='black')
            ax.annotate('1 arcmin', xy=(570, 20), xycoords='data', ha='center',
                        xytext=(0, -8), textcoords='offset points')
        
            ax.annotate('', xy=(540,30), xytext=(600,30),
                        xycoords='data', textcoords='data',
                        arrowprops={'arrowstyle': '|-|,widthA=0.2,widthB=0.2', })
            
            
        print("\nSaving data to %s..."%outfile)


    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    plt.axis('off')
    #plt.tight_layout()
    plt.draw()
    return len(inds)

def demo(sess, net, im_file, vis_file, fits_fn,radio_catalog, conf_thresh=0.8, eval_class=True):
    """
    Detect object classes in an image using pre-computed object proposals.
    im_file:    The "fused" image file path
    vis_file:   The background image file on which detections are laid.
                Normallly, this is just the IR image file path
    fits_fn:    The FITS file path
    ir_hosts:   The position (RA,Dec) of IR hosts in the given FOV
    eval_class: True - use traditional per class-based evaluation style
                False - use per RoI-based evaluation

    """
    show_img_size = cfg.TEST.SCALES[0]
    if (not os.path.exists(im_file)):
        print(('%s cannot be found' % (im_file)))
        return -1
    im = cv2.imread(im_file)
    
    scale = float(show_img_size) / float(im.shape[0])
    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    image_name = osp.basename(im_file)
    scores, boxes = im_detect(sess, net, im, save_vis_dir=None,
                             img_name=os.path.splitext(image_name)[0])
                    
    boxes *= scale
    
    timer.toc()
    sys.stdout.write('Done in {:.3f} secs'.format(timer.total_time))
    sys.stdout.flush()
    
    fits_file = osp.basename(fits_fn)
    
    output_fn = ""
                    
                    
    #output_nm.replace(".fits",".csv") # D4 input image
    #output_fn = output_nm.replace(".fits","_D1.csv") # D1 input image
    s = fits_file.replace(".fits","")
    sid = s.replace("gmrt_en1w610_clipped_","")
                    
    table = parse_single_table(radio_catalog)
    data = table.to_table(use_names_over_ids=True)
    #data = data.filled()
    
    #data = data.to_pandas()
    
    #i = np.where(table.array["Source_id"].data==sid)[0]
    source = ""
    RA = 0.0
    Dec = 0.0
    
    
    for i in range(len(data['RA_deg'])):
        if (str(data['Source_id'][i])==sid):
            
            source = str(data['Source_id'][i])
            RA = data['RA_deg'][i]
            Dec = data['Dec_deg'][i]


    im = cv2.imread(vis_file)
    if (vis_file.endswith('logminmax.png')): # D1
        output_fn = osp.basename(fits_fn).replace(".fits","_D1.csv")
    else:
        output_fn = fits_file.replace(".fits",".csv") # else D4

    my_dpi = 100
    fig = plt.figure()
    fig.set_size_inches(show_img_size / my_dpi, show_img_size / my_dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_xlim([0, show_img_size])
    ax.set_ylim([show_img_size, 0])
    #ax.set_aspect('equal')
    im = cv2.resize(im, (show_img_size, show_img_size))
    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    if (fits_fn is not None):
                    
        #data = np.loadtxt(ir_hosts,delimiter=',',skiprows=1)
        #position = SkyCoord(data[:,0],data[:,1], frame='icrs', unit ='deg')
        
        with fits.open(fits_fn) as file:
            hduradio = file[0]
            wcsradio = WCS(hduradio.header, naxis=2)
            #RA = hduradio.header['CRVAL1']
            #Dec = hduradio.header['CRVAL2']
            
            xscale = abs(hduradio.header['CDELT1'])
            yscale = abs(hduradio.header['CDELT2'])
            
            xpix = hduradio.header['NAXIS1']
            ypix = hduradio.header['NAXIS2']
            
            xsize = xscale*xpix
            ysize = yscale*ypix
    
        
        
        #x = position.to_pixel(wcsradio)
        #print(x[0][0],x[1][0])
        #print(type(scale))

        #x[0] = x[0] * scale
        #x[1] = x[0] * scale
        #RAs = '%.2f' % (RA)
        #Decs = '%.2f' % (Dec)
        
        #angular_scale = xsize
        
        #coords = "RA, Dec: {:.4f} {:.4f} [degrees] (center)".format(RA,Dec)
        #ax.annotate(coords, xy=(0.45,0.97),xycoords='axes fraction', color='w', fontsize=10)

        # scale, as text, not scale bar; this is in the same section as "coords"
        #scal = 'scale: '+str((xsize/scale)*60)+"' x "+str((ysize/scale)*60)+"' ("+str(scale)+'x'+str(scale)+'pix)'
        #scal = 'scale: %.2f'+"' x "+'%.2f'+"' (%.2f x %.2f"+'pix)'%((xsize/scale)*60,(ysize/scale)*60,scale,scale)
        #scal = "scale: %.2f' x %.2f' (%.0f x %.0f pix)"%((xsize)*60,(ysize)*60,show_img_size,show_img_size)
        #ax.annotate(scal, xy=(0.02,0.97), xycoords='axes fraction', color='w', fontsize=10)
        
        
        #source = "Source_ID: " + source
        #ax.annotate(source, xy=(0.02,0.02),xycoords='axes fraction', color='w', fontsize=10)
        #ax.scatter(x[0]*scale,x[1]*scale,s=70,c='black',marker='x')
        patch_contour = fuse(fits_fn, im, None, sigma_level=4, mask_ir=False,
                             get_path_patch_only=True)
        ax.add_patch(patch_contour)
        
    NMS_THRESH = cfg.TEST.NMS #cfg.TEST.RPN_NMS_THRESH # 0.3

    tt_vis = 0
    bbox_img = []
    bscore_img = []
    num_sources = 0
                           
    #if (eval_class):
    #with open(output_fn, 'w') as csvfile:
    #    fieldnames = ['class','scores', 'boxes']
    #    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #    writer.writeheader()
    with open(output_fn, "w") as f:
        f.write("class,scores,x1,y1,x2,y2\n")
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            cls_boxes = boxes[:, 4 * cls_ind : 4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis]))#.astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            """Draw detected bounding boxes."""
            inds = np.where(dets[:, -1] >= conf_thresh)[0]
            #print(inds)
            if len(inds) == 0:
                # get a box with a highest score
                # try:
                #     max_score = np.max(dets[:, -1])
                #     inds = np.where(dets[:, -1] == max_score)[0][0:1]
                # except Exception as exp:
                #     print('inds == 0, but %s' % str(exp))
                continue
                
            else:
                for i in inds:
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    pred_class = '{:s}'.format(cls.replace('_', 'C_') + 'P')

                    f.write("%s,%.4f,%.4f,%.4f,%.4f,%.4f\n"%(pred_class,score,bbox[0],bbox[1],bbox[2],bbox[3]))
                
                
#             for i in inds:

    #                 cl = cls[i]
    #                 pred_class = '{:s}'.format(cl.replace('_', 'C_') + 'P')




#             print(dets)

            num_sources += vis_detections(im, cls, dets, ax, thresh=conf_thresh)

            #write_detections(im, cls, dets, thresh=conf_thresh)
            #print(score)
            #writer.writerow({'class': cl,'scores': score, 'boxes': box})
        
        
        #dets = np.hstack((dets, np.ones([dets.shape[0], 1]) * cls_ind))
        # if (dets.shape[0] > 0):
        #     bbox_img.append(dets)
        #     bscore_img.append(np.reshape(dets[:, -2], [-1, 1]))
    # else:
    #     for eoi_ind, eoi in enumerate(boxes):
    #         eoi_scores = scores[eoi_ind, 1:] # skip background
    #         cls_ind = np.argmax(eoi_scores) + 1 # add the background index back
    #         cls_boxes = boxes[eoi_ind, 4 * cls_ind : 4 * (cls_ind + 1)]
    #         cls_scores = scores[eoi_ind, cls_ind]
    #         dets = np.hstack((np.reshape(cls_boxes, [1, -1]),
    #                           np.reshape(cls_scores, [-1, 1])))#.astype(np.float32)
    #         dets = np.hstack((dets, np.ones([dets.shape[0], 1]) * cls_ind))
    #         bbox_img.append(dets)
    #         bscore_img.append(np.reshape(dets[:, -2], [-1, 1]))
    #
    # boxes_im = np.vstack(bbox_img)
    # scores_im = np.vstack(bscore_img)
    #
    # #if (not eval_class):
    # # a numpy float is a C double, so need to use float32
    # keep = nms(boxes_im[:, :-1].astype(np.float32), NMS_THRESH)
    # boxes_im = boxes_im[keep, :]
    # scores_im = scores_im[keep, :]
    #
    # keep_indices = range(boxes_im.shape[0])
    #num_sources = vis_detections(im, None, boxes_im[keep_indices, :], ax, thresh=conf_thresh)

    print((', found %d sources' % num_sources))
    return 0

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Run a demo detector')
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use for GPUs',
                        default=0, type=int)
    parser.add_argument('--radio', dest='radio_fits',
                        help='full path of the radio fits file (compulsory)',
                        default=None, type=str)
    parser.add_argument('--ir', dest='ir_png',
                        help='full path of the infrared png file (compulsory)',
                        default=None, type=str)
    parser.add_argument('--output', dest='fig_path', help='Output path for detections',
                        default='.')
    parser.add_argument('--p-value', dest='conf_thresh', help='P-value',
                        default=0.8, type=float)
    parser.add_argument('--evaleoi', dest='eval_eoi',
                        help='evaluation based on EoI',
                        action='store_true', default=False)
    parser.add_argument('--model', dest='model', help='which pre-trained model to load',
                        default='D4')
    parser.add_argument('--radio-png', dest='radio_png',
                        help='full path of the radio png file (only for D1 method)',
                        default=None, type=str)
    parser.add_argument('--catalog', dest='radio_catalog',
                        help='full path of the infrared hosts text file (compulsory)',
                        default=None, type=str)

    args = parser.parse_args()
    if ('D1' != args.model):
        if (args.radio_fits is None or args.ir_png is None):
            parser.print_help()
            sys.exit(1)

        if (not osp.exists(args.radio_fits)):
            print(('Radio fits %s not found' % args.radio_fits))
            sys.exit(1)

        if (not osp.exists(args.ir_png)):
            print(('Infrared png %s not found' % args.ir_png))
            sys.exit(1)
            

        if (not args.model in ['D4', 'D5', 'D1', 'D3']):
            print(('Unknown model: %s' % args.model))
            sys.exit(1)
        
        if (not osp.exists(args.radio_catalog)):
            print(('catalog %s not found' % args.radio_catalog))
            sys.exit(1)


    else:
        if (args.radio_png is None):
            print('D1 method must specify radio png path with option "--radio-png"')
            sys.exit(1)
        elif (not osp.exists(args.radio_png)):
            print(('Radio png not found: %s' % args.radio_png))
            sys.exit(1)

    if args.device.lower() == 'gpu':
        cfg.USE_GPU_NMS = True
        cfg.GPU_ID = args.device_id
        device_name = '/{}:{:d}'.format(args.device, args.device_id)
        print(device_name)
    else:
        cfg.USE_GPU_NMS = False

    return args

def hard_code_cfg():
    """
    This could be parameters potentially. Hardcoded for now
    """
    cfg.TEST.HAS_RPN = True
    cfg.TEST.RPN_MIN_SIZE = 4
    cfg.TEST.RPN_POST_NMS_TOP_N = 5
    cfg.TEST.NMS = 0.2
    cfg.TEST.RPN_NMS_THRESH = 0.5
    cfg.TEST.SCALES = (600,)
    cfg.TEST.MAX_SIZE =2000

def fuse_radio_ir_4_pred(radio_fn, ir_fn, out_dir='/tmp', model='D4'):
    """
    return the full path to the generated fused file
    """
    if (model != 'D5'):
        nsz = None
        if ('D3' == model):
            mask_ir = False
        else:
            mask_ir = True
    else:
        nsz = cfg.TEST.SCALES[0] #i.e. 600
        mask_ir = True
    return fuse(radio_fn, ir_fn, out_dir, new_size=nsz, mask_ir=mask_ir)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #hide tensorflow warnings
    args = parse_args()
    if ('D1' == args.model):
        im_file = args.radio_png
        vis_file = args.radio_png
    else:
        im_file = fuse_radio_ir_4_pred(args.radio_fits, args.ir_png, model=args.model)
        vis_file = args.ir_png
    #print("im_file", im_file)
    if (im_file is None):
        print("Error in generating contours")
        sys.exit(1)
    hard_code_cfg()
    net = get_network('rgz_test')
    iter_num = 80000
    if ('D3' == args.model):
        iter_num = 60000
    model_weight = osp.join(get_rgz_root(), 'data/pretrained_model/rgz/%s/VGGnet_fast_rcnn-%d' % (args.model, iter_num))
    if (not osp.exists(model_weight + '.index')):
        print("Fail to load rgz model, have you done \'python download_data.py\' already?")
        sys.exit(1)

    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sys.stdout.write('Loading RGZ model from {:s}... '.format(model_weight))
    sys.stdout.flush()
    stt = time.time()
    saver.restore(sess, model_weight)
    print(("Done in %.3f seconds" % (time.time() - stt)))
    sys.stdout.write("Detecting radio sources... ")
    sys.stdout.flush()
    ret = demo(sess, net, im_file, vis_file, args.radio_fits, args.radio_catalog, conf_thresh=args.conf_thresh,
               eval_class=(not args.eval_eoi))
    if (-1 == ret):
        print(('Fail to detect in %s' % args.radio_fits))
    else:
        im_name = osp.basename(im_file)
        output_fn = osp.join(args.fig_path, im_name.replace('.png', '_pred.png'))
        plt.savefig(output_fn, dpi=150)
        plt.close()
        print(('Detection saved to %s' % output_fn))
        if ('D1' != args.model):
            for trf in [im_file]:
                if (osp.exists(trf)):
                    try:
                        os.remove(trf)
                    except:
                        pass

