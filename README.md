#

Python3 version of [ClaRAN](https://github.com/chenwuperth/rgz_rcnn)

All information about requirements, setup, tutorial, etc. of ClaRAN code can be found on the link above.

## Transfer learning tutorial

After generating cutouts following approach outlined in ClaRAN's paper.

**Run**: `cd tools` and `%run /users/cmofokeng/rgz_rcnn/tools/demo_gmrt.py --radio /idia/users/cmofokeng/data/gmrt/split_fits/test_1deg/gmrt_en1w610_clipped_97.fits --ir /idia/users/cmofokeng/data/gmrt/split_fits/test_1deg/gmrt_en1w610_clipped_97_infrared.png --catalog /idia/users/cmofokeng/data/gmrt/en1w610-5sg9-clean-offset.vot`

catalog argument was used in order to overlay positional information about the centre. The script above returns a csv file of detections from ClaRAN.

## Source characterization pipeline
Switch to `sc_pipeline` branch.

From the detection and classification phase, ClaRAN ouputs the csv files of the detections which were grouped into a catalog.

The bounding boxes of the detected sources are used to find the positions of the radio source, furthermore the estimated source positions are then used to cross-identify IR sources.

**Run**: `python3 source_positions.py` to characterize sources.

PS: the code currently uses static files, the next version will improve upon this. 
