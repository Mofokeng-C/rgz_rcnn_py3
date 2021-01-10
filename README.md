#

Python3 version of [ClaRAN](https://github.com/chenwuperth/rgz_rcnn)

All information about requirements, setup, tutorial, etc. of ClaRAN code can be found on the link above.

## Transfer learning tutorial

After generating cutouts following approach outlined in ClaRAN's paper.

**Run**: `cd tools` and `%run /users/cmofokeng/rgz_rcnn/tools/demo_gmrt.py --radio /idia/users/cmofokeng/data/gmrt/split_fits/test_1deg/gmrt_en1w610_clipped_97.fits --ir /idia/users/cmofokeng/data/gmrt/split_fits/test_1deg/gmrt_en1w610_clipped_97_infrared.png --catalog /idia/users/cmofokeng/data/gmrt/en1w610-5sg9-clean-offset.vot`

<img width="400" alt="gmrt_en1w610_clipped_21_logminmax_pred" src="https://user-images.githubusercontent.com/42966715/104115088-10b9de80-5314-11eb-83d3-afbb91840484.png" hspace="10"/><img width="400" alt="gmrt_en1w610_clipped_21_infraredctnmask_pred" src=https://user-images.githubusercontent.com/42966715/104115024-75c10480-5313-11eb-9a11-b0bce67a4437.png/>

catalog argument was used in order to overlay positional information about the centre of the cutout. The script above returns a csv file of detections from ClaRAN.

## Source characterization pipeline
Switch to [sc_pipeline](https://github.com/Mofokeng-C/rgz_rcnn_py3/tree/sc_pipeline) branch for the pipeline code.

From the detection and classification phase, ClaRAN ouputs the csv files of the detections which were grouped into a catalog.

The bounding boxes of the detected sources are used to find the positions of the radio source, furthermore the estimated source positions are then used to cross-identify IR sources.

**Run**: `python3 source_positions.py` to characterize sources.

PS: the code currently uses static files, the next version will improve upon this. 
