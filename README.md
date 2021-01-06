# Source characterization pipeline

From the detection and classification phase, ClaRAN ouputs the csv files of the detections which were grouped into a catalog.

The bounding boxes of the detected sources are used to find the positions of the radio source, furthermore the estimated source positions are then used to cross-identify IR sources.

The repository include:

* Jupyter notebooks used for preprocessing and post-processing of the catalogued data, as well as visualization of output images from CLARAN and the implemented pipeline.
* Tools used to implement the source characterization pipeline.

## Getting started

* [analysis\_&\_results.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/analysis_%26_results.ipynb): This notebook shows the analysis and result of running CLARAN on radio image data from the GMRT telescope.
* [claran-gmrt\_example.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/claran-gmrt_example.ipynb): This notebook is a tutorial showing steps taken to apply CLARAN on GMRT data and the adapted versions of code used.
* [com\_positions.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/com_positions.ipynb): This notebook shows a tutorial on the computation of the central positions of the detected radio sources.
* [filtered_examples.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/filtered_examples.ipynb): This notebook shows different techniques implemented to get rid of overlapping and edge detections, in order to obtain "unique" detections.
* [large\_cutout\_generation.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/large_cutout_generation.ipynb): This notebook is a tutorial on how to generate a 15 arcmin cutout and also how to apply CLARAN on it.
* [rc\_positions\_examples.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/rc_positions_examples.ipynb): This notebook shows the code and the output examples of the source characterization pipeline implemented.
* ([demo\_gmrt.py](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/tools/demo_gmrt.py), [source\_positions.py](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/tools/source_positions.py), [vis_dets](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/tools/vis_dets.py)): Tools adapted from CLARAN and others used to implement the source characterization pipeline.


`To run the source characterization pipeline, follow claran-gmrt\_example.ipynb tutorial.`

`# to characterize sources` <br>
`python3 source_positions.py` 

PS: the code currently uses static files, the next version will improve upon this. 
