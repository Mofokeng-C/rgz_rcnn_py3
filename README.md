# Source characterization pipeline

From the detection and classification phase, ClaRAN ouputs the csv files of the detections which were grouped into a catalog.

The bounding boxes of the detected sources are used to find the positions of the radio source, furthermore the estimated source positions are then used to cross-identify IR sources.

The repository include:

* Jupyter notebooks used for preprocessing and post-processing of the catalogued data, as well as visualization of output images from CLARAN and the implemented pipeline.
* Tools used to implement the source characterization pipeline.

## Getting started

* [analysis\_&\_results.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/analysis_%26_results.ipynb)
* [claran-gmrt\_example.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/claran-gmrt_example.ipynb)
* [com\_positions.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/com_positions.ipynb)
* [filtered_examples.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/filtered_examples.ipynb)
* [large\_cutout\_generation.ipynb](https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/large_cutout_generation.ipynb)
* [rc\_positions\_examples.ipynb(https://github.com/Mofokeng-C/rgz_rcnn_py3/blob/sc_pipeline/notebooks/rc_positions_examples.ipynb)

**Run**: `python3 source_positions.py` to characterize sources.

PS: the code currently uses static files, the next version will improve upon this. 
