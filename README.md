# NTUT IEECS Master Thesis - ReID pipeline for Unity Simulation

This is my master thesis project. This repository contains the code for launching re-identification computer vision pipeline for the [simulated environment in Unity](https://github.com/Stephan109-5/ReID-Unity).

# Prerequisite

Make sure to first install pytorch with CUDA support using:
```
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

Then install the other required libraries from the `requirements.txt` file:
```
pip install -r requirements.txt
```
Also make sure to have Unity 6.3 LTS (6000.3.14f1) installed and be ready to run a scene (from the Assets/Scenes folder in Unity).

# FastApi Gallery Service

To start the Embedding Gallery Service:
```
uvicorn src.gallery.server:app --host <IP> --port <Port No.>
uvicorn src.gallery.server:app --host 0.0.0.0 --port 8000
```

# Running ReID Pipeline
For Re-ID, ensure that the gallery service is running first, then run the main.py with a .yaml config file: 
```
python main.py --config <yaml file>
python main.py --config eval_reid.yaml  
```
When the script finished loading all components and is waiting for Unity connection, hit 'Play' in the Unity Editor to start the simulation and stream the video feeds from the Unity scene to the ReID pipeline.

After the run is finished, tracking and ReID data are saved under the output directory specified in the yaml file.
```
<run_name>
├───reid_results
│   ├───reid_metrics_<Cam id>.json
│   └───reid_metrics_<Cam id>.json
└───tracking_results
    ├───tracking_results_<Cam id>.json
    └───tracking_results_<Cam id>.json

```

# Evaluation By TrackEval

After running the experiments, the evaluation of CLEAR MOT, IDF1 and HOTA metrics are carried out with trackeval.

Assuming parent folder contains subfolders where each subfolder is for a test run, use the evaluate_trackeval_batch python file to evaluate all the test run in the parent folder:
```
python .\evaluate_trackeval_batch.py .\eval2\
```

For single folder of a test run:
```
python .\evaluate_trackeval.py .\eval2\eval0_yolo11n_bytetrack_osnet_ain_x1_0_duke_20260706_212741
```

A `trackeval_summary.json` will be generated within the folder of the test run, containing all the metrics computed.

# Data Preprocessing

To transform all the data saved in the json files into a csv, use `frame_record_processing.ipynb` and run the four main cells to produce: 
+ A csv file with all runs listed
+ A csv file with the runs aggregated by Re-ID model and pre-trained weights used
+ Re-ID scatter plots
+ GT ID scatter plots

Make sure to modify to the folder_list array to include the folders you want to perform the data preprocessing.
Further data processing were done on the csv files in Microsoft Word/Excel to produce the tables that are in the thesis report. 

# Extra
The `eval1_track.yaml` was used to evaluate only tracking performance of various combinations of detectors and trackers for the first part of the experiments in the thesis.

The `standalone_camera.py` was made for receiving video feed from a single camera from the Unity scene 'WalkingTest'. Screenshots can be taken by typing the 's' key when the scene is running. An image and the ground-truth data saved in a json file are produced in a created folder named 'saved_frames'.

`standalone_camera.py` should be run with the `standalone.yaml` file:
```
python standalone_camera.py --config standalone.yaml
```

The `fps_analyze.ipynb` is for plotting the FPS plots of all the experiment runs in a given folder into a single image.

# Pretrained Weights
Most pretrained weights were downloaded from https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO and added to the `trained_weights` folder 

The weights `osnet_clonedperson_ms_d_c.pth` and `osnet_randperson_ms_d_c.pth` were fine-tuned from the `osnet_ms_d_c.pth.tar` weight using the code in `train_clonedperson.ipynb` and `train_randperson.ipynb` files respectively. Training instructions can be found in those files.