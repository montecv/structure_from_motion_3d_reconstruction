# 3D Reconstruction from 2D Images

## Introduction

This project uses the OpenCV library to perform 3D reconstruction from 2D images.

Consider a simple scenario where we have a set of images of an object taken from different viewpoints using the same camera. The camera is calibrated, and its intrinsic parameters are known.

### Algorithm 


Initialization → New frame selection → New frame registration → New points triangulation → Global optimization if needed → New frame selection → ...

## Usage

Place your images in the ```images``` folder in the root directory. (Example images are from the South Building dataset available at the following link: [link](https://demuc.de/colmap/datasets/))

The main script is `sfm.py`. To see its available options, run `sfm.py --help`

##### Recommended usage:

```
python3 sfm.py --input ./images --num_features=10000 --resize_factor=0.5 --reprojection_threshold=1e-3 --ba_frequency=10 --vis_frequency=10
```

## Docker Usage

This project can also be run inside a Docker container to avoid dependency issues.

### 1. Build the Docker image

```
docker build -t sfm-project .
```

### 2. Run the container (interactive mode)


```docker run -it -v $(pwd):/app sfm-project bash```

This starts a shell inside the container. You can then manually run the pipeline:
```
python3 sfm.py --input ./images --num_features=10000 --resize_factor=0.5 --reprojection_threshold=1e-3 --ba_frequency=10 --vis_frequency=10
```

```
python3 sfm.py --video IMG_9434.MOV --frame_stride 25 --camera_model "SIMPLE_RADIAL 2160 3840 2530.384465475964 1087.474877030783 1910.051336865839 0.2296724800755303" --num_features=10000 --resize_factor=1.0 --reprojection_threshold=1e-3 --ba_frequency=10 --vis_frequency=10 --sequential 3
```



Any generated files (e.g. point clouds, images, logs) will be saved directly to your local filesystem.


### Additional info
If you don't know the parameters of your camera, check the README in the camera_calibration folder.