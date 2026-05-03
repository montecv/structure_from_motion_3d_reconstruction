# Camera Calibration Tool

This script performs camera calibration using a chessboard pattern and outputs parameters compatible with a **SIMPLE_RADIAL** camera model (e.g. used in COLMAP-style pipelines or custom SfM implementations).

It also supports **rescaling intrinsics** to match a different resolution (e.g. when calibrating on photos but using video frames later).

---

## Input Data

You need a set of chessboard images:

* Taken from different angles and positions
* Good lighting and sharp focus
* Chessboard fully visible in most images

### For example, you can use chessboard from this [link](https://markhedleyjones.com/media/projects/calibration-checkerboard-collection/Checkerboard-A4-20mm-13x9.pdf) 

---

## Chessboard Parameters

You must provide:

* `cols` – number of **inner corners horizontally**
* `rows` – number of **inner corners vertically**
* `square_size` – size of one square (in meters)

---

## Usage

### Basic calibration

```bash

python calibrate.py \
  --images_path chess_images \
  --cols 13 \
  --rows 9 \
  --square_size 0.0275
```

---

### Calibration + rescaling for video

```bash

python calibrate.py \
  --images_path chess_images \
  --cols 13 \
  --rows 9 \
  --square_size 0.0275 \
  --video_width 3840 \
  --video_height 2160
```
