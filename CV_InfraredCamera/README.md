# Infrared Vision

## Overview

The thermal camera is used to detect activity in the episodes of darkness (e.g. during the night). It is assumed that the uncertainty of this source will be higher than RGB camera's in the good light. 

## Hardware

To run infrared vision, the following hardware is used. 
* [InfiRay P2Pro Thermal Camera](https://manuals.plus/infiray/p2-pro-thermal-camera-manual)

## Run

### Requirements

```shell
pip install -r requirements.txt
```

### Configuration

In [*config.json*](config.json) file declare the following variables:
* `webcam_no` - infrared camera's number (in Linux: `N` from `/dev/video{N}`)

### Camera check

If you want to verify if the camera works fine, run the following [script](view_from_IR.py). 

```shell
python view_from_IR.py
```
Getting the video from the IR camera means that the hardware is well connected and the configuration is correct