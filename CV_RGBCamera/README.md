# Daylight Colour Vision

## Overview

The RGB camera will be used to detect birds and other animals in good light (mostly daylight). 

## Hardware

To run RGB vision, the following hardware is used. 
* ELP Svpro 4K Camera

But the program should work with any kind of RGB camera.

## Run

### Requirements

```shell
pip install -r requirements.txt
```

### Configuration

In [*config.json*](config.json) file declare the following variables:
* `webcam_no` - RGB camera's number (in Linux: `N` from `/dev/video{N}`)

### Camera check

If you want to verify if the camera works fine, run the following [script](view_from_RGB.py). 

```shell
python view_from_RGB.py
```
Getting the video from the RGB camera means that the hardware is well connected and the configuration is correct