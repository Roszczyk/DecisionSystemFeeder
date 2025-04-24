# Infrared Vision

## Overview

The thermal camera is used to detect activity in the episodes of darkness (e.g. during the night). It is assumed that the uncertainty of this source will be higher than RGB camera's in the good light. 

## Hardware

To run infrared vision, the following hardware is used. 
* [InfiRay P2Pro Thermal Camera](https://manuals.plus/infiray/p2-pro-thermal-camera-manual)

## Run

```shell
pip install -r requirements.txt
```

Before running, check which ID the thermal camera has in your case and adjust the code.

```python
cap = cv2.VideoCapture(ID)
```

**TO DO**: adjusting in config file.