# Audio Analysis

## Overview

The microphone is used to acquire sound from the feeder to analysis. 

## Hardware

To run audio acquisition, the following hardware is used. 
* [MEDIA-TECH Virtu 5.1 Sound Card](https://media-tech.eu/en/sound-card-virtu-5-1-usb-mt5101/?srsltid=AfmBOoq-Ml-67Cj5jOKRbtYw7dhxex3nJoqON-HslteiRDkbta-Sq12w)
* [Microphone BOYA BY-M1](https://prostage.no/media/multicase/documents/pdf%20boya/by-m1manual.pdf)

## Run

### Requirements

Apart from required Python libraries, some Linux drivers need to be installed. There is a script [linux_install](./linux_install.sh) prepared for this purpose.

```shell
sudo sh linux_install.sh
pip install -r requirements.txt
```

### List available devices

To verify if everything works fine and the system indetifies the sound card, run the [following script](./list_devices.py).

```shell
python list_devices.py
```
You should get the list of available audio devices. If you don't, there's additional troubleshooting necessary. 

### Record draft recording

Use the [following script](./record_draft.py) to record a 5-second demo of sound acquired by the microphone.

```shell
python record_draft.py
```

The file will be saved in ```.wav``` format. 