# Photo Taker

This tool can be used for collecting datasets.

### Getting started

1. Make sure that the port numbers are correctly assigned to cameras in [config.json](../../layer_1/config.json)
2. Install the [requirements](requirements.txt) in the environment
```bash
pip install -r requirements.txt
```
3. Run the [main.py](main.py) script (recommended in ```tmux```)
```bash
python main.py
```

### Output

As an output you should get the set of pictures:
 - RGB1 camera photos
 - RGB1 camera photos with bounding boxes 
 - bounding boxes described in txt file
 - RGB1 camera photos
 - IR camera photos

### Aftermath

The [InfraRed_Photos_Annotated_for_Birds_Detection dataset](https://huggingface.co/datasets/Roszczyk/InfraRed_Photos_Annotated_for_Birds_Detection/tree/main) was used to train [Infrared_Bird_Detection](https://huggingface.co/Roszczyk/Infrared_Bird_Detection/tree/main) model. The dataset was annotated and the model was trained using [Intel Geti](https://docs.geti.intel.com/) software. The model was based on YOLO architecture. 