# Simple model

This model's job is to detect objects from the following classes:
* bird
* squirrel
* cat
* human

### Dataset

It has been trained on the dataset created by merging various datasets from [images.cv](https://images.cv/search-labeled-image-dataset).

* bird (2124 items)
    * [Bearded barbet](https://images.cv/dataset/bearded-barbet-image-classification-dataset)
    * [Black throated warbler](https://images.cv/dataset/black-throated-warbler-image-classification-dataset)
    * [Chara de collar](https://images.cv/dataset/chara-de-collar-image-classification-dataset)
    * [Brown trasher](https://images.cv/dataset/brown-thrasher-image-classification-dataset)
    * [Crest nunhatch](https://images.cv/dataset/crested-nuthatch-image-classification-dataset)
    * [Antbird](https://images.cv/dataset/antbird-image-classification-dataset)
    * [Cuban trogon](https://images.cv/dataset/cuban-trogon-image-classification-dataset)
    * [Steamer duck](https://images.cv/dataset/steamer-duck-image-classification-dataset)
    * [Sparrow](https://images.cv/dataset/sparrow-image-classification-dataset)
    * [Pigeon](https://images.cv/dataset/pigeon-image-classification-dataset)
* squirrel (1121 items)
    * [Squirrel](https://images.cv/dataset/squirrel-image-classification-dataset)
* cat (6384 items)
    * [Cat](https://images.cv/dataset/cat-image-classification-dataset)
* human (1351 items)
    * [Person](https://images.cv/dataset/person-image-classification-dataset)
    * [Faces](https://images.cv/dataset/faces-image-classification-dataset)

```Issue #1```: These datasets' images are cropped to contain only the detection, so YOLO bounding box has ```0.5 0.5 1 1``` coordinates for every item. This causes the problem that the model becomes a classifier, not a detector - even if the model predicts the class quite well, it gives ```0.5 0.5 1 1``` bounding box coordinates for every input.

```Solution #1```: [The script](prepare_dataset_with_background.py) was developed that puts every image randomly in the white background creating new dataset with various label coordinates.

```Issue #2```: The results after Solution #1 are better - model works as a detector, not as a classifier. Problem is that the sizes of bounding box are always the same. 

```Solution #2```: Random changes of images in the background (bounding boxes) added to the [the script](prepare_dataset_with_background.py).

```Python
scale = random.uniform(0.5, 2.0)
new_w = int(orig_w * scale)
new_h = int(orig_h * scale)
resized_img = img.resize((new_w, new_h), resample=Image.LANCZOS)
```