from ultralytics import YOLO
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path
from PIL import Image
import os


def run():

    path_to_dataset = Path(__file__).parent / "../../../../datasets"
    path_yolo_dataset = Path(__file__).parent / "yolo_dataset"
    images_dir = path_yolo_dataset / "images"
    labels_dir = path_yolo_dataset / "labels"
    
    def copy_and_annotate(data, subset):
        for img_path, cls_id in data:
            fname = os.path.basename(img_path)
            dst_img = images_dir / subset / fname
            dst_label = labels_dir / subset / (os.path.splitext(fname)[0] + '.txt')

            shutil.copy(img_path, dst_img)

            x_center, y_center, w, h = 0.5, 0.5, 1.0, 1.0

            with open(dst_label, 'w') as f:
                f.write(f"{cls_id} {x_center} {y_center} {w} {h}\n")

    need_to_copy = not path_yolo_dataset.exists()

    if need_to_copy:
        os.makedirs(images_dir / "train")
        os.makedirs(images_dir / "val")
        os.makedirs(labels_dir / "train")
        os.makedirs(labels_dir / "val")

    class_names = sorted(os.listdir(path_to_dataset))
    class_map = {cls: i for i, cls in enumerate(class_names)}

    image_label_pairs = []
    for cls in class_names:
        folder = path_to_dataset / cls
        for fname in os.listdir(folder):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_label_pairs.append((folder / fname, class_map[cls]))

    if need_to_copy:
        train_data, val_data = train_test_split(image_label_pairs, test_size=0.2, stratify=[lbl for _, lbl in image_label_pairs])

        copy_and_annotate(train_data, 'train')
        copy_and_annotate(val_data, 'val')

    data_yaml = f"""
    path: {path_yolo_dataset}
    train: images/train
    val: images/val
    names: {class_names}
    """

    with open(path_yolo_dataset / 'data.yaml', 'w') as f:
        f.write(data_yaml)

    model = YOLO('yolov8n.pt')

    model.train(
        data = path_yolo_dataset / 'data.yaml',
        epochs = 50,
        imgsz = 640,
        batch = 16,
        device = 0,
        name = 'object-detector'
    )

if __name__ == "__main__":
    run()