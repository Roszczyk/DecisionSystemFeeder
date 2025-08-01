import os
import random
from tqdm import tqdm
from pathlib import Path
from PIL import Image

input_dir = Path(__file__).parent /  "../../../datasets"
output_dir = Path(__file__).parent / "yolo_dataset"
canvas_size = (640, 640)
num_augmented_per_image = 4 
split_ratio = 0.8

for split in ["train", "val"]:
    os.makedirs(f"{output_dir}/images/{split}", exist_ok=True)
    os.makedirs(f"{output_dir}/labels/{split}", exist_ok=True)

class_names = sorted([d.name for d in Path(input_dir).iterdir() if d.is_dir()])
class_to_idx = {cls: i for i, cls in enumerate(class_names)}

print("Classes found:", class_to_idx)

all_data = []

for cls in class_names:
    cls_path = Path(input_dir) / cls
    images = list(cls_path.glob("*.jpg")) + list(cls_path.glob("*.png"))

    for img_path in tqdm(images, desc=f"Przetwarzanie klasy: {cls}"):
        img = Image.open(img_path).convert("RGBA")
        orig_w, orig_h = img.size

        for _ in range(num_augmented_per_image):
            scale = random.uniform(0.5, 2.0)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized_img = img.resize((new_w, new_h), resample=Image.LANCZOS)

            canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))

            max_x = canvas_size[0] - new_w
            max_y = canvas_size[1] - new_h

            if max_x < 0 or max_y < 0:
                print(f"Scaled image {img_path.name} is bigger than the background - skipping...")
                continue

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            canvas.paste(resized_img, (x, y), resized_img)

            canvas_rgb = canvas.convert("RGB")

            center_x = (x + new_w / 2) / canvas_size[0]
            center_y = (y + new_h / 2) / canvas_size[1]
            width = new_w / canvas_size[0]
            height = new_h / canvas_size[1]

            all_data.append((canvas_rgb, cls, center_x, center_y, width, height))

random.shuffle(all_data)
split_idx = int(len(all_data) * split_ratio)
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]

def save_yolo_dataset(data, split):
    for i, (img, cls, cx, cy, w, h) in enumerate(tqdm(data, desc=f"Zapis {split}")):
        img_name = f"{cls}_{i:05d}.jpg"
        label_name = img_name.replace(".jpg", ".txt")

        img.save(f"{output_dir}/images/{split}/{img_name}")

        with open(f"{output_dir}/labels/{split}/{label_name}", "w") as f:
            f.write(f"{class_to_idx[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

save_yolo_dataset(train_data, "train")
save_yolo_dataset(val_data, "val")

print("✅ Done! YOLO dataset saved in:", output_dir)
