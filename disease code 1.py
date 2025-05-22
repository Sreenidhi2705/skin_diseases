import os
import shutil
import zipfile
from sklearn.model_selection import train_test_split

# Step 1: Define correct ZIP path and extract
zip_path = '/content/skin diseasess.zip'  # <-- Make sure the file is uploaded here
extract_path = '/content/dataset_raw'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# Step 2: Auto-detect the dataset folder
def find_data_root(root_path):
    for root, dirs, files in os.walk(root_path):
        if dirs and all(os.path.isdir(os.path.join(root, d)) for d in dirs):
            return root
    return None

original_dataset_path = find_data_root(extract_path)
if not original_dataset_path:
    raise Exception("Could not locate dataset folder.")

print(f"Dataset root found at: {original_dataset_path}")

# Step 3: Create train and validation directories
base_dir = '/content/skin_diseases_split'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Step 4: Split and copy images
for class_name in os.listdir(original_dataset_path):
    class_path = os.path.join(original_dataset_path, class_name)
    if os.path.isdir(class_path):
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Class '{class_name}': {len(images)} images found")
        if len(images) == 0:
            print(f"No images found in class folder '{class_name}', skipping...")
            continue
        
        train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(train_dir, class_name, img))
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img), os.path.join(val_dir, class_name, img))

print("Dataset successfully split into train/ and validation/ folders.")