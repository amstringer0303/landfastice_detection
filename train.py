import numpy as np
import tensorflow as tf
import rasterio
import glob
import os
from sklearn.model_selection import train_test_split

# === CONFIGURATION ===
batch_size = 8
epochs = 30
optimizer = 'adam'
loss = 'categorical_crossentropy'
img_height = 320
img_width = 320
img_bands = 4  # <-- ✅ include 4 bands now (RGB + distance)
num_classes = 4

# === PATHS ===
data_path = '/Volumes/toshiba/UNET/tiles/'
distance_path = os.path.join(data_path, 'distance/')
image_files = sorted(glob.glob(os.path.join(data_path, 'images/*.tif')))
mask_files = sorted(glob.glob(os.path.join(data_path, 'masks/*.tif')))

print(f"🖼️ Found {len(image_files)} images and {len(mask_files)} masks")

# === SPLIT DATA ===
train_imgs, test_imgs, train_masks, test_masks = train_test_split(
    image_files, mask_files, test_size=0.2, random_state=42
)
train_imgs, val_imgs, train_masks, val_masks = train_test_split(
    train_imgs, train_masks, test_size=0.1, random_state=42
)
print(f"🔀 Data split: {len(train_imgs)} train / {len(val_imgs)} val / {len(test_imgs)} test")

# === DATA LOADING FUNCTIONS ===
def load_geotiff_image_with_distance(image_path):
    base = os.path.basename(image_path).replace('.tif', '')
    dist_path = os.path.join(distance_path, f"{base}_distance.tif")
    if not os.path.exists(dist_path):
        raise FileNotFoundError(f"Distance raster not found for: {dist_path}")

    with rasterio.open(image_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0
    with rasterio.open(dist_path) as dist_src:
        dist = dist_src.read(1).astype(np.float32)
        dist = np.expand_dims(dist, axis=0)  # shape: (1, H, W)
        dist /= np.max(dist) if np.max(dist) != 0 else 1

    rgbd = np.concatenate([rgb, dist], axis=0)
    rgbd = np.moveaxis(rgbd, 0, -1)  # shape: (H, W, 4)
    return rgbd

def load_mask_image(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1).astype(np.uint8)
    mask = np.where(mask == 255, 0, mask)  # map NODATA to class 0
    return tf.keras.utils.to_categorical(mask, num_classes=num_classes)

def create_tf_dataset(image_paths, mask_paths):
    images, masks = [], []
    for img_path, mask_path in zip(image_paths, mask_paths):
        images.append(load_geotiff_image_with_distance(img_path))
        masks.append(load_mask_image(mask_path))
    print(f"📦 Loaded {len(images)} image/mask pairs into memory")
    return tf.data.Dataset.from_tensor_slices((np.array(images), np.array(masks))).shuffle(len(images)).batch(batch_size)

print("📂 Loading TF datasets...")
train_ds = create_tf_dataset(train_imgs, train_masks)
val_ds = create_tf_dataset(val_imgs, val_masks)
test_ds = create_tf_dataset(test_imgs, test_masks)

# === U-NET MODEL ===
def unet_model(input_shape=(img_height, img_width, img_bands)):
    inputs = tf.keras.Input(shape=input_shape)
    c1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D(2)(c1)

    c2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D(2)(c2)

    b = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p2)

    u1 = tf.keras.layers.UpSampling2D(2)(b)
    u1 = tf.keras.layers.Concatenate()([u1, c2])
    c3 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u1)

    u2 = tf.keras.layers.UpSampling2D(2)(c3)
    u2 = tf.keras.layers.Concatenate()([u2, c1])
    c4 = tf.keras.layers.Conv2D(16, 3, activation='relu', padding='same')(u2)

    outputs = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax')(c4)
    return tf.keras.Model(inputs, outputs)

print("🧠 Building model...")
model = unet_model()
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print("🚀 Starting training...")
model.fit(train_ds, validation_data=val_ds, epochs=epochs)

print("🧪 Evaluating on test set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"\n✅ Final test accuracy: {test_acc:.4f}")

# === SAVE TO DESKTOP ===
desktop_path = os.path.expanduser("~/Desktop/unet-4class-RGBD.keras")
print("💾 Saving model to Desktop...")
model.save(desktop_path)
print(f"🎉 Done! Model saved at: {desktop_path}")
