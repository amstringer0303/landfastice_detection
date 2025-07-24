import os
import numpy as np
import rasterio
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# === CONFIG ===
rgb_dir = r"D:\UNET\tiles\images"
dist_dir = r"D:\UNET\tiles\distance"
mask_dir = r"D:\UNET\tiles\masks"
model_path = r"D:\UNET\models\unet_landfastice_distance.keras"
tile_size = 320
n_classes = 4
nodata_value = 255

# === Load all tile sets ===
rgb_tiles = sorted(glob(os.path.join(rgb_dir, "*.tif")))
X = []
y = []

print(f"📦 Loading {len(rgb_tiles)} tiles...")

for rgb_tile in rgb_tiles:
    base = os.path.basename(rgb_tile).replace(".tif", "")
    row_col = "_".join(base.split("_")[-2:])

    dist_tile = os.path.join(dist_dir, f"{base}_distance.tif")
    mask_tile = os.path.join(mask_dir, f"{base}.tif")

    if not os.path.exists(dist_tile) or not os.path.exists(mask_tile):
        continue

    with rasterio.open(rgb_tile) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32) / 255.0

    with rasterio.open(dist_tile) as src:
        dist = src.read(1).astype(np.float32)
        dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
        dist = dist[np.newaxis, :, :]

    with rasterio.open(mask_tile) as src:
        mask = src.read(1)
        mask[mask == nodata_value] = 0  # Replace nodata with background (optional)

    X.append(np.concatenate([rgb, dist], axis=0).transpose(1, 2, 0))
    y.append(tf.keras.utils.to_categorical(mask, num_classes=n_classes))

X = np.stack(X)
y = np.stack(y)

print(f"✅ Final dataset shape: {X.shape}, Labels: {y.shape}")

# === Train/val split ===
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)

# === Define U-Net ===
def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    return x

def unet_model(input_shape=(tile_size, tile_size, 4), n_classes=4):
    inputs = layers.Input(shape=input_shape)

    c1 = conv_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = conv_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = conv_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = conv_block(p3, 256)
    p4 = layers.MaxPooling2D()(c4)

    c5 = conv_block(p4, 512)

    u6 = layers.UpSampling2D()(c5)
    u6 = layers.Concatenate()([u6, c4])
    c6 = conv_block(u6, 256)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.Concatenate()([u7, c3])
    c7 = conv_block(u7, 128)

    u8 = layers.UpSampling2D()(c7)
    u8 = layers.Concatenate()([u8, c2])
    c8 = conv_block(u8, 64)

    u9 = layers.UpSampling2D()(c8)
    u9 = layers.Concatenate()([u9, c1])
    c9 = conv_block(u9, 32)

    outputs = layers.Conv2D(n_classes, 1, activation='softmax')(c9)

    return models.Model(inputs, outputs)

# === Custom Weighted Loss ===
weights = tf.constant([2.5, 5.0, 1.0, 0.5], dtype=tf.float32)

def weighted_cce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    weights_map = tf.reduce_sum(y_true * weights, axis=-1)
    return loss * weights_map

# === Compile + Train ===
model = unet_model()
model.compile(optimizer='adam', loss=weighted_cce, metrics=['accuracy'])

print("🚀 Training...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=25, batch_size=8)

# === Save Model ===
os.makedirs(os.path.dirname(model_path), exist_ok=True)
model.save(model_path)
print(f"✅ Model saved to {model_path}")
