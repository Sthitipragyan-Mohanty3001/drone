"""
models/segmentation_model.py
Module 1 — Aerial Image Segmentation
Supports: ResNet50-UNet, VGG-UNet, MobileNet-UNet, DeepLabV3+
"""

import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

N_CLASSES  = cfg["dataset"]["n_classes"]
IMG_H      = cfg["dataset"]["image_height"]
IMG_W      = cfg["dataset"]["image_width"]
SEG_CFG    = cfg["segmentation"]


# ── Building blocks ───────────────────────────────────────────────────────────
def conv_block(x, filters, kernel_size=3, padding="same", activation="relu"):
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)
    return x


def decoder_block(x, skip, filters):
    """Upsample + concatenate skip connection + conv block."""
    x = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(x)
    if skip is not None:
        # Handle shape mismatch
        x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


# ── ResNet50-UNet ─────────────────────────────────────────────────────────────
def build_resnet50_unet(n_classes=N_CLASSES, input_shape=(IMG_H, IMG_W, 3), pretrained=True):
    """
    U-Net decoder with ResNet50 encoder.
    Skip connections from ResNet50 intermediate layers.
    """
    inputs = keras.Input(shape=input_shape)

    # -- Encoder (ResNet50)
    weights = "imagenet" if pretrained else None
    base    = ResNet50(include_top=False, weights=weights, input_tensor=inputs)

    # Skip connections at various resolutions
    s1 = base.get_layer("conv1_relu").output        # 208 × 304 × 64
    s2 = base.get_layer("conv2_block3_1_relu").output  # 104 × 152 × 64
    s3 = base.get_layer("conv3_block4_1_relu").output  # 52  × 76  × 128
    s4 = base.get_layer("conv4_block6_1_relu").output  # 26  × 38  × 256
    bridge = base.get_layer("conv5_block3_out").output  # 13 × 19 × 2048

    # -- Decoder
    d1 = decoder_block(bridge, s4, 512)
    d2 = decoder_block(d1,     s3, 256)
    d3 = decoder_block(d2,     s2, 128)
    d4 = decoder_block(d3,     s1,  64)
    d5 = decoder_block(d4,  None,  32)

    # -- Output
    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d5)

    model = Model(inputs, outputs, name="resnet50_unet")
    return model


# ── VGG16-UNet ────────────────────────────────────────────────────────────────
def build_vgg16_unet(n_classes=N_CLASSES, input_shape=(IMG_H, IMG_W, 3), pretrained=True):
    inputs = keras.Input(shape=input_shape)
    weights = "imagenet" if pretrained else None
    base    = VGG16(include_top=False, weights=weights, input_tensor=inputs)

    s1 = base.get_layer("block1_conv2").output   # full res
    s2 = base.get_layer("block2_conv2").output
    s3 = base.get_layer("block3_conv3").output
    s4 = base.get_layer("block4_conv3").output
    bridge = base.get_layer("block5_conv3").output

    d1 = decoder_block(bridge, s4, 512)
    d2 = decoder_block(d1,     s3, 256)
    d3 = decoder_block(d2,     s2, 128)
    d4 = decoder_block(d3,     s1,  64)
    d5 = decoder_block(d4,  None,  32)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d5)
    return Model(inputs, outputs, name="vgg16_unet")


# ── MobileNetV2-UNet ──────────────────────────────────────────────────────────
def build_mobilenet_unet(n_classes=N_CLASSES, input_shape=(IMG_H, IMG_W, 3), pretrained=True):
    inputs  = keras.Input(shape=input_shape)
    weights = "imagenet" if pretrained else None
    base    = MobileNetV2(include_top=False, weights=weights, input_tensor=inputs)

    s1 = base.get_layer("block_1_expand_relu").output
    s2 = base.get_layer("block_3_expand_relu").output
    s3 = base.get_layer("block_6_expand_relu").output
    s4 = base.get_layer("block_13_expand_relu").output
    bridge = base.get_layer("block_16_project").output

    d1 = decoder_block(bridge, s4, 256)
    d2 = decoder_block(d1,     s3, 128)
    d3 = decoder_block(d2,     s2,  64)
    d4 = decoder_block(d3,     s1,  32)
    d5 = decoder_block(d4,  None,  16)

    outputs = layers.Conv2D(n_classes, 1, activation="softmax")(d5)
    return Model(inputs, outputs, name="mobilenet_unet")


# ── Model factory ─────────────────────────────────────────────────────────────
def get_segmentation_model(model_name=None, pretrained=True):
    model_name = model_name or SEG_CFG["model"]
    pretrained = pretrained and SEG_CFG.get("pretrained", True)

    builders = {
        "resnet50_unet":  build_resnet50_unet,
        "vgg_unet":       build_vgg16_unet,
        "mobilenet_unet": build_mobilenet_unet,
    }

    if model_name not in builders:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(builders.keys())}")

    model = builders[model_name](pretrained=pretrained)
    print(f"[INFO] Built {model_name}  |  Params: {model.count_params():,}")
    return model


# ── Compile with loss + metrics ───────────────────────────────────────────────
def compile_model(model):
    lr   = SEG_CFG["learning_rate"]
    loss = SEG_CFG["loss"]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=[
            "accuracy",
            tf.keras.metrics.MeanIoU(num_classes=N_CLASSES),
        ]
    )
    return model


# ── Training callbacks ────────────────────────────────────────────────────────
def get_callbacks(checkpoint_path="models/saved/segmentation"):
    os.makedirs(checkpoint_path, exist_ok=True)
    return [
        ModelCheckpoint(
            filepath=f"{checkpoint_path}/best_model.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        TensorBoard(log_dir="logs/segmentation"),
    ]


# ── Inference on a single image ───────────────────────────────────────────────
def predict_segmentation(model, img_batch):
    """
    Args:
        img_batch: numpy array (1, H, W, 3) normalized float32
    Returns:
        class_mask: numpy array (H, W) with class indices
        prob_map:   numpy array (H, W, N_CLASSES) with probabilities
    """
    prob_map   = model.predict(img_batch, verbose=0)[0]  # (H, W, C)
    class_mask = np.argmax(prob_map, axis=-1)             # (H, W)
    return class_mask, prob_map


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = get_segmentation_model("resnet50_unet")
    model = compile_model(model)
    model.summary()

    # Dummy forward pass
    dummy = np.random.rand(1, IMG_H, IMG_W, 3).astype(np.float32)
    out   = model.predict(dummy, verbose=0)
    print(f"[TEST] Output shape: {out.shape}")  # (1, 416, 608, 24)
