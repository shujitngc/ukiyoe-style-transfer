from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime
from pathlib import Path

app = FastAPI()

generator_g_path = "models/generator_g_200.h5"
generator_f_path = "models/generator_f_200.h5"

save_dir = Path("outputs")
save_dir.mkdir(exist_ok=True)


class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name="scale",
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(1.0, 0.02),
            trainable=True
        )
        self.offset = self.add_weight(
            name="offset",
            shape=input_shape[-1:],
            initializer="zeros",
            trainable=True
        )

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset


generator_g = tf.keras.models.load_model(
    generator_g_path,
    custom_objects={"InstanceNormalization": InstanceNormalization},
    compile=False
)

generator_f = tf.keras.models.load_model(
    generator_f_path,
    custom_objects={"InstanceNormalization": InstanceNormalization},
    compile=False
)


def preprocess_image(image: Image.Image):
    original_size = image.size
    image = image.convert("RGB")
    image = image.resize((1024, 1024))
    image = np.array(image).astype(np.float32)
    image = (image / 127.5) - 1.0
    image = np.expand_dims(image, axis=0)
    return image, original_size


def deprocess_image(image: np.ndarray, original_size):
    image = (image[0] * 0.5 + 0.5) * 255
    image = np.clip(image, 0, 255).astype(np.uint8)
    image = Image.fromarray(image)
    image = image.resize(original_size)
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = "g2z"):
    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    input_image, original_size = preprocess_image(image)

    if model == "g2z":
        output = generator_g(input_image, training=False)
    elif model == "z2g":
        output = generator_f(input_image, training=False)
    else:
        raise HTTPException(status_code=400, detail="Invalid model type")

    output_image = deprocess_image(output.numpy(), original_size)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = save_dir / f"transformed_{model}_{timestamp}.jpg"
    output_image.save(save_path)

    buffer = io.BytesIO()
    output_image.save(buffer, format="JPEG")
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")