import argparse
from pathlib import Path

import tensorflow as tf


IMAGE_SIZE = (160, 160)
DEFAULT_H5_MODEL = "resources/models/icaonet.h5"
DEFAULT_EXPORT_DIR = "resources/models/icaonet_saved_model"


class ICAONetSavedModel(tf.Module):
    """TF2 SavedModel wrapper for ICAONet inference."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                [None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
                tf.float32,
                name="image",
            )
        ]
    )
    @tf.autograph.experimental.do_not_convert
    def predict(self, image):
        predictions = self.model(image, training=False)
        return {"requirements": predictions}

    @tf.function(
        input_signature=[
            tf.TensorSpec(
                [None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
                tf.uint8,
                name="image_uint8",
            )
        ]
    )
    @tf.autograph.experimental.do_not_convert
    def predict_uint8(self, image_uint8):
        image = tf.image.convert_image_dtype(image_uint8, tf.float32)
        predictions = self.model(image, training=False)
        return {"requirements": predictions}


def export_saved_model(h5_model_path, export_dir):
    model = tf.keras.models.load_model(h5_model_path, compile=False)
    module = ICAONetSavedModel(model)

    export_path = Path(export_dir)
    tf.saved_model.save(
        module,
        str(export_path),
        signatures={
            "serving_default": module.predict,
            "from_uint8": module.predict_uint8,
        },
    )
    return export_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export resources/models/icaonet.h5 as a TensorFlow 2 SavedModel."
    )
    parser.add_argument(
        "--h5-model",
        default=DEFAULT_H5_MODEL,
        help="Path to the Keras h5 ICAONet model.",
    )
    parser.add_argument(
        "--export-dir",
        default=DEFAULT_EXPORT_DIR,
        help="Directory where the SavedModel will be written.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_path = export_saved_model(args.h5_model, args.export_dir)
    print(f"SavedModel exported to: {export_path}")
