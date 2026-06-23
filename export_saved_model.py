import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import tensorflow as tf


IMAGE_SIZE = (160, 160)
DEFAULT_H5_MODEL = "resources/models/icaonet_with_decoder.h5"
DEFAULT_OUTPUT_ROOT = "resources/models"
DEFAULT_INPUT_COLOR_ORDER = "rgb"


@dataclass(frozen=True)
class ExportSpec:
    name: str
    output_dir: str
    branches: Tuple[str, ...]


EXPORT_SPECS = (
    ExportSpec(
        name="all",
        output_dir="saved_model_icaonet_all_branches",
        branches=("decoded", "output_reqs", "output_eyes", "output_pixelation"),
    ),
    ExportSpec(
        name="without_decoder",
        output_dir="saved_model_icaonet_without_decoder",
        branches=("output_reqs", "output_eyes", "output_pixelation"),
    ),
    ExportSpec(
        name="without_decoder_eyes",
        output_dir="saved_model_icaonet_without_decoder_eyes",
        branches=("output_reqs", "output_pixelation"),
    ),
    ExportSpec(
        name="output_reqs",
        output_dir="saved_model_icaonet_output_reqs",
        branches=("output_reqs",),
    ),
)


class ICAONetSavedModel(tf.Module):
    """TF2 SavedModel wrapper for a selected set of ICAONet branches."""

    def __init__(self, model, output_names, input_color_order):
        super().__init__()
        self.model = model
        self.output_names = tuple(output_names)
        self.input_name = model.inputs[0].name.split(":")[0]
        self.input_color_order = input_color_order
        self.image_input_name = f"image_{input_color_order}"
        self.image_uint8_input_name = f"image_{input_color_order}_uint8"

        self.predict = tf.function(
            self._predict,
            input_signature=[
                tf.TensorSpec(
                    [None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
                    tf.float32,
                    name=self.image_input_name,
                )
            ],
        )
        self.predict_uint8 = tf.function(
            self._predict_uint8,
            input_signature=[
                tf.TensorSpec(
                    [None, IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
                    tf.uint8,
                    name=self.image_uint8_input_name,
                )
            ],
        )

    def _prepare_image(self, image):
        if self.input_color_order == "rgb":
            # ICAONet was trained with OpenCV BGR images. RGB SavedModels swap
            # channels at the boundary so callers can pass standard RGB input.
            # Use Slice + Concat instead of Reverse for converter compatibility.
            blue = image[..., 2:3]
            green = image[..., 1:2]
            red = image[..., 0:1]
            return tf.concat([blue, green, red], axis=-1)
        return image

    def _as_output_dict(self, predictions):
        if not isinstance(predictions, (list, tuple)):
            predictions = (predictions,)
        return {
            output_name: prediction
            for output_name, prediction in zip(self.output_names, predictions)
        }

    @tf.autograph.experimental.do_not_convert
    def _predict(self, image):
        image = self._prepare_image(image)
        predictions = self.model({self.input_name: image}, training=False)
        return self._as_output_dict(predictions)

    @tf.autograph.experimental.do_not_convert
    def _predict_uint8(self, image_uint8):
        image = tf.image.convert_image_dtype(image_uint8, tf.float32)
        image = self._prepare_image(image)
        predictions = self.model({self.input_name: image}, training=False)
        return self._as_output_dict(predictions)


def load_source_model(h5_model_path):
    model = tf.keras.models.load_model(h5_model_path, compile=False)
    missing = [
        branch
        for branch in ("decoded", "output_reqs", "output_eyes", "output_pixelation")
        if branch not in {layer.name for layer in model.layers}
    ]
    if missing:
        raise ValueError(
            f"{h5_model_path} is missing required branch layer(s): {', '.join(missing)}"
        )
    return model


def make_branch_model(source_model, branches):
    outputs = [source_model.get_layer(branch).output for branch in branches]
    if len(outputs) == 1:
        outputs = outputs[0]
    return tf.keras.Model(
        inputs=source_model.inputs,
        outputs=outputs,
        name=f"ICAONet_{'_'.join(branches)}",
    )


def output_dir_for_spec(spec, input_color_order):
    if input_color_order == "bgr":
        return spec.output_dir
    return spec.output_dir.replace("saved_model_icaonet_", "saved_model_icaonet_rgb_")


def save_branch_model(source_model, spec, output_root, input_color_order, overwrite):
    export_path = Path(output_root) / output_dir_for_spec(spec, input_color_order)
    if export_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"{export_path} already exists. Pass --overwrite to replace it."
            )
        shutil.rmtree(export_path)

    branch_model = make_branch_model(source_model, spec.branches)
    module = ICAONetSavedModel(branch_model, spec.branches, input_color_order)
    tf.saved_model.save(
        module,
        str(export_path),
        signatures={
            "serving_default": module.predict,
            "from_uint8": module.predict_uint8,
        },
    )
    return export_path


def export_all_saved_models(h5_model_path, output_root, input_color_order, overwrite):
    source_model = load_source_model(h5_model_path)
    exported = []
    for spec in EXPORT_SPECS:
        export_path = save_branch_model(
            source_model, spec, output_root, input_color_order, overwrite
        )
        exported.append((spec, export_path))
    return exported


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export ICAONet TensorFlow 2 SavedModels with different output "
            "branch combinations."
        )
    )
    parser.add_argument(
        "--h5-model",
        default=DEFAULT_H5_MODEL,
        help=(
            "Path to the Keras h5 ICAONet model with decoder and all inference "
            "branches."
        ),
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where the SavedModel directories will be written.",
    )
    parser.add_argument(
        "--input-color-order",
        choices=("rgb", "bgr"),
        default=DEFAULT_INPUT_COLOR_ORDER,
        help=(
            "Color order accepted by the exported SavedModels. Use rgb for "
            "standard RGB callers; the wrapper will swap RGB to BGR before "
            "calling ICAONet. Use bgr to preserve the original OpenCV input "
            "contract."
        ),
    )
    overwrite_group = parser.add_mutually_exclusive_group()
    overwrite_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Replace existing output directories. This is the default.",
    )
    overwrite_group.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Fail if an output directory already exists.",
    )
    parser.set_defaults(overwrite=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exported_models = export_all_saved_models(
        h5_model_path=args.h5_model,
        output_root=args.output_root,
        input_color_order=args.input_color_order,
        overwrite=args.overwrite,
    )
    for spec, export_path in exported_models:
        print(
            f"{spec.name}: {export_path} "
            f"({', '.join(spec.branches)}; input={args.input_color_order})"
        )
