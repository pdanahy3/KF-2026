import argparse
import os

import tensorflow as tf
from keras.models import load_model


def rounded_accuracy(y_true, y_pred):
    # Kept only to load the original training-time model cleanly.
    from keras import backend as K
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def resolve_path(path: str) -> str:
    path = os.path.expanduser(path)
    if os.path.isabs(path):
        return path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base_dir, path))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DST encoder (.keras) to ONNX.")
    p.add_argument("--keras-path", default="./checkpoints/saved_model.keras", help="Path to .keras file.")
    p.add_argument("--onnx-path", default="./checkpoints/encoder.onnx", help="Output path for .onnx file.")
    p.add_argument("--img-width", type=int, default=256)
    p.add_argument("--img-height", type=int, default=256)
    p.add_argument("--opset", type=int, default=17, help="ONNX opset (17 recommended).")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        import tf2onnx  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency tf2onnx. Install with: pip install tf2onnx onnx"
        ) from exc

    import tf2onnx

    keras_path = resolve_path(args.keras_path)
    onnx_path = resolve_path(args.onnx_path)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"Keras model not found: {keras_path}")

    model = load_model(keras_path, custom_objects={"rounded_accuracy": rounded_accuracy})

    # Match DST.py behavior: use the penultimate layer as "encoder features".
    encoder = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

    spec = (
        tf.TensorSpec(
            (None, int(args.img_height), int(args.img_width), 3),
            tf.float32,
            name="input",
        ),
    )

    tf2onnx.convert.from_keras(
        encoder,
        input_signature=spec,
        opset=int(args.opset),
        output_path=onnx_path,
    )

    print(f"Wrote ONNX encoder to: {onnx_path}")


if __name__ == "__main__":
    main()

