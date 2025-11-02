import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import tensorflow as tf
from onnx2tf import convert


def load_model(num_classes: int) -> torch.nn.Module:
    """
    Load and customize a pretrained MobileNetV3-Small model.

    Args:
        num_classes (int): Number of output classes.

    Returns:
        torch.nn.Module: Customized MobileNetV3-Small model.
    """
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model


def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor, output_path: str) -> None:
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        dummy_input (torch.Tensor): Dummy input tensor for tracing.
        output_path (str): Destination ONNX file path.
    """
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"[INFO] ONNX model saved to: {output_path}")


def convert_to_tflite(saved_model_dir: str, output_path: str) -> None:
    """
    Convert a TensorFlow SavedModel to a TFLite model.

    Args:
        saved_model_dir (str): Directory containing the TensorFlow SavedModel.
        output_path (str): Destination .tflite file path.
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"[INFO] TFLite model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a PyTorch MobileNetV3 model to ONNX and TFLite formats."
    )

    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the PyTorch checkpoint (.pt) file.")
    parser.add_argument("--onnx_output", type=str, default="model.onnx",
                        help="Output path for the ONNX model.")
    parser.add_argument("--tflite_output", type=str, default="model.tflite",
                        help="Output path for the TFLite model.")
    parser.add_argument("--img_size", type=int, default=640,
                        help="Input image size (square).")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of output classes.")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use for model loading and export.")
    parser.add_argument("--saved_model_dir", type=str, default="saved_model",
                        help="Directory name for TensorFlow SavedModel output.")

    args = parser.parse_args()

    # Device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device(args.device)

    # Load and prepare model
    print("[INFO] Loading model...")
    model = load_model(args.num_classes)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval().to(device)
    print("[INFO] Model loaded successfully.")

    # Dummy input for export
    dummy_input = torch.randn(1, 3, args.img_size, args.img_size, device=device)

    # Export to ONNX
    export_to_onnx(model, dummy_input, args.onnx_output)

    # Convert ONNX → TensorFlow → TFLite
    print("[INFO] Converting ONNX to TensorFlow...")
    convert(
        input_onnx_file_path=args.onnx_output,
        output_integer_quantized_tflite=True,
        disable_strict_mode=True,
        output_signaturedefs=False,
    )
    print("[INFO] ONNX successfully converted to TensorFlow.")

    convert_to_tflite(args.saved_model_dir, args.tflite_output)


if __name__ == "__main__":
    main()
