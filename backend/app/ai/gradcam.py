"""Grad-CAM explainability generator for MediSense AI X-ray models.

Computes Grad-CAM from a real forward pass and gradients of a trained PyTorch
model. It never fabricates a heatmap. Failure to generate an artifact is
reported as unavailable and must not invalidate an otherwise valid prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import uuid

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def _unavailable(message: str, reason_code: str = "gradcam_unavailable") -> dict[str, Any]:
    """Return a consistent, non-fabricated explainability status."""
    return {
        "status": "unavailable",
        "reason_code": reason_code,
        "label": "Model Explainability Visualization",
        "message": message,
        "artifact_path": None,
        "filename": None,
    }


def _read_original_image(image_path: str) -> Optional[np.ndarray]:
    """Read an image safely and return an RGB-correct BGR array for OpenCV."""
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is not None and image.size:
            return image
        with Image.open(image_path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            rgb = np.asarray(pil_image)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
            return None
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, ValueError, cv2.error) as exc:
        logger.warning("Unable to read image for Grad-CAM: %s", exc)
        return None


def _find_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """Prefer the final ResNet block; otherwise use the last convolution."""
    layer4 = getattr(model, "layer4", None)
    if isinstance(layer4, (nn.Sequential, list, tuple)) and len(layer4):
        candidate = layer4[-1]
        if isinstance(candidate, nn.Module):
            return candidate

    last_conv: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def generate_gradcam(
    model: Any,
    image_path: str,
    target_class: int = 0,
) -> dict[str, Any]:
    """Generate a genuine Grad-CAM overlay for a PyTorch convolutional model.

    ``target_class`` is used only when it is a valid class index. If it is not,
    the model's top output class is selected. The output is an explanatory
    visualization, not a clinical localization or diagnostic confirmation.
    """
    if model is None or not isinstance(model, nn.Module):
        return _unavailable(
            "Grad-CAM is unavailable because no PyTorch model is loaded.",
            "model_unavailable",
        )

    if not isinstance(image_path, str) or not image_path.strip():
        return _unavailable("No valid image path was supplied.", "invalid_image_path")

    try:
        image_file = Path(image_path)
        if not image_file.is_file() or image_file.stat().st_size <= 0:
            return _unavailable("The image file is missing or empty.", "invalid_image")
    except OSError:
        return _unavailable("The image file could not be accessed.", "invalid_image")

    original = _read_original_image(image_path)
    if original is None:
        return _unavailable("The uploaded image could not be decoded.", "image_decode_failed")
    height, width = original.shape[:2]

    target_layer = _find_target_layer(model)
    if target_layer is None:
        return _unavailable(
            "The model has no convolutional layer suitable for Grad-CAM.",
            "target_layer_missing",
        )

    forward_handle = None
    activations: list[torch.Tensor] = []
    was_training = model.training

    try:
        # Use the same preprocessing pipeline as inference, but place the
        # tensor on the actual model device rather than a separately inferred device.
        from app.ai.xray_model import XRayModelService

        service = XRayModelService()
        tensor = service.preprocess_image(image_path)
        if tensor is None:
            return _unavailable(
                "Image preprocessing failed for Grad-CAM computation.",
                "preprocessing_failed",
            )

        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = tensor.device
        tensor = tensor.to(model_device)
        tensor.requires_grad_(True)

        def capture_activation(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, torch.Tensor):
                activations.append(output)

        forward_handle = target_layer.register_forward_hook(capture_activation)
        model.eval()
        # Grad-CAM requires gradients, so do not wrap this block in no_grad().
        with torch.enable_grad():
            outputs = model(tensor)
            if not isinstance(outputs, torch.Tensor) or outputs.ndim != 2 or outputs.shape[0] < 1:
                return _unavailable("The model returned an invalid output shape.", "invalid_model_output")
            if outputs.shape[1] < 1 or not torch.isfinite(outputs).all():
                return _unavailable("The model output contained no valid class scores.", "invalid_model_output")
            if not activations:
                return _unavailable("The target layer did not produce activations.", "activation_missing")

            class_index = target_class
            if not isinstance(class_index, int) or isinstance(class_index, bool) or not 0 <= class_index < outputs.shape[1]:
                class_index = int(outputs[0].argmax(dim=0).item())

            activation = activations[-1]
            if activation.ndim != 4 or activation.shape[0] < 1:
                return _unavailable("The target layer output is not a valid spatial feature map.", "invalid_activation_shape")

            gradients = torch.autograd.grad(
                outputs[0, class_index], activation, retain_graph=False, create_graph=False, allow_unused=True
            )[0]
            if gradients is None or not torch.isfinite(gradients).all():
                return _unavailable("Gradients could not be calculated for the selected class.", "gradients_unavailable")

            # Global-average-pool gradients and weight the feature maps.
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam_tensor = torch.relu((weights * activation).sum(dim=1))[0]
            cam = cam_tensor.detach().float().cpu().numpy()

        if cam.size == 0 or not np.isfinite(cam).all():
            return _unavailable("The generated activation map was empty or invalid.", "invalid_heatmap")

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        spread = cam_max - cam_min
        # A flat activation map carries no useful spatial explanation. Do not
        # save a misleading all-zero/all-one visualization as if it were useful.
        if not np.isfinite(spread) or spread <= 1e-12:
            return _unavailable("The model produced no meaningful spatial activation for this class.", "flat_heatmap")
        cam = (cam - cam_min) / spread
        cam_resized = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(np.clip(cam_resized * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.65, heatmap, 0.35, 0)

        settings = get_settings()
        explain_dir = Path(settings.upload_path) / "explainability"
        explain_dir.mkdir(parents=True, exist_ok=True)
        filename = f"gradcam_{uuid.uuid4().hex}.jpg"
        output_path = explain_dir / filename
        if not cv2.imwrite(str(output_path), overlay):
            return _unavailable("The Grad-CAM image could not be saved.", "artifact_write_failed")

        return {
            "status": "ok",
            "reason_code": None,
            "label": "Model Explainability Visualization (Grad-CAM)",
            "message": (
                "Grad-CAM was generated from the model's feature activations and gradients. "
                "It is an assistive visualization, not a diagnosis or proof that a highlighted area is abnormal."
            ),
            "artifact_path": str(output_path),
            "filename": filename,
            "target_class_index": class_index,
        }
    except Exception as exc:
        logger.warning("Grad-CAM generation failed: %s", exc, exc_info=True)
        return _unavailable(
            "Grad-CAM could not be generated for this image and model. The model prediction, if available, should be reported separately.",
            "gradcam_execution_failed",
        )
    finally:
        if forward_handle is not None:
            try:
                forward_handle.remove()
            except Exception:
                logger.debug("Could not remove Grad-CAM forward hook.")
        # Avoid changing the caller's model mode permanently.
        try:
            model.train(was_training)
        except Exception:
            logger.debug("Could not restore model training mode after Grad-CAM.")
