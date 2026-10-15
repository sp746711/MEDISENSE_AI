"""Grad-CAM explainability generator for MediSense AI X-ray models.

Computes genuine gradient-weighted class activation mapping (Grad-CAM)
from the final convolutional feature maps of trained PyTorch models.
Never fabricates or simulates heatmaps using OpenCV image blurring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import uuid

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.core.config import get_settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def generate_gradcam(
    model: Any,
    image_path: str,
    target_class: int = 0,
) -> dict[str, Any]:
    """Generate Grad-CAM activation heatmap overlay for visual explainability.

    Requires an active PyTorch convolutional model. When no model is configured,
    returns status='unavailable' with null artifact rather than fabricating
    synthetic OpenCV blur visualizations.
    """
    if model is None or not isinstance(model, nn.Module):
        logger.info("Grad-CAM generation skipped: no trained PyTorch model provided.")
        return {
            "status": "unavailable",
            "label": "Model Explainability Visualization",
            "message": "Grad-CAM explainability is unavailable when model checkpoint is not configured.",
            "artifact_path": None,
        }

    try:
        settings = get_settings()
        explain_dir = settings.upload_path / "explainability"
        explain_dir.mkdir(parents=True, exist_ok=True)

        # Check and load original image for overlay
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            pil_img = Image.open(image_path).convert("RGB")
            orig_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        h, w = orig_img.shape[:2]

        # Locate target convolutional layer (e.g. layer4 in ResNet)
        target_layer = None
        if hasattr(model, "layer4"):
            target_layer = model.layer4[-1]
        else:
            # Fallback: search for last Conv2d module
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    target_layer = m

        if target_layer is None:
            return {
                "status": "unavailable",
                "label": "Model Explainability Visualization",
                "message": "Target convolutional layer not found for Grad-CAM.",
                "artifact_path": None,
            }

        # Hook activations and gradients
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        f_handle = target_layer.register_forward_hook(forward_hook)
        b_handle = target_layer.register_full_backward_hook(backward_hook)

        # Preprocess input tensor
        from app.ai.xray_model import XRayModelService
        svc = XRayModelService()
        tensor = svc.preprocess_image(image_path)
        if tensor is None:
            f_handle.remove()
            b_handle.remove()
            return {
                "status": "unavailable",
                "label": "Model Explainability Visualization",
                "message": "Image preprocessing failed for Grad-CAM computation.",
                "artifact_path": None,
            }

        tensor.requires_grad = True
        model.eval()

        outputs = model(tensor)
        if target_class >= outputs.shape[1]:
            target_class = int(outputs.argmax(dim=1).item())

        model.zero_grad()
        score = outputs[0, target_class]
        score.backward()

        f_handle.remove()
        b_handle.remove()

        if not activations or not gradients:
            return {
                "status": "unavailable",
                "label": "Model Explainability Visualization",
                "message": "Unable to extract gradients for Grad-CAM visualization.",
                "artifact_path": None,
            }

        act = activations[0].detach()
        grad = gradients[0].detach()

        # Global average pool the gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1)).squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize to original image size
        cam_resized = cv2.resize(cam, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        # Alpha blend original image with heatmap
        overlay = cv2.addWeighted(orig_img, 0.65, heatmap, 0.35, 0)

        filename = f"gradcam_{uuid.uuid4().hex[:12]}.jpg"
        out_path = explain_dir / filename
        cv2.imwrite(str(out_path), overlay)

        return {
            "status": "ok",
            "label": "Model Explainability Visualization (Grad-CAM)",
            "message": "Grad-CAM heatmap generated from convolutional feature activation maps.",
            "artifact_path": str(out_path),
            "filename": filename,
        }
    except Exception as exc:
        logger.warning("Grad-CAM generation error: %s", exc)
        return {
            "status": "unavailable",
            "label": "Model Explainability Visualization",
            "message": f"Grad-CAM generation failed: {exc}",
            "artifact_path": None,
        }
