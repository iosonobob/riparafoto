#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
photorepair_extras.py - Advanced selective face restoration and detection helpers.
"""
from __future__ import annotations
import sys
import numpy as np
import cv2

try:
    import torch
    import torchvision.transforms.functional as F
    sys.modules['torchvision.transforms.functional_tensor'] = F
    from torchvision.transforms.functional import normalize
except Exception:
    torch = None

def _patch_detection_threshold(gfpgan, det_thr: float, logger=None) -> bool:
    """Lower the face detector score threshold from default 0.97 to det_thr."""
    if not hasattr(gfpgan, "face_helper") or not hasattr(gfpgan.face_helper, "face_det"):
        msg = "[GFPGAN] face_helper o face_det mancante, patch soglia saltata"
        if logger: logger(msg)
        return False
    det = gfpgan.face_helper.face_det
    if hasattr(det, "detect_faces"):
        orig_detect = getattr(det, "_orig_detect_faces", det.detect_faces)
        det._orig_detect_faces = orig_detect
        det.detect_faces = lambda img, threshold=0.97: orig_detect(img, det_thr)
        return True
    else:
        msg = f"[GFPGAN] face_det ({type(det).__name__}) non ha 'detect_faces': controlla nome interno"
        if logger: logger(msg)
        return False

def restore_faces_selective(
    rgb_out: np.ndarray,
    mask: np.ndarray,
    gfpgan,
    det_thr: float = 0.55,
    min_damage: float = 0.02,
    strength: float = 0.6,
    keep_real: float = 0.4,
    max_damage: float = 0.80,
    logger=None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Selectively restores faces with GFPGAN only if they intersect the damage mask.
    Returns (rgb_out, face_info) where face_info details the action for each detected face.
    """
    if gfpgan is None:
        return rgb_out, []

    h, w = rgb_out.shape[:2]
    _patch_detection_threshold(gfpgan, det_thr, logger=logger)

    bgr = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    face_helper = gfpgan.face_helper
    face_helper.clean_all()
    face_helper.read_image(bgr)

    try:
        face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
    except Exception as e:
        if logger: logger(f"[GFPGAN] errore estrazione landmark: {e}")
        return rgb_out, []

    if len(face_helper.det_faces) == 0:
        return rgb_out, []

    face_info = []
    needs_restore = False

    for bbox in face_helper.det_faces:
        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        box = (x1, y1, x2, y2)
        bx1, by1 = max(0, x1), max(0, y1)
        bx2, by2 = min(w, x2), min(h, y2)
        face_area = max(1, (bx2 - bx1) * (by2 - by1))
        dmg_pixels = np.count_nonzero(mask[by1:by2, bx1:bx2]) if mask is not None else 0
        dmg_ratio = dmg_pixels / face_area

        if dmg_ratio < min_damage:
            azione = "saltato perché sano"
        elif dmg_ratio > max_damage:
            azione = "saltato perché troppo rovinato"
        else:
            azione = "ripristinato"
            needs_restore = True

        face_info.append({"box": box, "azione": azione, "damage": dmg_ratio})

    if not needs_restore:
        return rgb_out, face_info

    try:
        face_helper.align_warp_face()
    except Exception as e:
        if logger: logger(f"[GFPGAN] errore allineamento volti: {e}")
        return rgb_out, face_info

    for i, cropped_face in enumerate(face_helper.cropped_faces):
        if i < len(face_info) and face_info[i]["azione"] == "ripristinato":
            try:
                cropped_face_t = torch.from_numpy(cropped_face.transpose(2, 0, 1)).float() / 255.0
                cropped_face_t = cropped_face_t[[2, 1, 0], :, :]  # BGR to RGB
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(gfpgan.device)
                with torch.no_grad():
                    output = gfpgan.gfpgan(cropped_face_t, return_rgb=False, weight=0.5)[0]
                    out_img = output.squeeze(0).permute(1, 2, 0).clamp_(-1, 1).add_(1).div_(2).mul_(255)
                    out_bgr = out_img.cpu().numpy().astype(np.uint8)[:, :, [2, 1, 0]]
                    face_helper.add_restored_face(out_bgr)
            except Exception as e:
                if logger: logger(f"[GFPGAN] errore inferenza volto {i}: {e}")
                face_helper.add_restored_face(cropped_face)
        else:
            face_helper.add_restored_face(cropped_face)

    try:
        face_helper.get_inverse_affine(None)
        restored_bgr = face_helper.paste_faces_to_input_image()
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        if logger: logger(f"[GFPGAN] errore paste-back: {e}")
        return rgb_out, face_info

    total_w = keep_real + strength
    w_orig = keep_real / total_w if total_w > 0 else 0.4
    w_rest = strength / total_w if total_w > 0 else 0.6
    out = np.clip(
        rgb_out.astype(np.float32) * w_orig + restored_rgb.astype(np.float32) * w_rest,
        0, 255
    ).astype(np.uint8)

    return out, face_info
