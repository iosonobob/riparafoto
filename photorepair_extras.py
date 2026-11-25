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

def _rotate_image(img: np.ndarray, angle: float):
    if angle == 0:
        return img, np.eye(3, dtype=np.float32)[:2]
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += (nw / 2.0) - cx
    M[1, 2] += (nh / 2.0) - cy
    rotated = cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    M_homo = np.vstack([M, [0, 0, 1]])
    M_inv = np.linalg.inv(M_homo)[:2]
    return rotated, M_inv

def _transform_point(pt, M_inv):
    x = M_inv[0, 0] * pt[0] + M_inv[0, 1] * pt[1] + M_inv[0, 2]
    y = M_inv[1, 0] * pt[0] + M_inv[1, 1] * pt[1] + M_inv[1, 2]
    return [x, y]

def _dedup_detected(faces_with_lms, center_ratio=0.30):
    if len(faces_with_lms) <= 1:
        return faces_with_lms
    faces_with_lms = sorted(faces_with_lms, key=lambda t: float(t[0][4]), reverse=True)
    kept = []
    for bbox, lm in faces_with_lms:
        x1, y1, x2, y2, sc = bbox[:5]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        w_b, h_b = max(1.0, x2 - x1), max(1.0, y2 - y1)
        sz = max(w_b, h_b)
        drop = False
        for k_bbox, k_lm in kept:
            kx1, ky1, kx2, ky2, ksc = k_bbox[:5]
            kcx, kcy = (kx1 + kx2) / 2.0, (ky1 + ky2) / 2.0
            ksz = max(kx2 - kx1, ky2 - ky1)
            dist = np.hypot(cx - kcx, cy - kcy)
            if dist < max(sz, ksz) * center_ratio:
                drop = True
                break
            x_ov = max(0.0, min(x2, kx2) - max(x1, kx1))
            if x_ov / min(w_b, kx2 - kx1) > 0.6 and (abs(y1 - ky2) < 0.3 * sz or abs(ky1 - y2) < 0.3 * sz):
                drop = True
                break
        if not drop:
            kept.append((bbox, lm))
    return kept

def restore_faces_selective(
    rgb_out: np.ndarray,
    mask: np.ndarray,
    gfpgan,
    det_thr: float = 0.60,
    min_damage: float = 0.05,
    strength: float = 0.7,
    keep_real: float = 0.3,
    max_damage: float = 0.60,
    gfp_weight: float = 0.5,
    logger=None,
) -> tuple[np.ndarray, list[dict]]:
    """
    Selectively restores faces with GFPGAN only if they intersect the damage mask.
    Deduplicates boxes, supports multi-angle detection probe, and avoids warping healthy faces.
    """
    if gfpgan is None:
        return rgb_out, []

    h, w = rgb_out.shape[:2]
    _patch_detection_threshold(gfpgan, det_thr, logger=logger)

    bgr = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
    face_helper = gfpgan.face_helper
    face_helper.clean_all()

    # Multi-orientation detection probe: 0°, then ±20°, ±35° if <2 faces found
    all_detected: list[tuple[list, np.ndarray]] = []
    angles = [0]
    for angle in angles:
        rot_bgr, M_inv = _rotate_image(bgr, angle)
        face_helper.clean_all()
        face_helper.read_image(rot_bgr)
        try:
            face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        except Exception:
            continue
        if not face_helper.det_faces:
            continue
        for bbox, lms in zip(face_helper.det_faces, face_helper.all_landmarks_5):
            if angle == 0:
                all_detected.append((list(bbox), np.array(lms, dtype=np.float32)))
            else:
                x1, y1, x2, y2, sc = bbox[:5]
                corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                t_corners = np.array([_transform_point(c, M_inv) for c in corners])
                orig_bbox = [t_corners[:, 0].min(), t_corners[:, 1].min(),
                             t_corners[:, 0].max(), t_corners[:, 1].max(), sc]
                orig_lms = np.array([_transform_point(pt, M_inv) for pt in lms], dtype=np.float32)
                all_detected.append((orig_bbox, orig_lms))

    # If only 0 or 1 face found, probe tilted angles to capture severely rolled faces
    if len(all_detected) < 2:
        for angle in [-20, 20, -35, 35]:
            rot_bgr, M_inv = _rotate_image(bgr, angle)
            face_helper.clean_all()
            face_helper.read_image(rot_bgr)
            try:
                face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
            except Exception:
                continue
            if not face_helper.det_faces:
                continue
            for bbox, lms in zip(face_helper.det_faces, face_helper.all_landmarks_5):
                x1, y1, x2, y2, sc = bbox[:5]
                corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                t_corners = np.array([_transform_point(c, M_inv) for c in corners])
                orig_bbox = [t_corners[:, 0].min(), t_corners[:, 1].min(),
                             t_corners[:, 0].max(), t_corners[:, 1].max(), sc]
                orig_lms = np.array([_transform_point(pt, M_inv) for pt in lms], dtype=np.float32)
                all_detected.append((orig_bbox, orig_lms))
            if len(all_detected) >= 2:
                break

    # Deduplicate overlapping boxes and vertical splits
    deduped = _dedup_detected(all_detected, center_ratio=0.30)
    if not deduped:
        return rgb_out, []

    scores = [round(float(b[0][4]), 2) for b in deduped]
    if logger: logger(f"GFPGAN: {len(deduped)} volti deduplicati (scores: {scores})")

    face_info = []
    for bbox, _ in deduped:
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

        face_info.append({"box": box, "azione": azione, "damage": round(dmg_ratio, 3)})

    restore_indices = [i for i, f in enumerate(face_info) if f["azione"] == "ripristinato"]
    if not restore_indices:
        return rgb_out, face_info

    # Setup face_helper with ONLY the faces to restore: healthy faces are NOT touched
    face_helper.clean_all()
    face_helper.read_image(bgr)
    face_helper.det_faces = [deduped[i][0] for i in restore_indices]
    face_helper.all_landmarks_5 = [deduped[i][1] for i in restore_indices]

    try:
        face_helper.align_warp_face()
    except Exception as e:
        if logger: logger(f"[GFPGAN] errore allineamento volti: {e}")
        return rgb_out, face_info

    for cropped_face in face_helper.cropped_faces:
        try:
            cropped_face_t = torch.from_numpy(cropped_face.transpose(2, 0, 1)).float() / 255.0
            cropped_face_t = cropped_face_t[[2, 1, 0], :, :]  # BGR to RGB
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(gfpgan.device)
            with torch.no_grad():
                output = gfpgan.gfpgan(cropped_face_t, return_rgb=False, weight=gfp_weight)[0]
                out_img = output.squeeze(0).permute(1, 2, 0).clamp_(-1, 1).add_(1).div_(2).mul_(255)
                out_bgr = out_img.cpu().numpy().astype(np.uint8)[:, :, [2, 1, 0]]
                face_helper.add_restored_face(out_bgr)
        except Exception as e:
            if logger: logger(f"[GFPGAN] errore inferenza volto: {e}")
            face_helper.add_restored_face(cropped_face)

    try:
        face_helper.get_inverse_affine(None)
        restored_bgr = face_helper.paste_faces_to_input_image()
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        if logger: logger(f"[GFPGAN] errore paste-back: {e}")
        return rgb_out, face_info

    # Blend specifically only inside restored face boxes intersecting damaged pixels
    restored_mask = np.zeros((h, w), dtype=np.float32)
    for idx in restore_indices:
        x1, y1, x2, y2 = face_info[idx]["box"]
        bx1, by1 = max(0, x1), max(0, y1)
        bx2, by2 = min(w, x2), min(h, y2)
        if mask is not None and mask.any():
            box_dmg = (mask[by1:by2, bx1:bx2] > 0).astype(np.uint8)
            box_dmg = cv2.dilate(box_dmg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            restored_mask[by1:by2, bx1:bx2] = np.maximum(restored_mask[by1:by2, bx1:bx2], box_dmg.astype(np.float32))
        else:
            restored_mask[by1:by2, bx1:bx2] = 1.0

    restored_mask = cv2.GaussianBlur(restored_mask, (15, 15), 0)[..., None]
    total_w = keep_real + strength
    w_orig = keep_real / total_w if total_w > 0 else 0.3
    w_rest = strength / total_w if total_w > 0 else 0.7
    restored_blend = rgb_out.astype(np.float32) * w_orig + restored_rgb.astype(np.float32) * w_rest
    out = np.clip(
        rgb_out.astype(np.float32) * (1.0 - restored_mask) + restored_blend * restored_mask,
        0, 255
    ).astype(np.uint8)

    return out, face_info
