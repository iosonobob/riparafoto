#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
riparafoto.py - Riparazione avanzata (Analisi locale varianza, LaMa puro, SSL bypass).
"""
from __future__ import annotations
import ssl

# FIX SSL per i download da HuggingFace
ssl._create_default_https_context = ssl._create_unverified_context

import argparse
import gc
import io
import os
import queue
import re
import struct
import subprocess
import sys
import tempfile
import threading
import traceback
import warnings
import zlib
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from PIL import Image, ImageChops, ImageFile
except ImportError:
    sys.exit("Manca Pillow. Installalo con: pip install pillow")

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None

# --- INTEGRAZIONE GPU E AI ---
TORCH_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE_OBJ = torch.device(DEVICE)
    print(f"\n{'='*60}")
    if DEVICE == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[SISTEMA] ✅ GPU RILEVATA: {gpu_name}")
    else:
        print("[ERRORE CRITICO] GPU non trovata. L'AI userà la CPU (lentissima).")
except ImportError:
    DEVICE = "cpu"
    DEVICE_OBJ = None
    print("\n[WARN] PyTorch non installato. Funzionalità AI disabilitate.")

# 1. LaMa
_LAMA = None
_LAMA_LOCK = threading.Lock()
def _get_lama():
    global _LAMA
    if not TORCH_AVAILABLE: return None
    with _LAMA_LOCK:
        if _LAMA is None:
            try:
                from simple_lama_inpainting import SimpleLama
                _LAMA = SimpleLama(device=DEVICE_OBJ)
                if hasattr(_LAMA, "model") and hasattr(_LAMA.model, "to"):
                    _LAMA.model.to(DEVICE_OBJ)
                print(f"[INFO] ✅ LaMa caricato.")
            except Exception as e:
                print(f"[ERR] ❌ Errore caricamento LaMa: {e}")
                _LAMA = False
    return _LAMA or None

# 2. SDXL
_SDXL = None
_SDXL_LOCK = threading.Lock()
def _get_sdxl():
    global _SDXL
    if not TORCH_AVAILABLE: return None
    with _SDXL_LOCK:
        if _SDXL is None:
            try:
                from diffusers import StableDiffusionXLInpaintPipeline
                print(f"[INFO] Scarico/Carico SDXL Inpainting (attendere, ~7GB)...")
                pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    safety_checker=None
                )
                
                if hasattr(pipe, "enable_model_cpu_offload"):
                    pipe.enable_model_cpu_offload()
                
                if hasattr(pipe, "vae"):
                    if hasattr(pipe.vae, "enable_slicing"): pipe.vae.enable_slicing()
                    if hasattr(pipe.vae, "enable_tiling"): pipe.vae.enable_tiling()
                elif hasattr(pipe, "enable_vae_slicing"):
                    pipe.enable_vae_slicing()
                    if hasattr(pipe, "enable_vae_tiling"):
                        pipe.enable_vae_tiling()

                _SDXL = pipe
                print(f"[INFO] ✅ SDXL caricato.")
            except Exception as e:
                print(f"\n[ERR] ❌ ERRORE CRITICO SDXL: {type(e).__name__}: {e}\n")
                traceback.print_exc()
                _SDXL = False
    return _SDXL or None

# 3. GFPGAN
_GFPGAN = None
_GFPGAN_LOCK = threading.Lock()
def _get_gfpgan():
    global _GFPGAN
    if not TORCH_AVAILABLE: return None
    with _GFPGAN_LOCK:
        if _GFPGAN is None:
            try:
                import torchvision.transforms.functional as F
                sys.modules['torchvision.transforms.functional_tensor'] = F
                from gfpgan import GFPGANer
                print(f"[INFO] Scarico/Carico GFPGAN...")
                _GFPGAN = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)
                print(f"[INFO] ✅ GFPGAN caricato.")
            except Exception as e:
                print(f"\n[ERR] ❌ ERRORE CRITICO GFPGAN: {type(e).__name__}: {e}\n")
                _GFPGAN = False
    return _GFPGAN or None

def _free_vram():
    global _SDXL
    if TORCH_AVAILABLE:
        _SDXL = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

warnings.simplefilter("ignore")

PNG_SIG = b"\x89PNG\r\n\x1a\n"
HEIF_BRANDS = (b"heic", b"heix", b"hevc", b"hevx", b"heim", b"heis", b"mif1", b"msf1", b"avif")
EXT_TO_KIND = {".jpg": "jpeg", ".jpeg": "jpeg", ".jpe": "jpeg", ".png": "png", ".gif": "gif",
               ".bmp": "bmp", ".tif": "tiff", ".tiff": "tiff", ".webp": "webp"}
KIND_TO_EXT = {"jpeg": ".jpg", "png": ".png", "gif": ".gif", "bmp": ".bmp",
               "tiff": ".tif", "webp": ".webp"}
LABEL = {"ok": "Integra", "riparata": "Riparata", "ricostruita": "Ricostruita (approssimata)",
         "parziale": "Parziale", "fallita": "Non recuperabile", "saltata": "Saltata"}
PLURAL = {"ok": "integre", "riparata": "riparate", "ricostruita": "ricostruite",
          "parziale": "parziali", "fallita": "non recuperabili", "saltata": "saltate"}

@dataclass
class Result:
    status: str
    note: str = ""
    out: Path | None = None
    details: list = field(default_factory=list)

@dataclass
class Options:
    repair_binary: bool = True
    reconstruct: bool = True
    force_repair: bool = False
    reference_path: Path | None = None

# --------------------------------------------------------------------------- utilità

def sniff(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"): return "jpeg"
    if data.startswith(PNG_SIG): return "png"
    if data[:4] == b"GIF8": return "gif"
    if data[:2] == b"BM": return "bmp"
    if data[:4] in (b"II*\x00", b"MM\x00*"): return "tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP": return "webp"
    if data[4:8] == b"ftyp" and data[8:12] in HEIF_BRANDS: return "heif"
    return None

def decode(data: bytes) -> tuple[Image.Image | None, bool]:
    for lenient in (False, True):
        ImageFile.LOAD_TRUNCATED_IMAGES = lenient
        try:
            im = Image.open(io.BytesIO(data))
            im.load()
            return im, not lenient
        except Exception:
            continue
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False
    return None, False

def estimate_recovered(im: Image.Image) -> float:
    rgb = im.convert("RGB")
    first_col = rgb.crop((0, 0, 1, rgb.height)).resize(rgb.size)
    varying = ImageChops.difference(rgb, first_col).convert("L").point(lambda v: 255 if v > 6 else 0)
    rows = varying.getprojection()[1]
    top = next((i for i, v in enumerate(rows) if v), len(rows))
    bottom = next((i for i, v in enumerate(reversed(rows)) if v), len(rows))
    return 1 - max(top, bottom) / len(rows)

def save_image(im: Image.Image, out: Path, high_quality=False) -> None:
    extra = {k: im.info[k] for k in ("exif", "icc_profile") if im.info.get(k)}
    if out.suffix.lower() in (".jpg", ".jpeg"):
        if im.mode not in ("L", "RGB"): im = im.convert("RGB")
        q = 98 if high_quality else 95
        sub = 0 if high_quality else 2
        try: im.save(out, "JPEG", quality=q, subsampling=sub, **extra)
        except Exception: im.save(out, "JPEG", quality=q, subsampling=sub)
    else:
        if im.mode not in ("1", "L", "LA", "P", "RGB", "RGBA"): im = im.convert("RGB")
        im.save(out, "PNG")

def write_output(out: Path, payload: bytes | Image.Image, high_quality=False) -> Path:
    if isinstance(payload, Image.Image) and out.suffix.lower() not in (".jpg", ".jpeg"):
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, Image.Image): save_image(payload, out, high_quality)
    else: out.write_bytes(payload)
    return out

# --------------------------------------------------------------------------- A) danni al file: JPEG

def _is_marker(b: int) -> bool:
    return 0xE0 <= b <= 0xEF or b in (0xC0, 0xC1, 0xC2, 0xC4, 0xDB, 0xDA, 0xDD, 0xFE)

def jpeg_heads(data: bytes) -> list[tuple[bytes, str]]:
    if data.startswith(b"\xff\xd8\xff"): return [(data, "")]
    heads = []
    if len(data) > 4 and _is_marker(data[3]):
        heads.append((b"\xff\xd8\xff" + data[3:], "ripristinata l'intestazione (SOI)"))
    i = data.find(b"\xff\xd8\xff")
    if i > 0: heads.append((data[i:], f"tolti {i} byte di spazzatura iniziale"))
    return heads

def jpeg_tail(data: bytes) -> tuple[bytes, bool]:
    data = data.rstrip(b"\x00")
    if data.endswith(b"\xff\xd9"): return data, False
    return data + b"\xff\xd9", True

def carve_jpegs(data: bytes, limit: int = 50) -> Image.Image | None:
    best, area, pos = None, 0, 0
    for _ in range(limit):
        i = data.find(b"\xff\xd8\xff", pos)
        if i < 0: break
        pos = i + 3
        im, _complete = decode(data[i:])
        if im is not None and im.width * im.height > area:
            best, area = im, im.width * im.height
    return best

# --------------------------------------------------------------------------- A) danni al file: PNG
_ADAM7 = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2))
_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
def _chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body))
def _raw_size(w: int, h: int, bits_per_pixel: int, interlaced: bool) -> int:
    if not interlaced: return h * (1 + (w * bits_per_pixel + 7) // 8)
    total = 0
    for x0, y0, dx, dy in _ADAM7:
        pw, ph = (w - x0 + dx - 1) // dx, (h - y0 + dy - 1) // dy
        if pw > 0 and ph > 0: total += ph * (1 + (pw * bits_per_pixel + 7) // 8)
    return total
def _inflate_partial(stream: bytes) -> tuple[bytes, bool]:
    d = zlib.decompressobj()
    out = bytearray()
    for i in range(0, len(stream), 4096):
        piece = stream[i:i + 4096]
        backup = d.copy()
        try: out += d.decompress(piece)
        except zlib.error:
            for byte in piece:
                try: out += backup.decompress(bytes([byte]))
                except zlib.error: break
            return bytes(out), False
    return bytes(out), True
def rebuild_png(data: bytes) -> tuple[bytes, float, bool] | None:
    if not data.startswith(PNG_SIG):
        if data[12:16] == b"IHDR": data = PNG_SIG + data[8:]
        else:
            i = data.find(PNG_SIG)
            if i < 0: return None
            data = data[i:]
    pos, ihdr, idat, others = 8, None, bytearray(), []
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        if not ctype.isalpha(): break
        if ctype == b"IHDR": length = 13
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IDAT": idat += body
        elif ctype == b"IEND": break
        elif ctype == b"IHDR": ihdr = body
        elif len(body) == length: others.append((ctype, body))
        pos += 12 + length
    if ihdr is None or len(ihdr) < 13 or not idat: return None
    w, h, depth, color, _comp, _filt, interlace = struct.unpack(">IIBBBBB", ihdr[:13])
    if color not in _CHANNELS or depth not in (1, 2, 4, 8, 16) or not (0 < w < 2**31 and 0 < h < 2**31): return None
    expected = _raw_size(w, h, _CHANNELS[color] * depth, bool(interlace))
    if expected > 1_500_000_000: return None
    raw, intact = _inflate_partial(bytes(idat))
    frac = min(1.0, len(raw) / expected)
    raw = raw[:expected].ljust(expected, b"\x00")
    png = PNG_SIG + _chunk(b"IHDR", ihdr[:13])
    png += b"".join(_chunk(t, b) for t, b in others)
    png += _chunk(b"IDAT", zlib.compress(raw, 6)) + _chunk(b"IEND", b"")
    return png, frac, intact

# --------------------------------------------------------------------------- A2) riparazione del binario

Seg = namedtuple("Seg", "marker start end bad")
SOF_MARKERS = (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF)
_MARK = re.compile(rb"\xff[\xc0-\xc7\xc9-\xcf\xd8-\xfe]")
_ECS_END = re.compile(rb"\xff+[\xc4\xd9\xda\xdb\xdc\xdd\xe0-\xef\xfe]")
_RST = re.compile(rb"\xff[\xd0-\xd7]")
_SOF_DIMS = re.compile(rb"\xff[\xc0-\xc2].{2}\x08(.{2})(.{2})[\x01\x03\x04]", re.S)
STD_DHT = bytes.fromhex(
    "ffc4001f0000010501010101010100000000000000000102030405060708090a0bffc400b510"
    "0002010303020403050504040000017d01020300041105122131410613516107227114328191"
    "a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a434445"
    "464748494a535455565758595a636465666768696a737475767778797a838485868788898a92"
    "939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4"
    "d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1f2f3f4f5f6f7f8f9faffc4001f0100030101010101"
    "010101010000000000000102030405060708090a0bffc400b511000201020404030407050404"
    "00010277000102031104052131061241510761711322328108144291a1b1c109233352f01562"
    "72d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758"
    "595a636465666768696a737475767778797a82838485868788898a92939495969798999aa2a3"
    "a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5"
    "e6e7e8e9eaf2f3f4f5f6f7f8f9fa"
)
_oracle_lock = threading.Lock()
_oracle_state: dict = {}

def undo_transport_damage(data: bytes) -> tuple[bytes, list[str]]:
    notes: list[str] = []
    if sniff(data) is None:
        body = data[3:] if data.startswith(b"\xef\xbb\xbf") else data
        for enc in ("latin-1", "cp1252"):
            try: fixed = body.decode("utf-8").encode(enc)
            except (UnicodeDecodeError, UnicodeEncodeError): continue
            if any(sniff(c) in EXT_TO_KIND.values() for c in (fixed, fixed.replace(b"\r\n", b"\n"))):
                data = fixed
                notes.append("annullata la ricodifica di testo UTF-8")
                break
    lf, crlf = data.count(b"\n"), data.count(b"\r\n")
    if lf >= 8 and crlf >= 0.9 * lf:
        fixed = data.replace(b"\r\n", b"\n")
        if sniff(fixed) in EXT_TO_KIND.values() and decode(fixed)[0] is not None:
            data = fixed
            notes.append("annullata la conversione degli a capo LF→CRLF")
    return data, notes

def _libjpeg_check(data: bytes) -> tuple[bool, str]:
    sys.stderr.flush()
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile()
    try:
        os.dup2(tmp.fileno(), 2)
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    tmp.seek(0)
    text = tmp.read().decode("latin-1", "replace").strip()
    tmp.close()
    return img is not None, text

def _oracle_works() -> bool:
    if "ok" not in _oracle_state:
        ok = False
        try:
            buf = io.BytesIO()
            Image.effect_noise((96, 96), 60).convert("RGB").save(buf, "JPEG", quality=85)
            good = buf.getvalue()
            mid = len(good) // 2
            damaged = [good[:mid] + bytes(200) + good[mid + 200:], good[:mid] + good[mid + 1:],
                       good[:mid] + bytes(600) + good[mid + 600:]]
            with _oracle_lock:
                clean_ok, clean_msg = _libjpeg_check(good)
                ok = clean_ok and not clean_msg and any(_libjpeg_check(d)[1] for d in damaged)
        except Exception:
            ok = False
        _oracle_state["ok"] = ok
    return _oracle_state["ok"]

def jpeg_damaged(data: bytes) -> bool | None:
    if cv2 is None or not _oracle_works(): return None
    try:
        with _oracle_lock: decoded, text = _libjpeg_check(data)
    except Exception: return None
    low = text.lower()
    return (not decoded) or any(k in low for k in ("corrupt", "premature", "invalid", "bogus", "bad "))

def _declared_area(data: bytes) -> int:
    best = 0
    for m in _SOF_DIMS.finditer(data):
        best = max(best, int.from_bytes(m.group(1), "big") * int.from_bytes(m.group(2), "big"))
    return best

def _scan_end(data: bytes, start: int, baseline: bool) -> int:
    n = len(data)
    if not baseline:
        e = _ECS_END.search(data, start)
        return e.end() - 2 if e else n
    last, pos = -1, start
    while True:
        p = data.find(b"\xff\xd9", pos)
        if p < 0: return last if last >= 0 else n
        after = data[p + 2:p + 18]
        if not after.strip(b"\x00\xff") or after[:2] == b"\xff\xd8": return p
        last, pos = p, p + 1

def jpeg_layout(data: bytes, forced: dict | None = None):
    n = len(data)
    forced = forced or {}
    start = data.find(b"\xff\xd8")
    if start < 0: return None, ["manca l'inizio del file (SOI)"]
    problems = [f"{start} byte di spazzatura prima dell'inizio"] if start else []
    segs: list[Seg] = []
    i, ended = start + 2, False
    while i < n - 1:
        if data[i] != 0xFF:
            nxt = _MARK.search(data, i)
            if nxt is None:
                problems.append("dati illeggibili in fondo al file")
                i = n
                break
            problems.append(f"{nxt.start() - i} byte di spazzatura tra i segmenti")
            i = nxt.start()
            continue
        m = data[i + 1]
        if m == 0xFF:
            i += 1
            continue
        if m == 0xD9:
            segs.append(Seg(0xD9, i, i + 2, False))
            i += 2
            ended = True
            break
        if m == 0xD8:
            problems.append("inizio file (SOI) ripetuto")
            i += 2
            continue
        if m in (0x00, 0x01, 0xC8) or 0xD0 <= m <= 0xD7:
            problems.append("marcatore fuori posto")
            i += 2
            continue
        if i + 5 > n:
            problems.append("ultimo segmento troncato")
            i = n
            break
        length = (data[i + 2] << 8) | data[i + 3]
        end = resume = i + 2 + length
        if m == 0xDA:
            bad = not (1 <= data[i + 4] <= 4 and length == 6 + 2 * data[i + 4] and end <= n)
        else:
            bad = not (length >= 2 and end <= n and (end == n or data[end] == 0xFF))
            if bad and length >= 2 and end < n and i not in forced:
                pad = _MARK.search(data, end, min(n, end + 8))
                if pad: bad, resume = False, pad.start()
        if bad:
            if i in forced: end = forced[i]
            elif m == 0xDA and 1 <= data[i + 4] <= 4 and i + 10 + 2 * data[i + 4] <= n:
                end = i + 8 + 2 * data[i + 4]
            else:
                nxt = _MARK.search(data, i + 4)
                end = nxt.start() if nxt else n
            resume = end
            problems.append(f"lunghezza errata nel segmento FF{m:02X}")
        segs.append(Seg(m, i, end, bad))
        i = resume
        if m == 0xDA:
            j = _scan_end(data, i, any(sg.marker in (0xC0, 0xC1) for sg in segs))
            segs.append(Seg(None, i, j, False))
            i = j
    if not ended: problems.append("manca la fine del file (EOI)")
    return segs, problems

def jpeg_rebuild(data: bytes, segs, inject_dht: bool = False) -> bytes:
    out = bytearray(b"\xff\xd8")
    seen_dht, end_of_image = False, None
    for s in segs:
        if s.marker == 0xD9:
            end_of_image = s.end
            continue
        if s.marker == 0xC4: seen_dht = True
        if s.marker == 0xDA and inject_dht and not seen_dht:
            out += STD_DHT
            seen_dht = True
        chunk = data[s.start:s.end]
        if s.marker is not None and s.bad:
            body = len(chunk) - 2
            if body > 65535:
                if 0xE0 <= s.marker <= 0xEF or s.marker == 0xFE: continue
                raise ValueError(f"segmento FF{s.marker:02X} troppo lungo")
            chunk = chunk[:2] + struct.pack(">H", body) + chunk[4:]
        out += chunk
    out += b"\xff\xd9"
    if end_of_image is not None: out += data[end_of_image:]
    return bytes(out)

def _build_lookup(bits, vals):
    table = [0] * 65536
    code = k = 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            base, span = code << (16 - length), 1 << (16 - length)
            if k >= len(vals) or base + span > 65536: return None
            table[base:base + span] = [(length << 8) | vals[k]] * span
            code += 1
            k += 1
        code <<= 1
    return table

def baseline_params(data: bytes, segs):
    sof = sos = None
    dht, qt, dri = {}, {}, 0
    for s in segs:
        if s.marker is None or s.marker == 0xD9: continue
        p, m = data[s.start + 4:s.end], s.marker
        if m in (0xC0, 0xC1):
            if sof is not None: return None
            sof = p
        elif m in SOF_MARKERS: return None
        elif m == 0xC4:
            i = 0
            while i + 17 <= len(p):
                bits = list(p[i + 1:i + 17])
                dht[(p[i] >> 4, p[i] & 15)] = (bits, p[i + 17:i + 17 + sum(bits)])
                i += 17 + sum(bits)
        elif m == 0xDB:
            i = 0
            while i < len(p):
                step = 2 if p[i] >> 4 else 1
                if i + 1 + 64 * step > len(p): break
                qt[p[i] & 15] = [int.from_bytes(p[i + 1 + j * step:i + 1 + (j + 1) * step], "big") for j in range(64)]
                i += 1 + 64 * step
        elif m == 0xDD and len(p) >= 2: dri = (p[0] << 8) | p[1]
        elif m == 0xDA:
            if sos is not None: return None
            sos = p
    if sof is None or sos is None or len(sof) < 6 or sof[0] != 8: return None
    h, w, nc = (sof[1] << 8) | sof[2], (sof[3] << 8) | sof[4], sof[5]
    if not (h and w and nc in (1, 3)) or len(sof) < 6 + 3 * nc: return None
    comps = {sof[6 + 3 * k]: (sof[7 + 3 * k] >> 4, sof[7 + 3 * k] & 15, sof[8 + 3 * k]) for k in range(nc)}
    ns = sos[0]
    if ns != nc or len(sos) < 4 + 2 * ns or tuple(sos[1 + 2 * ns:4 + 2 * ns]) != (0, 63, 0): return None
    luts, blocks = {}, []
    for key in dht: luts[key] = _build_lookup(*dht[key])
    for ci in range(ns):
        cid, tdta = sos[1 + 2 * ci], sos[2 + 2 * ci]
        if cid not in comps or 0 in comps[cid][:2]: return None
        dct, act = luts.get((0, tdta >> 4)), luts.get((1, tdta & 15))
        if dct is None or act is None: return None
        q = qt.get(comps[ci][2]) if False else (qt.get(comps[cid][2]) or [1] * 64)
        blocks += [(dct, act, ci, q, 1024 // max(q[0], 1) + 2)] * (comps[cid][0] * comps[cid][1] if ns > 1 else 1)
    
    hmax = max(c[0] for c in comps.values())
    vmax = max(c[1] for c in comps.values())
    if ns > 1:
        mx = -(-w // (8 * hmax))
        my = -(-h // (8 * vmax))
    else:
        mx, my = -(-w // 8), -(-h // 8)
    return {"mcus": mx * my, "blocks": blocks, "ri": dri, "w": w, "h": h, "nc": nc, "mx": mx, "my": my, "hmax": hmax, "vmax": vmax}

def _decode_interval(buf: bytes, n_mcus: int, blocks, dc_jump_luma: int = 700, dc_jump_chroma: int = 350):
    nbits = len(buf) * 8
    padded = buf + bytes(3)
    words = [(padded[i] << 16) | (padded[i + 1] << 8) | padded[i + 2] for i in range(len(buf) + 1)]
    words.extend([0] * 300)
    pos, pred, starts = 0, [0, 0, 0], []
    prev_pred = None
    first_dc = None
    for m in range(n_mcus):
        starts.append(pos)
        prev_pred = tuple(pred)
        for dct, act, ci, q, lim in blocks:
            v = dct[(words[pos >> 3] >> (8 - (pos & 7))) & 0xFFFF]
            if not v: return m, starts, "codice Huffman non valido", pos, first_dc, tuple(pred)
            pos += v >> 8
            s = v & 255
            if s > 11: return m, starts, "categoria DC non valida", pos, first_dc, tuple(pred)
            if s:
                bits = (words[pos >> 3] >> (24 - (pos & 7) - s)) & ((1 << s) - 1)
                pos += s
                if bits < (1 << (s - 1)): bits -= (1 << s) - 1
                pred[ci] += bits
                if not -lim <= pred[ci] <= lim: return m, starts, "valore DC fuori scala", pos, first_dc, tuple(pred)
            k = 1
            while k < 64:
                v = act[(words[pos >> 3] >> (8 - (pos & 7))) & 0xFFFF]
                if not v: return m, starts, "codice Huffman non valido", pos, first_dc, tuple(pred)
                pos += v >> 8
                rs = v & 255
                s = rs & 15
                if s:
                    k += rs >> 4
                    if k > 63 or s > 10: return m, starts, "coefficiente AC fuori posto", pos, first_dc, tuple(pred)
                    if (1 << (s - 1)) * q[k] > 2600: return m, starts, "coefficiente AC implausibile", pos, first_dc, tuple(pred)
                    pos += s
                    k += 1
                elif rs == 0xF0: k += 16
                elif rs == 0: break
                else: return m, starts, "simbolo AC non valido", pos, first_dc, tuple(pred)
            if pos > nbits: return m, starts, "dati insufficienti", pos, first_dc, tuple(pred)
        if m == 0: first_dc = tuple(pred)
        else:
            for ci in range(3):
                limit = dc_jump_luma if ci == 0 else dc_jump_chroma
                if abs(pred[ci] - prev_pred[ci]) > limit:
                    return m, starts, "salto DC sospetto", pos, first_dc, tuple(pred)
    left = nbits - pos
    if left > 7 and not (left <= 15 and buf[-1] == 0xFF): return n_mcus, starts, "dati in eccesso", pos, first_dc, tuple(pred)
    return n_mcus, starts, None, pos, first_dc, tuple(pred)

def _split_intervals(raw: bytes):
    marks = [(m.start(), m.end(), raw[m.start() + 1] - 0xD0) for m in _RST.finditer(raw)]
    parts, merged = {}, set()
    k, start = 0, 0
    for i, (a, b, num) in enumerate(marks):
        nxt = marks[i + 1][2] if i + 1 < len(marks) else None
        if num == k % 8: step = 1
        elif nxt == k % 8: continue
        elif num == (k + 1) % 8 and (nxt is None or nxt == (num + 1) % 8):
            step = 2
            merged.add(k)
        else: continue
        parts[k] = raw[start:a]
        k, start = k + step, b
    parts[k] = raw[start:]
    return parts, merged

def check_baseline(data: bytes, segs, params, report=None):
    ecs = next(s for s in segs if s.marker is None)
    ri, total = params["ri"], params["mcus"]
    raw = data[ecs.start:ecs.end]
    parts, merged = _split_intervals(raw) if ri else ({0: raw}, set())
    expected = -(-total // ri) if ri else 1

    def mcus_in(k): return (ri if k < expected - 1 else total - ri * (expected - 1)) if ri else total

    results, carry = [], None
    for k in range(expected):
        n_mcu = mcus_in(k)
        if carry is not None: buf, carry = carry, None
        else: buf = parts.get(k, b"").replace(b"\xff\x00", b"\xff")
        res = _decode_interval(buf, n_mcu, params["blocks"])
        if k in merged and k + 1 < expected and res[2] in (None, "dati in eccesso"):
            cut = (res[3] + 7) // 8
            rest = buf[cut:]
            scored = [(_decode_interval(rest[skip:], mcus_in(k + 1), params["blocks"])[0], rest[skip:]) for skip in (2, 0, 1)]
            carry, buf = max(scored, key=lambda t: t[0])[1], buf[:cut]
            res = _decode_interval(buf, n_mcu, params["blocks"])
        results.append((*res, buf, n_mcu))
        if report and k % 4 == 0: report(k / expected, "controllo i dati dell'immagine")
    return results

def _cut_bits(buf: bytes, end_bit: int) -> bytes:
    nbytes, rem = divmod(end_bit, 8)
    if rem == 0: return buf[:nbytes]
    return buf[:nbytes] + bytes([buf[nbytes] | (0xFF >> rem)])

def repair_baseline(data: bytes, segs, params, results, margin: int = 4):
    ecs_i = next(i for i, s in enumerate(segs) if s.marker is None)
    out = bytearray(b"\xff\xd8")
    for s in segs[:ecs_i]: out += data[s.start:s.end]
    kept = 0
    prev_last_dc = None
    bad_intervals = []
    for k, (n_ok, starts, reason, pos, first_dc, last_dc, buf, n_mcu) in enumerate(results):
        if k: out += bytes([0xFF, 0xD0 + (k - 1) % 8])
        dc_incoherent = False
        if reason is not None and prev_last_dc is not None and first_dc is not None and n_ok > 0:
            for ci in range(3):
                if abs(prev_last_dc[ci] - first_dc[ci]) > 400:
                    dc_incoherent = True
                    break
        if reason is None:
            kept += n_mcu
            piece = buf
        elif reason == "dati in eccesso":
            kept += n_mcu
            piece = _cut_bits(buf, pos)
        elif dc_incoherent:
            piece = b""
            bad_intervals.append((k, n_mcu, 0, n_mcu))
        else:
            keep = max(0, n_ok - margin)
            kept += keep
            piece = _cut_bits(buf, starts[keep] if keep < len(starts) else pos)
            bad_intervals.append((k, n_mcu, keep, n_mcu))
        out += piece.replace(b"\xff", b"\xff\x00")
        if reason is None or not dc_incoherent: prev_last_dc = last_dc
        else: prev_last_dc = None
    out += b"\xff\xd9"
    return bytes(out), kept / params["mcus"], bad_intervals

# --------------------------------------------------------------------------- B) danni ai pixel

def _find_visual_glitches_safe(a: np.ndarray, force: bool = False) -> np.ndarray:
    """Trova i difetti visivi (aree piatte corrotte, buchi grigi/neri, colori fluo) senza falsi positivi."""
    h, w, _ = a.shape
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(a, cv2.COLOR_RGB2HSV)

    # Varianza locale calcolata su finestra 15x15
    blur = cv2.blur(gray, (15, 15))
    sq_blur = cv2.blur(gray**2, (15, 15))
    variance = np.maximum(0, sq_blur - blur**2)

    # 1. Colori sentinella tipici di corruzione o riempimenti di decoder (libjpeg, OpenCV)
    # Grigio piatto 128 (default di riempimento libjpeg sui dati mancanti)
    is_gray = (np.abs(a[:, :, 0].astype(int) - 128) < 10) & \
              (np.abs(a[:, :, 1].astype(int) - 128) < 10) & \
              (np.abs(a[:, :, 2].astype(int) - 128) < 10)
    # Nero profondo di riempimento
    is_black = (a[:, :, 0] < 6) & (a[:, :, 1] < 6) & (a[:, :, 2] < 6)
    # Colori neon/fluorescenti innaturali (artefatti di sfasamento bitstream DCT)
    is_neon = (hsv[:, :, 1] > 220) & (hsv[:, :, 2] > 220)

    suspicious_color = is_gray | is_black | is_neon
    if force:
        is_white = (a[:, :, 0] > 248) & (a[:, :, 1] > 248) & (a[:, :, 2] > 248)
        suspicious_color = suspicious_color | is_white

    # Difetto: colore sospetto combinato con varianza piatta
    damage = (suspicious_color & (variance < 5.0)).astype(np.uint8) * 255

    # 2. Pulizia morfologica per togliere rumore sparso
    damage = cv2.morphologyEx(damage, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(damage, connectivity=8)
    filtered = np.zeros_like(damage)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        if area > 100 or (max(bw, bh) > 30 and min(bw, bh) <= 8):
            filtered[labels == i] = 255

    # 3. DILATAZIONE GENEROSA (19x19 ellittica, raggio ~9px):
    # La finestra 15x15 del calcolo varianza perde i primi ~7px dal bordo del danno.
    # Dilatando con raggio 9px superiamo il bordo ed entriamo nel tessuto sano,
    # garantendo che NESSUN pixel grigio o corrotto rimanga non mascherato.
    dilated = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)))
    return dilated

def _structural_mask_from_intervals(im: Image.Image, bad_intervals, params):
    w, h = im.size
    mask = np.zeros((h, w), dtype=np.uint8)
    if not bad_intervals: return mask
    
    hmax = params.get("hmax", 2)
    vmax = params.get("vmax", 2)
    mcu_w = 8 * hmax
    mcu_h = 8 * vmax
    
    ri = max(1, params.get("ri", 1))
    mcus_per_row = (w + mcu_w - 1) // mcu_w
    
    for k, total_mcus, start_bad, end_bad in bad_intervals:
        idx_start = k * ri + start_bad
        idx_end = k * ri + end_bad - 1
        
        for mcu_idx in range(idx_start, min(idx_end + 1, params["mcus"])):
            r = mcu_idx // mcus_per_row
            c = mcu_idx % mcus_per_row
            y1 = r * mcu_h
            y2 = min(h, y1 + mcu_h)
            x1 = c * mcu_w
            x2 = min(w, x1 + mcu_w)
            mask[y1:y2, x1:x2] = 255
        
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

def blend_masked(orig: Image.Image, new: Image.Image, mask_np: np.ndarray, feather: int = 15) -> Image.Image:
    """Fonde il nuovo contenuto sopra l'originale preservando i pixel sani non mascherati."""
    k = feather if feather % 2 == 1 else feather + 1
    m = cv2.GaussianBlur(mask_np.astype(np.float32) / 255.0, (k, k), 0)[..., None]
    o = np.asarray(orig.convert("RGB"), np.float32)
    n = np.asarray(new.convert("RGB").resize(orig.size, Image.LANCZOS), np.float32)
    res = o * (1.0 - m) + n * m
    return Image.fromarray(np.clip(res, 0, 255).astype(np.uint8))

def reconstruct_pixels(im: Image.Image, structural_mask: np.ndarray | None = None, report=None, force=False, visual=True, ref_path=None):
    if cv2 is None: return None
    rep = report or (lambda frac, msg: None)
    
    alpha = im.getchannel("A") if "A" in im.getbands() else None
    im = im.convert("RGB")
    rgb = np.array(im)
    h, w = rgb.shape[:2]
    
    rep(0.10, "analizzo l'immagine alla ricerca di difetti visivi e strutturali")
    
    mask_combined = np.zeros((h, w), dtype=np.uint8)
    if structural_mask is not None:
        mask_combined = np.maximum(mask_combined, structural_mask)
        
    if visual:
        vis_mask = _find_visual_glitches_safe(rgb, force=force)
        mask_combined = np.maximum(mask_combined, vis_mask)

    if mask_combined.max() == 0:
        return None

    total_hole_ratio = np.count_nonzero(mask_combined) / mask_combined.size
    
    error_log = []
    engine = "OpenCV"
    res = None

    rep(0.30, "ricostruzione globale con AI (LaMa)...")
    lama = _get_lama()
    
    if lama is not None:
        try:
            # Ricostruzione globale a contesto completo: LaMa riceve l'intera immagine
            # con TUTTI i difetti mascherati contemporaneamente, evitando artefatti tra blocchi
            max_dim = 2048
            orig_w, orig_h = im.size
            if max(orig_w, orig_h) > max_dim:
                scale = max_dim / max(orig_w, orig_h)
                nw = max(8, int(orig_w * scale // 8) * 8)
                nh = max(8, int(orig_h * scale // 8) * 8)
                img_feed = im.resize((nw, nh), Image.LANCZOS)
                mask_feed = Image.fromarray(mask_combined).resize((nw, nh), Image.NEAREST)
            else:
                nw = max(8, (orig_w // 8) * 8)
                nh = max(8, (orig_h // 8) * 8)
                img_feed = im.resize((nw, nh), Image.LANCZOS)
                mask_feed = Image.fromarray(mask_combined).resize((nw, nh), Image.NEAREST)

            rep(0.50, f"inferenza LaMa GPU su risoluzione {nw}x{nh}...")
            rec = lama(img_feed, mask_feed)
            
            if (nw, nh) != (orig_w, orig_h):
                rec = rec.resize((orig_w, orig_h), Image.LANCZOS)
            
            # Fusione morbida: i pixel originali sani non mascherati rimangono intatti al 100%
            res = blend_masked(im, rec, mask_combined, feather=15)
            engine = f"LaMa (GPU {DEVICE.upper()})"
        except Exception as e:
            error_log.append(f"LaMa fallito: {e}")
            traceback.print_exc()

    if res is None:
        rep(0.60, "fallback inpainting OpenCV...")
        repaired = cv2.inpaint(rgb, mask_combined, 5, cv2.INPAINT_TELEA)
        res = blend_masked(im, Image.fromarray(repaired), mask_combined, feather=11)
        engine = "OpenCV (fallback)"

    # Ripristino volti con GFPGAN: solo per i volti che intersecano zone danneggiate
    gfpgan = _get_gfpgan()
    if gfpgan is not None:
        try:
            bgr = cv2.cvtColor(np.array(res.convert("RGB")), cv2.COLOR_RGB2BGR)
            _, _, restored = gfpgan.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True)
            
            face_damaged = False
            if hasattr(gfpgan, 'face_helper') and hasattr(gfpgan.face_helper, 'det_faces') and gfpgan.face_helper.det_faces:
                for det in gfpgan.face_helper.det_faces:
                    x1, y1, x2, y2 = [int(v) for v in det[:4]]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    if mask_combined[y1:y2, x1:x2].any():
                        face_damaged = True
                        break
                        
            if face_damaged and restored is not None:
                rep(0.90, "ottimizzo la risoluzione dei volti danneggiati (GFPGAN)")
                # GFPGAN con paste_back=True fonde già i volti coi landmark e contorni morbidi
                res = Image.fromarray(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
                engine += " + GFPGAN"
        except Exception as e:
            error_log.append(f"GFPGAN fallito: {e}")
    
    if alpha is not None: res.putalpha(alpha)
    
    return res, float(total_hole_ratio), engine, mask_combined, error_log

# --------------------------------------------------------------------------- orchestrazione

def repair_structure(data: bytes, kind: str | None, im0: Image.Image | None, complete0: bool):
    if complete0: return "ok", data, im0, []
    if kind == "jpeg":
        readable = None
        for head, head_note in jpeg_heads(data):
            fixed, truncated = jpeg_tail(head)
            steps = [s for s in (head_note, "mancava la fine del file (EOI)" if truncated else "") if s]
            im, complete = decode(fixed)
            if im is None: continue
            if complete:
                frac = estimate_recovered(im) if truncated else 1.0
                if frac < 0.99: return "parziale", fixed, im, steps + [f"file troncato: recuperato circa {frac:.0%}"]
                return "riparata", fixed, im, steps
            readable = readable or (im, steps)
        if readable:
            im, steps = readable
            return "parziale", im, im, steps + ["ricodificata la parte leggibile"]
    elif kind == "png":
        rebuilt = rebuild_png(data)
        if rebuilt:
            blob, frac, intact = rebuilt
            im, _ = decode(blob)
            if im is not None:
                if frac >= 0.999 and intact: return "riparata", blob, im, ["ricostruiti i blocchi (chunk) del PNG"]
                if frac >= 0.999: return "parziale", blob, im, ["dati compressi danneggiati"]
                return "parziale", blob, im, [f"recuperato circa {frac:.0%}"]
    elif im0 is not None:
        return "parziale", im0, im0, [f"decodificata solo in parte: circa {estimate_recovered(im0):.0%}"]
    im = carve_jpegs(data)
    if im is not None: return "parziale", im, im, [f"recuperato JPEG incorporato ({im.width}x{im.height})"]
    return None

def process(path: Path, outdir: Path, opts: Options | None = None, report=None) -> Result:
    opts = opts or Options()
    rep = report or (lambda frac, msg: None)
    rep(0.03, "leggo il file")
    data = path.read_bytes()
    if not data: return Result("fallita", "file vuoto")
    
    original = data
    details = [f"{path.name}: {len(data)} byte"]
    notes: list[str] = []

    rep(0.05, "controllo il binario del file")
    structural_mask = None
    
    if opts.repair_binary:
        try:
            data, transport_notes = undo_transport_damage(data)
            notes.extend(transport_notes)
            kind = sniff(data)
            if kind == "jpeg":
                segs, problems = jpeg_layout(data)
                if segs:
                    sof = next((s.marker for s in segs if s.marker in SOF_MARKERS), None)
                    if sof in (0xC0, 0xC1):
                        params = baseline_params(data, segs)
                        if params:
                            results = check_baseline(data, segs, params, lambda f, m: rep(0.05 + 0.15 * f, m))
                            if any(r[2] for r in results):
                                data, binary_recovered, bad_intervals = repair_baseline(data, segs, params, results)
                                im_temp, _ = decode(data)
                                if im_temp:
                                    structural_mask = _structural_mask_from_intervals(im_temp, bad_intervals, params)
                                notes.append(f"binario riparato: {binary_recovered:.0%} recuperato, {len(bad_intervals)} intervalli corretti")
        except Exception as e:
            details.append(f"controllo binario fallito: {e}")

    def done(status: str, note: str, out: Path | None = None) -> Result:
        return Result(status, note, out, details)

    real = sniff(data)
    if real == "heif": return done("saltata", "è un HEIC/HEIF/AVIF")
    declared = EXT_TO_KIND.get(path.suffix.lower())
    kind = real or declared
    out = outdir / f"{path.stem}_riparata{KIND_TO_EXT.get(kind, path.suffix)}"

    rep(0.2, "analizzo il file")
    im0, complete0 = decode(data)
    rep(0.25, "controllo la struttura")
    fixed = repair_structure(data, kind, im0, complete0)
    if fixed is None: return done("fallita", "nessun dato immagine recuperabile")
    status, payload, image, step_notes = fixed
    notes.extend(step_notes)

    if status == "ok" and data != original:
        status = "riparata"
        payload = data
        notes.append("riparato il binario del file")

    structural_any = structural_mask is not None and structural_mask.any()
    evidence = structural_any or data != original or status != "ok" or opts.force_repair

    if opts.reconstruct and cv2 is not None and image is not None:
        rec = reconstruct_pixels(image, structural_mask, lambda frac, msg: rep(0.3 + 0.6 * frac, msg), 
                                 force=opts.force_repair, visual=True, ref_path=opts.reference_path)
        if rec is not None:
            res, share, engine, mask_combined, error_log = rec
            details.extend(error_log)
            
            for key in ("exif", "icc_profile"):
                if image.info.get(key): res.info[key] = image.info[key]
            
            notes.append(f"ricostruito circa il {share:.0%} dell'immagine (motore: {engine})")
            details.append(f"ricostruzione: {share:.1%}, motore: {engine}, GPU: {DEVICE}")
            rep(0.95, "salvo il risultato")
            return done("ricostruita", "; ".join(notes), write_output(out, res, high_quality=True))

    if status == "ok":
        if not opts.repair_binary and not opts.reconstruct: return done("ok", "struttura a posto")
        hint = ""
        if not opts.reconstruct: hint = " (ricostruzione disattivata)"
        elif cv2 is None: hint = " (opencv non installato)"
        elif opts.force_repair: hint = " (forzatura attiva, ma immagine pulita)"
        return done("ok", ("; ".join(notes) if notes else "nessun difetto trovato") + hint)
    
    rep(0.95, "salvo il risultato")
    return done(status, "; ".join(notes), write_output(out, payload))

def run_batch(files, opts: Options, emit, cancel: threading.Event | None = None):
    counts: dict[str, int] = {}
    n = len(files)
    for i, (key, path, outdir) in enumerate(files):
        if cancel is not None and cancel.is_set(): return counts, True
        def report(frac, msg, i=i, name=path.name):
            emit(("progress", (i + frac) / n, f"{i + 1}/{n} · {name} — {msg}"))
        report(0.0, "inizio")
        try: res = process(path, outdir, opts, report)
        except Exception as e:
            res = Result("fallita", f"errore imprevisto: {type(e).__name__}: {e}", None, traceback.format_exc().splitlines()[-8:])
        counts[res.status] = counts.get(res.status, 0) + 1
        emit(("file", key, res))
        emit(("progress", (i + 1) / n, f"{i + 1}/{n} completate"))
    
    _free_vram()
    return counts, False

def write_report(outdir: Path, items) -> Path | None:
    try:
        lines = [f"RiparaFoto - rapporto del {datetime.now():%d/%m/%Y %H:%M}", ""]
        for path, res in items:
            lines += [str(path), f"  esito: {res.status} - {res.note}"] + [f"  {d}" for d in res.details] + [""]
        outdir.mkdir(parents=True, exist_ok=True)
        target = outdir / "rapporto_riparafoto.txt"
        target.write_text("\n".join(lines), encoding="utf-8")
        return target
    except Exception: return None

def iter_files(target: Path, recursive: bool, skip: Path):
    if target.is_file():
        yield target, Path()
        return
    for p in sorted(target.glob("**/*" if recursive else "*")):
        if p.is_file() and p.suffix.lower() in EXT_TO_KIND and skip not in p.resolve().parents:
            yield p, p.parent.relative_to(target)

# --------------------------------------------------------------------------- CLI & GUI

def run_gui() -> None:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ImportError: sys.exit("Tkinter non è installato. Su Linux: sudo apt install python3-tk")

    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception: pass

    filetypes = [("Immagini", " ".join(f"*{e} *{e.upper()}" for e in EXT_TO_KIND)), ("Tutti i file", "*.*")]

    class App:
        def __init__(self, root):
            self.root = root
            root.title("RiparaFoto (LaMa + GFPGAN)")
            root.geometry("860x700")
            root.minsize(700, 620)
            self.items: dict[str, Path] = {}
            self.counter = 0
            self.running = False
            self.written = 0
            self.reports: list = []
            self.results: dict = {}
            self.cancel = threading.Event()
            self.q: queue.Queue = queue.Queue()
            self.dest = tk.StringVar()
            self.binary = tk.BooleanVar(value=True)
            self.recon = tk.BooleanVar(value=cv2 is not None)
            self.force = tk.BooleanVar(value=False)
            self.status = tk.StringVar(value=f"Pronto. Motore AI: {DEVICE.upper()}. Aggiungi le foto.")
            self._build()
            self.dest.trace_add("write", lambda *_: self._update_buttons())
            self._update_buttons()

        def _build(self):
            main = ttk.Frame(self.root, padding=12)
            main.pack(fill="both", expand=True)
            box1 = ttk.LabelFrame(main, text=" 1. Foto da riparare ", padding=8)
            box1.pack(fill="both", expand=True)
            bar = ttk.Frame(box1)
            bar.pack(fill="x", pady=(0, 6))
            self.edit_buttons = [
                ttk.Button(bar, text="Aggiungi foto…", command=self.add_files),
                ttk.Button(bar, text="Aggiungi cartella…", command=self.add_folder),
                ttk.Button(bar, text="Rimuovi selezionate", command=self.remove_selected),
                ttk.Button(bar, text="Svuota elenco", command=self.clear),
            ]
            for b in self.edit_buttons: b.pack(side="left", padx=(0, 6))

            table = ttk.Frame(box1)
            table.pack(fill="both", expand=True)
            self.tree = ttk.Treeview(table, columns=("file", "esito"), show="headings", selectmode="extended", height=9)
            self.tree.heading("file", text="Foto")
            self.tree.heading("esito", text="Esito")
            self.tree.column("file", width=430, anchor="w")
            self.tree.column("esito", width=330, anchor="w")
            scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
            self.tree.configure(yscrollcommand=scroll.set)
            self.tree.pack(side="left", fill="both", expand=True)
            scroll.pack(side="left", fill="y")
            self.tree.bind("<Delete>", lambda _e: self.remove_selected())
            self.tree.bind("<Double-1>", self.show_details)

            box2 = ttk.LabelFrame(main, text=" 2. Cartella dove salvare le foto riparate ", padding=8)
            box2.pack(fill="x", pady=(10, 0))
            self.dest_entry = ttk.Entry(box2, textvariable=self.dest)
            self.dest_entry.pack(side="left", fill="x", expand=True)
            browse = ttk.Button(box2, text="Sfoglia…", command=self.choose_dest)
            browse.pack(side="left", padx=(6, 0))
            self.edit_buttons.append(browse)

            self.chk_bin = ttk.Checkbutton(main, variable=self.binary, text="Controlla e ripara anche il binario del file")
            self.chk_bin.pack(anchor="w", pady=(10, 0))
            self.chk = ttk.Checkbutton(main, variable=self.recon, text="Ricostruisci anche strisce e zone perse")
            self.chk.pack(anchor="w", pady=(2, 0))
            self.chk_force = ttk.Checkbutton(main, variable=self.force, text="Forza riparazione (usa euristiche aggressive sui pixel)")
            self.chk_force.pack(anchor="w", pady=(2, 0))
            if cv2 is None:
                self.chk.state(["disabled"])
                self.chk_force.state(["disabled"])

            box3 = ttk.Frame(main)
            box3.pack(fill="x", pady=(12, 0))
            self.pbar = ttk.Progressbar(box3, maximum=1000, mode="determinate")
            self.pbar.pack(fill="x")
            ttk.Label(box3, textvariable=self.status, wraplength=780, justify="left").pack(anchor="w", pady=(4, 0))

            row = ttk.Frame(main)
            row.pack(fill="x", pady=(10, 0))
            self.btn_start = ttk.Button(row, text="Ripara le foto", command=self.start)
            self.btn_start.pack(side="left")
            self.btn_cancel = ttk.Button(row, text="Annulla", command=self.cancel_run)
            self.btn_cancel.pack(side="left", padx=6)
            self.btn_open = ttk.Button(row, text="Apri la cartella", command=self.open_dest)
            self.btn_open.pack(side="right")

        def _update_buttons(self):
            idle = not self.running
            for b in self.edit_buttons: b.config(state="normal" if idle else "disabled")
            self.dest_entry.config(state="normal" if idle else "disabled")
            self.chk_bin.state(["!disabled"] if idle else ["disabled"])
            if cv2 is not None: 
                self.chk.state(["!disabled"] if idle else ["disabled"])
                self.chk_force.state(["!disabled"] if idle else ["disabled"])
            self.btn_start.config(state="normal" if idle and self.items else "disabled")
            self.btn_cancel.config(state="disabled" if idle else "normal")
            d = self.dest.get().strip()
            self.btn_open.config(state="normal" if idle and d and Path(d).is_dir() else "disabled")

        def _add(self, paths):
            known = set(self.items.values())
            for p in paths:
                p = Path(p)
                if p in known: continue
                known.add(p)
                iid = f"f{self.counter}"
                self.counter += 1
                self.items[iid] = p
                self.tree.insert("", "end", iid=iid, values=(str(p), "In attesa"))
            if self.items and not self.dest.get().strip():
                self.dest.set(str(next(iter(self.items.values())).parent / "riparate"))
            self._update_buttons()

        def add_files(self):
            paths = filedialog.askopenfilenames(title="Scegli le foto", filetypes=filetypes)
            self._add(self.root.tk.splitlist(paths))

        def add_folder(self):
            d = filedialog.askdirectory(title="Scegli una cartella")
            if not d: return
            found = [p for p in sorted(Path(d).iterdir()) if p.is_file() and p.suffix.lower() in EXT_TO_KIND]
            if not found:
                messagebox.showinfo("RiparaFoto", "Nessuna immagine trovata.")
                return
            self._add(found)

        def remove_selected(self):
            if self.running: return
            for iid in self.tree.selection():
                self.items.pop(iid, None)
                self.tree.delete(iid)
            self._update_buttons()

        def clear(self):
            if self.running: return
            self.tree.delete(*self.tree.get_children())
            self.items.clear()
            self._update_buttons()

        def choose_dest(self):
            d = filedialog.askdirectory(title="Scegli dove salvare")
            if d: self.dest.set(d)

        def start(self):
            if self.running or not self.items: return
            dest = self.dest.get().strip()
            if not dest:
                messagebox.showinfo("RiparaFoto", "Scegli la cartella di destinazione.")
                return
            outdir = Path(dest)
            try: outdir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                messagebox.showerror("RiparaFoto", f"Errore cartella:\n{e}")
                return
            files = [(iid, p, outdir) for iid, p in self.items.items()]
            for iid, _p, _o in files: self.tree.set(iid, "esito", "In attesa")
            self.cancel.clear()
            self.written = 0
            self.reports = []
            self.results = {}
            self.running = True
            self.pbar["value"] = 0
            self.status.set("Avvio…")
            self._update_buttons()
            opts = Options(repair_binary=bool(self.binary.get()), reconstruct=bool(self.recon.get()) and cv2 is not None, force_repair=bool(self.force.get()))
            threading.Thread(target=self._work, args=(files, opts), daemon=True).start()
            self.root.after(100, self._poll)

        def show_details(self, event):
            iid = self.tree.identify_row(event.y)
            res = self.results.get(iid)
            if res is None: return
            body = f"{self.items.get(iid, '')}\n\n{LABEL[res.status]}\n{res.note}"
            if res.details: body += "\n\nDettagli:\n" + "\n".join(res.details[-14:])
            messagebox.showinfo("Dettagli", body)

        def cancel_run(self):
            self.cancel.set()
            self.status.set("Annullo…")

        def _work(self, files, opts):
            try: counts, cancelled = run_batch(files, opts, self.q.put, self.cancel)
            except Exception as e:
                self.q.put(("error", str(e)))
                return
            self.q.put(("done", counts, cancelled))

        def _poll(self):
            finished = False
            try:
                while True:
                    ev = self.q.get_nowait()
                    tag = ev[0]
                    if tag == "progress":
                        self.pbar["value"] = ev[1] * 1000
                        if not self.cancel.is_set(): self.status.set(ev[2])
                    elif tag == "file":
                        _tag, iid, res = ev
                        self.reports.append((self.items.get(iid), res))
                        if res.out: self.written += 1
                        self.results[iid] = res
                        if self.tree.exists(iid):
                            txt = LABEL[res.status] + (f" — {res.note}" if res.note else "")
                            self.tree.set(iid, "esito", txt if len(txt) <= 170 else txt[:167] + "… (doppio clic)")
                    elif tag == "done":
                        self._finish(ev[1], ev[2])
                        finished = True
                    elif tag == "error":
                        self._finish({}, False, error=ev[1])
                        finished = True
            except queue.Empty: pass
            if not finished: self.root.after(100, self._poll)

        def _finish(self, counts, cancelled, error=None):
            self.running = False
            self._update_buttons()
            if error:
                self.status.set("Errore: " + error)
                messagebox.showerror("RiparaFoto", "Errore:\n" + error)
                return
            summary = ", ".join(f"{n} {PLURAL[s]}" for s, n in counts.items()) or "nessuna foto"
            if not cancelled: self.pbar["value"] = 1000
            self.status.set(("Annullato. " if cancelled else "Fatto: ") + summary + ".")
            lines = [("Annullato: " if cancelled else "Fatto: ") + summary + "."]
            if self.written: lines.append(f"Ho salvato {self.written} file in:\n{self.dest.get()}")
            else: lines.append("Nessun file salvato.")
            if counts.get("ricostruita"):
                lines.append("Le foto ricostruite sono approssimate: controllale.")
            rapporto = write_report(Path(self.dest.get().strip()), [(p, r) for p, r in self.reports if p is not None])
            if rapporto: lines.append(f"Rapporto tecnico:\n{rapporto}")
            messagebox.showinfo("RiparaFoto", "\n\n".join(lines))

        def open_dest(self):
            d = self.dest.get().strip()
            if not d or not Path(d).is_dir(): return
            try:
                if sys.platform.startswith("win"): os.startfile(d)
                elif sys.platform == "darwin": subprocess.Popen(["open", d])
                else: subprocess.Popen(["xdg-open", d])
            except Exception as e: messagebox.showerror("RiparaFoto", f"Errore:\n{e}")

        def on_close(self):
            if self.running and not messagebox.askyesno("RiparaFoto", "Riparazione in corso. Chiudere?"): return
            self.cancel.set()
            self.root.destroy()

    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

def main() -> None:
    try: sys.stdout.reconfigure(errors="replace")
    except Exception: pass
    ap = argparse.ArgumentParser(description="Ripara foto corrotte.")
    ap.add_argument("percorso", type=Path, nargs="?", help="file o cartella")
    ap.add_argument("-o", "--output", type=Path, default=Path("riparate"), help="cartella di destinazione")
    ap.add_argument("-r", "--ricorsivo", action="store_true", help="includi sottocartelle")
    ap.add_argument("--senza-ricostruzione", action="store_true", help="non ricostruire pixel")
    ap.add_argument("--senza-binario", action="store_true", help="non riparare il binario")
    ap.add_argument("--forza", action="store_true", help="forza la ricostruzione visiva anche se non rilevano danni palesi")
    ap.add_argument("--riferimento", type=Path, help="cartella o file con una copia sana da usare come riferimento")
    ap.add_argument("--gui", action="store_true", help="apri la GUI")
    args = ap.parse_args()

    if args.gui or args.percorso is None:
        run_gui()
        return
    if not args.percorso.exists():
        sys.exit(f"Percorso non trovato: {args.percorso}")

    opts = Options(repair_binary=not args.senza_binario, reconstruct=not args.senza_ricostruzione, force_repair=args.forza, reference_path=args.riferimento)
    files = [(f, f, args.output / rel) for f, rel in iter_files(args.percorso, args.ricorsivo, args.output.resolve())]
    items = []

    def emit(ev):
        if ev[0] == "file":
            _tag, f, res = ev
            items.append((f, res))
            dest = f" -> {res.out}" if res.out else ""
            print(f"[{res.status.upper()}] {f.name}: {res.note}{dest}")

    counts, _cancelled = run_batch(files, opts, emit)
    summary = ", ".join(f"{n} {PLURAL[s]}" for s, n in counts.items())
    print(f"\nFatto: {summary or 'nessun file trovato'}.")
    if any(res.status != "ok" for _f, res in items):
        rapporto = write_report(args.output, items)
        if rapporto: print(f"Rapporto tecnico: {rapporto}")

if __name__ == "__main__":
    main()
