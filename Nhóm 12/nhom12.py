"""
NHOM 12 - NHAN DIEN KHUON MAT CO BAN BANG DAC TRUNG HINH HOC

Y tuong chinh:
  1) Tien xu ly anh: Grayscale + can bang Histogram.
  2) Xay dung Integral Image de tinh tong vung O(1).
  3) Phat hien ung vien khuon mat bang Haar Cascade (Viola-Jones):
       - Moc thang: haarcascade_frontalface_default.xml
       - Moc nghieng trai/phai: haarcascade_profileface.xml
  4) Xac nhan them bang dac trung hinh hoc co ban trong ROI:
       - Phat hien vung mat toi (nua tren), co doi xung ngang
       - Phat hien vung mieng (nua duoi) - khong bat buoc
       - Kiem tra ty le da (skin ratio) bang YCrCb
  5) Non-Maximum Suppression (NMS) loai bung trung lap.
  6) Tinh toan va bao cao dac trung hinh hoc day du.

Thu vien:
    pip install opencv-python numpy

Cach chay:
    python nhom12.py                           # hoi duong dan anh
    python nhom12.py --image anh.jpg           # xu ly 1 file anh
    python nhom12.py --image anh.jpg --show    # hien thi cua so
    python nhom12.py --image anh.jpg --out ket_qua.jpg
    python nhom12.py --demo                    # dung anh mau noi bo
    python nhom12.py --watch-folder ANH/       # theo doi thu muc
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────
# 1. CAU HINH THAM SO
# ──────────────────────────────────────────────────────────────

# --- Cascade Haar (co san trong OpenCV) ---
CASCADE_FACE    = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_FACE_ALT= cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
CASCADE_PROFILE = cv2.data.haarcascades + "haarcascade_profileface.xml"
CASCADE_EYE     = cv2.data.haarcascades + "haarcascade_eye.xml"
CASCADE_SMILE   = cv2.data.haarcascades + "haarcascade_smile.xml"

# --- Tham so Viola-Jones (mat nhin thang) ---
FACE_SCALE      = 1.08   # Buoc thu nho pyramid
FACE_NEIGHBORS  = 4      # So cua so tich cuc toi thieu (thap = bat duoc nhieu hon)
FACE_MIN_SIZE   = (30, 30)

# --- Tham so mat nghieng ---
PROFILE_SCALE     = 1.08
PROFILE_NEIGHBORS = 3
PROFILE_MIN_SIZE  = (30, 30)

# --- Tham so xac nhan mat/nuoc cuoi ---
EYE_SCALE     = 1.1
EYE_NEIGHBORS = 5
EYE_MIN_SIZE  = (15, 15)

SMILE_SCALE     = 1.6
SMILE_NEIGHBORS = 18
SMILE_MIN_SIZE  = (20, 20)

# --- Nguong NMS ---
NMS_IOU_THRESH = 0.35

# Mau ve ket qua (BGR)
COLOR_FACE   = (0, 200, 255)   # Vang cam
COLOR_EYE    = (80,  220,  80) # Xanh la
COLOR_MOUTH  = (220,  80, 220) # Tim
COLORS_MULTI = [
    (0,  200, 255),
    (0,  230, 100),
    (255, 100,  0),
    (200,   0, 255),
    (0,  170, 255),
]


# ──────────────────────────────────────────────────────────────
# 2. CAU TRUC DU LIEU KET QUA
# ──────────────────────────────────────────────────────────────

Rect = Tuple[int, int, int, int]   # (x, y, w, h)


@dataclass
class FaceCandidate:
    """Luu thong tin mot khuon mat ung vien."""
    rect  : Rect                           # Hop gioi han khuon mat
    score : float                          # Diem tin cay tong hop
    eyes  : List[Rect] = field(default_factory=list)   # Hop gioi han mat
    mouth : Optional[Rect] = None          # Hop gioi han nuoc cuoi
    skin_ratio : float = 0.0               # Ti le pixel da trong vung mat


# ──────────────────────────────────────────────────────────────
# 3. TIEN XU LY ANH
# ──────────────────────────────────────────────────────────────

def tien_xu_ly(anh_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUOC TIEN XU LY:
      (a) BGR -> Grayscale: loai bo thong tin mau, giu do sang.
      (b) Can bang Histogram (equalizeHist): tang tuong phan toan cuc.
          Giup phat hien on dinh hon trong anh toi hoac nguoc sang.
      (c) Lam tron nhe GaussianBlur (5x5): giam nhieu truoc Haar.

    Returns:
        gray_eq : Anh xam 1 kenh da can bang (dung cho Haar Cascade)
        gray_raw: Anh xam chua xu ly  (dung cho kiem tra ROI noi bo)
    """
    gray_raw = cv2.cvtColor(anh_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq  = cv2.equalizeHist(gray_raw)
    gray_eq  = cv2.GaussianBlur(gray_eq, (3, 3), 0)
    return gray_eq, gray_raw


# ──────────────────────────────────────────────────────────────
# 4. INTEGRAL IMAGE
# ──────────────────────────────────────────────────────────────

def tinh_integral_image(gray: np.ndarray) -> np.ndarray:
    """
    BUOC 2 - INTEGRAL IMAGE (Summed Area Table):
      II(x,y) = tong tat ca pixel I(i,j) voi i <= x, j <= y.

    Tong bat ky vung [x1,y1 -> x2,y2] chi can 4 phep tinh:
      S = II(x2,y2) - II(x1-1,y2) - II(x2,y1-1) + II(x1-1,y1-1)

    Do phuc tap: O(n) tien xu ly, O(1) moi truy van.
    OpenCV tu dung noi bo khi goi detectMultiScale().

    Returns:
        ii (np.ndarray): Integral Image co kich thuoc (H+1) x (W+1)
    """
    return cv2.integral(gray)


# ──────────────────────────────────────────────────────────────
# 5. KHOI TAO BAN PHAN LOAI HAAR CASCADE
# ──────────────────────────────────────────────────────────────

def khoi_tao_cascade() -> dict:
    """
    Nap cac bo phan loai Haar Cascade.

    Thuat toan Viola-Jones (2001):
      - Dac trung Haar: tinh chenh lech tong pixel giua vung sang
        va toi (mat toi hon go ma, song mui sang hon hai ben...).
      - AdaBoost Cascade: chuoi 25-38 tang loc, loai ~99.99% vung
        khong phai khuon mat rat som.
      - Image Pyramid: thu nho anh nhieu lan (scaleFactor) de phat
        hien khuon mat o moi kich thuoc (gan/xa camera).
      - Non-Maximum Suppression: hop nhat hop chong lap.
    """
    cascades = {}
    spec = {
        "face"   : CASCADE_FACE,
        "face_alt": CASCADE_FACE_ALT,
        "profile": CASCADE_PROFILE,
        "eye"    : CASCADE_EYE,
        "smile"  : CASCADE_SMILE,
    }
    for ten, duong_dan in spec.items():
        clf = cv2.CascadeClassifier(duong_dan)
        if not clf.empty():
            cascades[ten] = clf
            print(f"  [OK] cascade: {ten}")
        else:
            print(f"  [CANH BAO] Khong nap duoc cascade: {ten}")

    if "face" not in cascades:
        raise RuntimeError(
            "Khong tim thay cascade khuon mat!\n"
            "  Chay: pip install opencv-python"
        )
    return cascades


# ──────────────────────────────────────────────────────────────
# 6. PHAT HIEN KHUON MAT (HAAR MULTI-SCALE + PROFILE)
# ──────────────────────────────────────────────────────────────

def phat_hien_haar(gray_eq: np.ndarray, cascades: dict) -> List[Rect]:
    """
    BUOC 3a - PHAT HIEN KHUON MAT BANG HAAR CASCADE:
      - Mat nhin thang: dung cascade chinh + cascade alt2
        (2 cascade bo sung cho nhau, tang ty le bat duoc).
      - Mat nghieng trai: dung profile cascade.
      - Mat nghieng phai: lat anh ngang + profile cascade
        -> chuyen toa do nguoc lai.
      - Loai bo hop trung lap bang NMS nhe (IoU > 0.3).

    Args:
        gray_eq : Anh xam da can bang histogram
        cascades: Dict cac CascadeClassifier
    Returns:
        List[(x, y, w, h)]: Tat ca ung vien khuon mat
    """
    H, W = gray_eq.shape
    ket_qua: List[Rect] = []

    # ── Mat nhin thang (chinh) ──
    raw = cascades["face"].detectMultiScale(
        gray_eq,
        scaleFactor  = FACE_SCALE,
        minNeighbors = FACE_NEIGHBORS,
        minSize      = FACE_MIN_SIZE,
        flags        = cv2.CASCADE_SCALE_IMAGE,
    )
    if len(raw) > 0:
        for r in raw:
            ket_qua.append(tuple(int(v) for v in r))

    # ── Mat nhin thang (alt2 - bo sung) ──
    if "face_alt" in cascades:
        raw_alt = cascades["face_alt"].detectMultiScale(
            gray_eq,
            scaleFactor  = FACE_SCALE,
            minNeighbors = FACE_NEIGHBORS,
            minSize      = FACE_MIN_SIZE,
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )
        if len(raw_alt) > 0:
            for r in raw_alt:
                rect = tuple(int(v) for v in r)
                if not _trung_lap(rect, ket_qua, NMS_IOU_THRESH):
                    ket_qua.append(rect)

    # ── Mat nghieng (profile) ──
    if "profile" in cascades:
        # Nghieng trai
        raw_l = cascades["profile"].detectMultiScale(
            gray_eq,
            scaleFactor  = PROFILE_SCALE,
            minNeighbors = PROFILE_NEIGHBORS,
            minSize      = PROFILE_MIN_SIZE,
        )
        if len(raw_l) > 0:
            for r in raw_l:
                rect = tuple(int(v) for v in r)
                if not _trung_lap(rect, ket_qua, NMS_IOU_THRESH):
                    ket_qua.append(rect)

        # Nghieng phai (lat anh)
        anh_lat = cv2.flip(gray_eq, 1)
        raw_r   = cascades["profile"].detectMultiScale(
            anh_lat,
            scaleFactor  = PROFILE_SCALE,
            minNeighbors = PROFILE_NEIGHBORS,
            minSize      = PROFILE_MIN_SIZE,
        )
        if len(raw_r) > 0:
            for (x, y, w, h) in raw_r:
                rect = (int(W - x - w), int(y), int(w), int(h))
                if not _trung_lap(rect, ket_qua, NMS_IOU_THRESH):
                    ket_qua.append(rect)

    return ket_qua


def _iou(a: Rect, b: Rect) -> float:
    """Tinh IoU (Intersection over Union) giua 2 hop."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


def _chua_trong(nho: Rect, lon: Rect) -> bool:
    """Kiem tra hop nho co nam hoan toan ben trong hop lon khong."""
    sx, sy, sw, sh = nho
    lx, ly, lw, lh = lon
    return sx >= lx and sy >= ly and sx+sw <= lx+lw and sy+sh <= ly+lh


def _trung_lap(rect: Rect, danh_sach: List[Rect], nguong: float) -> bool:
    """Kiem tra rect co IoU > nguong HOAC nam hoan toan ben trong 1 hop khac."""
    for r in danh_sach:
        if _iou(rect, r) > nguong:
            return True
        # Loai neu rect nam hoan toan trong r (sub-region)
        if _chua_trong(rect, r):
            return True
    return False


# ──────────────────────────────────────────────────────────────
# 7. XAC NHAN BANG DAC TRUNG HINH HOC (ROI)
# ──────────────────────────────────────────────────────────────

def _tim_vung_toi(gray_roi: np.ndarray) -> List[Rect]:
    """
    Tim cac vung toi (mat, mieng) trong ROI bang Adaptive Threshold.

    Su dung ADAPTIVE_THRESH_GAUSSIAN_C thay vi nguong toan cuc
    de chiu duoc bien dong anh sang cuc bo.
    Loc theo: dien tich, ti le w/h, loai qua nho/to.
    """
    if gray_roi.size == 0:
        return []
    blur  = cv2.GaussianBlur(gray_roi, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        21, 6,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    rects: List[Rect] = []
    h, w  = gray_roi.shape
    S_roi = h * w

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area  = cw * ch
        if area < 0.001 * S_roi or area > 0.14 * S_roi:
            continue
        ratio = cw / max(ch, 1)
        if ratio < 0.4 or ratio > 4.5:
            continue
        rects.append((int(x), int(y), int(cw), int(ch)))
    return rects


def xac_nhan_mat(gray_roi: np.ndarray, cascades: dict) -> List[Rect]:
    """
    BUOC 4a - PHAT HIEN MAT TRONG ROI KHUON MAT:

    Chien luoc hai lop:
      Lop 1 (Cascade): Chay haarcascade_eye trong nua TREN ROI.
                       Nhanh, chinh xac nhat.
      Lop 2 (Hinh hoc): Neu cascade bat duoc < 2 mat, bo sung
                         bang _tim_vung_toi() tim cap vung toi
                         co doi xung ngang (dieu kien doi xung
                         la rang buoc hinh hoc cot loi).

    Dieu kien doi xung ngang:
      - Khoang cach ngang giua 2 mat: 12% - 78% chieu rong roi
      - Lech doc < 18% chieu cao roi
      - Ti le dien tich 2 mat: > 0.30

    Returns:
        List[(ex, ey, ew, eh)]: Toa do tuyet doi trong anh goc
                                (toi da 2 mat)
    """
    h, w = gray_roi.shape
    roi_tren = gray_roi[: h // 2, :]
    mat: List[Rect] = []

    # -- Lop 1: Cascade --
    if "eye" in cascades:
        raw = cascades["eye"].detectMultiScale(
            roi_tren,
            scaleFactor  = EYE_SCALE,
            minNeighbors = EYE_NEIGHBORS,
            minSize      = EYE_MIN_SIZE,
        )
        if len(raw) > 0:
            raw_sorted = sorted(raw, key=lambda e: e[2]*e[3], reverse=True)[:2]
            mat = [(int(ex), int(ey), int(ew), int(eh))
                   for (ex, ey, ew, eh) in raw_sorted]

    # -- Lop 2: Hinh hoc bo sung khi cascade bat < 2 mat --
    if len(mat) < 2:
        components = _tim_vung_toi(roi_tren)
        best_pair: List[Rect] = []
        best_score = -1.0

        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                a = components[i]
                b = components[j]
                ax, ay, aw, ah = a
                bx, by, bw, bh = b
                acx = ax + aw / 2.0
                bcx = bx + bw / 2.0
                acy = ay + ah / 2.0
                bcy = by + bh / 2.0

                # Dam bao a la mat trai
                if acx > bcx:
                    a, b = b, a
                    ax, ay, aw, ah = a
                    bx, by, bw, bh = b
                    acx, bcx = bcx, acx
                    acy, bcy = bcy, acy

                dist_ngang  = bcx - acx
                lech_doc    = abs(acy - bcy)
                ti_le_dt    = min(aw*ah, bw*bh) / max(aw*ah, bw*bh, 1)

                # Rang buoc hinh hoc doi xung
                if dist_ngang < 0.12 * w or dist_ngang > 0.78 * w:
                    continue
                if lech_doc > 0.18 * h:
                    continue
                if ti_le_dt < 0.30:
                    continue

                tam_giua = (acx + bcx) / 2.0
                doi_xung = 1.0 - abs(tam_giua - w / 2.0) / (w / 2.0)
                score    = ti_le_dt + doi_xung

                if score > best_score:
                    best_score = score
                    best_pair  = [a, b]

        if best_pair:
            mat = best_pair

    return mat


def xac_nhan_mieng(gray_roi: np.ndarray, cascades: dict) -> Optional[Rect]:
    """
    BUOC 4b - PHAT HIEN MIENG TRONG ROI KHUON MAT (nua duoi):

    Chien luoc hai lop tuong tu mat:
      Lop 1: Cascade smile trong nua DUOI ROI.
      Lop 2: Tim vung toi lon nhat gan trung tam nua duoi.

    Rang buoc hinh hoc:
      - Ti le w/h cua vung mieng: 0.8 - 5.0
      - Vi tri ngang: gan trung tam (+/- 35%)
      - Uu tien vung nam thap hon trong nua duoi

    Returns:
        (sx, sy, sw, sh) toa do tuyet doi, hoac None
    """
    h, w = gray_roi.shape
    roi_duoi = gray_roi[h // 2 :, :]
    offset_y = h // 2

    # -- Lop 1: Cascade --
    if "smile" in cascades:
        raw = cascades["smile"].detectMultiScale(
            roi_duoi,
            scaleFactor  = SMILE_SCALE,
            minNeighbors = SMILE_NEIGHBORS,
            minSize      = SMILE_MIN_SIZE,
        )
        if len(raw) > 0:
            raw_s = sorted(raw, key=lambda s: s[2]*s[3], reverse=True)
            sx, sy, sw, sh = raw_s[0]
            return (int(sx), int(sy + offset_y), int(sw), int(sh))

    # -- Lop 2: Hinh hoc bo sung --
    components = _tim_vung_toi(roi_duoi)
    best = None
    best_score = -1.0

    for x, y, cw, ch in components:
        ti_le = cw / max(ch, 1)
        if ti_le < 0.8 or ti_le > 5.0:
            continue
        cx = x + cw / 2.0
        vi_tri_ngang = abs(cx - w / 2.0) / (w / 2.0)
        if vi_tri_ngang > 0.35:
            continue
        vi_tri_doc = y / max(roi_duoi.shape[0], 1)
        score = ti_le * 0.4 + vi_tri_doc * 0.6 - vi_tri_ngang * 0.3
        if score > best_score:
            best_score = score
            best       = (int(x), int(y + offset_y), int(cw), int(ch))

    return best


def tinh_skin_ratio(anh_bgr: np.ndarray, rect: Rect) -> float:
    """
    BUOC 4c - XAC NHAN TY LE DA (SKIN RATIO):

    Phan vung da trong khong gian YCrCb:
      Nguong: Cr in [130, 180], Cb in [70, 140] (pho bien cho mau da nguoi)
    Dung de xac nhan them rang vung Haar bat duoc co nen la da nguoi,
    KHONG phai la phuong phap phat hien chinh.

    Returns:
        Ti le [0.0, 1.0] pixel da trong ROI
    """
    x, y, w, h = rect
    H, W = anh_bgr.shape[:2]
    x = max(0, x); y = max(0, y)
    w = min(w, W - x); h = min(h, H - y)
    if w <= 0 or h <= 0:
        return 0.0

    roi_bgr = anh_bgr[y:y+h, x:x+w]
    ycrcb   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)

    # Nguong YCrCb pho bien cho da nguoi
    lower = np.array([0,   130, 70],  dtype=np.uint8)
    upper = np.array([255, 180, 140], dtype=np.uint8)
    mask  = cv2.inRange(ycrcb, lower, upper)

    # Mo rong them bang HSV
    hsv   = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower_h = np.array([0, 15, 40],   dtype=np.uint8)
    upper_h = np.array([30, 255, 255], dtype=np.uint8)
    mask_h  = cv2.inRange(hsv, lower_h, upper_h)

    mask_final = cv2.bitwise_or(mask, mask_h)
    return float(np.count_nonzero(mask_final)) / (w * h)


# ──────────────────────────────────────────────────────────────
# 8. NON-MAXIMUM SUPPRESSION
# ──────────────────────────────────────────────────────────────

def nms(candidates: List[FaceCandidate], nguong_iou: float = NMS_IOU_THRESH) -> List[FaceCandidate]:
    """
    Non-Maximum Suppression:
    Sap xep giam theo score, giu cac hop khong trung lap qua nguong.
    """
    ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: List[FaceCandidate] = []
    for cand in ordered:
        trung = any(_iou(cand.rect, s.rect) > nguong_iou for s in kept)
        if not trung:
            kept.append(cand)
    return kept


# ──────────────────────────────────────────────────────────────
# 9. PIPELINE CHINH
# ──────────────────────────────────────────────────────────────

def nhan_dien_khuon_mat(anh_bgr: np.ndarray, cascades: dict) -> List[FaceCandidate]:
    """
    PIPELINE DAY DU:

      Buoc 1: Tien xu ly (Grayscale + Histogram EQ + Blur nhe)
      Buoc 2: Integral Image (OpenCV dung noi bo)
      Buoc 3: Phat hien ung vien Haar (mat thang + nghieng T/P)
      Buoc 4: Xac nhan dac trung ben trong moi ROI:
                4a. Phat hien mat (cascade + hinh hoc doi xung)
                4b. Phat hien mieng (cascade + hinh hoc trung tam)
                4c. Tinh ty le da (YCrCb skin ratio)
              Tinh diem tin cay tong hop:
                score = score_haar
                      + 2.0 * (so_mat >= 2)
                      + 0.5 * (so_mat == 1)
                      + 0.4 * (co_mieng)
                      + 1.0 * (skin_ratio > 0.25)
      Buoc 5: NMS loai bung trung lap

    Uu tien Haar lam phuong phap chinh, dac trung hinh hoc
    va skin ratio la xac nhan bo sung -> do chinh xac cao nhat.
    """
    gray_eq, gray_raw = tien_xu_ly(anh_bgr)
    tinh_integral_image(gray_eq)               # Integral Image

    # Buoc 3: Haar detection
    ung_vien_rects = phat_hien_haar(gray_eq, cascades)

    candidates: List[FaceCandidate] = []

    for rect in ung_vien_rects:
        x, y, w, h = rect
        H_img, W_img = anh_bgr.shape[:2]

        # Dam bao ROI trong anh
        x  = max(0, x);  y  = max(0, y)
        w  = min(w, W_img - x); h = min(h, H_img - y)
        if w < 10 or h < 10:
            continue
        rect = (x, y, w, h)

        roi_gray = gray_raw[y:y+h, x:x+w]

        # Buoc 4a: Xac nhan mat
        mat_local = xac_nhan_mat(roi_gray, cascades)
        mat_abs   = [(x+ex, y+ey, ew, eh) for (ex, ey, ew, eh) in mat_local]

        # Buoc 4b: Xac nhan mieng
        mieng_local = xac_nhan_mieng(roi_gray, cascades)
        mieng_abs   = None
        if mieng_local is not None:
            mx, my, mw, mh = mieng_local
            mieng_abs = (x+mx, y+my, mw, mh)

        # Buoc 4c: Skin ratio
        skin = tinh_skin_ratio(anh_bgr, rect)

        # Tinh diem tong hop
        score = 3.0   # Diem co ban (da bat duoc boi Haar = co co so)
        score += 2.0 if len(mat_abs) >= 2 else (0.5 if len(mat_abs) == 1 else 0.0)
        score += 0.4 if mieng_abs is not None else 0.0
        score += 1.0 if skin > 0.25 else (0.3 if skin > 0.10 else 0.0)

        candidates.append(FaceCandidate(
            rect       = rect,
            score      = score,
            eyes       = mat_abs,
            mouth      = mieng_abs,
            skin_ratio = skin,
        ))

    return nms(candidates)


# ──────────────────────────────────────────────────────────────
# 10. VE KET QUA LEN ANH
# ──────────────────────────────────────────────────────────────

def ve_ket_qua(anh_bgr: np.ndarray, faces: List[FaceCandidate]) -> np.ndarray:
    """
    Ve hop gioi han, goc trang tri, nhan va thong tin hinh hoc
    len ban sao cua anh goc.
    """
    out  = anh_bgr.copy()
    H, W = out.shape[:2]

    for i, face in enumerate(faces):
        x, y, w, h = face.rect
        mau = COLORS_MULTI[i % len(COLORS_MULTI)]

        # -- Hop chinh --
        cv2.rectangle(out, (x, y), (x+w, y+h), mau, 2)

        # -- Goc trang tri (bracket corners) --
        do_dai = max(14, w // 7)
        for (gx, gy, dx, dy) in [
            (x,     y,     1,  1),
            (x+w,   y,    -1,  1),
            (x,     y+h,   1, -1),
            (x+w,   y+h,  -1, -1),
        ]:
            cv2.line(out, (gx, gy), (gx + dx*do_dai, gy),  mau, 3)
            cv2.line(out, (gx, gy), (gx, gy + dy*do_dai),  mau, 3)

        # -- Diem tam --
        cx_f, cy_f = x + w//2, y + h//2
        cv2.circle(out, (cx_f, cy_f), 4, mau, -1)
        cv2.line(out, (cx_f-10, cy_f), (cx_f+10, cy_f), mau, 1)
        cv2.line(out, (cx_f, cy_f-10), (cx_f, cy_f+10), mau, 1)

        # -- Nhan phia tren hop --
        co_cuoi     = face.mouth is not None
        nhan        = f"Mat #{i+1}  {w}x{h}px" + ("  :)" if co_cuoi else "")
        (tw, th), _ = cv2.getTextSize(nhan, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        ny = max(y - 6, th + 6)
        cv2.rectangle(out, (x, ny-th-4), (x+tw+10, ny+2), mau, cv2.FILLED)
        cv2.putText(out, nhan, (x+5, ny-1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1, cv2.LINE_AA)

        # -- Vung mat --
        for (ex, ey, ew, eh) in face.eyes:
            cv2.rectangle(out, (ex, ey), (ex+ew, ey+eh), COLOR_EYE, 1)
            cv2.circle(out, (ex+ew//2, ey+eh//2), 2, COLOR_EYE, -1)

        # -- Vung mieng --
        if face.mouth:
            mx, my, mw, mh = face.mouth
            cv2.rectangle(out, (mx, my), (mx+mw, my+mh), COLOR_MOUTH, 1)

        # -- Thong tin hinh hoc ben duoi hop --
        dien_tich = w * h
        ti_le_wh  = w / h
        pct_x     = cx_f / W * 100
        pct_y     = cy_f / H * 100
        dong_tt   = [
            f"Vi tri   : ({x}, {y})",
            f"Kich thuoc: {w} x {h} px",
            f"Dien tich: {dien_tich:,} px2",
            f"Ti le W/H: {ti_le_wh:.2f}  Tam: ({pct_x:.0f}%, {pct_y:.0f}%)",
            f"Da (skin): {face.skin_ratio*100:.0f}%  Score: {face.score:.2f}",
        ]
        for j, dong in enumerate(dong_tt):
            vy = y + h + 14 + j * 13
            if vy + 10 < H:
                cv2.putText(out, dong, (x+2, vy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, mau, 1, cv2.LINE_AA)

    # -- Thanh tieu de --
    cv2.rectangle(out, (0, 0), (W, 30), (20, 20, 20), cv2.FILLED)
    tieu_de = (f"NHOM 12 | {len(faces)} khuon mat | "
               f"{W}x{H}px | Viola-Jones + Hinh hoc")
    cv2.putText(out, tieu_de, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

    return out


# ──────────────────────────────────────────────────────────────
# 11. BAO CAO PHAN TICH HINH HOC
# ──────────────────────────────────────────────────────────────

def in_bao_cao(faces: List[FaceCandidate], anh_shape, ten_file: str = "") -> None:
    """In bang bao cao day du ra console."""
    H, W  = anh_shape[:2]
    sep72 = "=" * 72

    print(f"\n{sep72}")
    print("  BAO CAO NHAN DIEN KHUON MAT - DAC TRUNG HINH HOC - NHOM 12")
    if ten_file:
        print(f"  File  : {ten_file}")
    print(f"  Anh   : {W} x {H} px")
    print(f"  Ket qua: {len(faces)} khuon mat phat hien duoc")
    print("-" * 72)

    if not faces:
        print("  -> Khong tim thay khuon mat nao.")
        print("  Goi y: kiem tra anh sang, goc chup, thu --neighbors 2")
        print(sep72)
        return

    print(f"  {'#':>3}  {'bbox (x,y,w,h)':>18}  {'W/H':>5}  "
          f"{'Dien tich':>11}  {'Tam anh':>13}  "
          f"{'Mat':>4}  {'Mieng':>6}  {'Da%':>5}  {'Score':>6}")
    print("-" * 72)

    for i, face in enumerate(faces):
        x, y, w, h = face.rect
        cx       = x + w // 2
        cy       = y + h // 2
        pct_x    = cx / W * 100
        pct_y    = cy / H * 100
        co_mieng = "Co" if face.mouth else "Khong"
        so_mat   = len(face.eyes)

        print(f"  {i+1:>3}  ({x:>4},{y:>4},{w:>3},{h:>3})"
              f"  {w/h:>5.2f}"
              f"  {w*h:>11,}"
              f"  ({pct_x:>4.0f}%,{pct_y:>4.0f}%)"
              f"  {so_mat:>4}"
              f"  {co_mieng:>6}"
              f"  {face.skin_ratio*100:>4.0f}%"
              f"  {face.score:>6.2f}")

    print("-" * 72)
    print(f"  Tong mat   : {len(faces)}")
    print(f"  Tong mat(2): {sum(1 for f in faces if len(f.eyes) >= 2)}")
    print(f"  Co nuoc cuoi: {sum(1 for f in faces if f.mouth)}")
    if len(faces) > 1:
        dt  = [f.rect[2]*f.rect[3] for f in faces]
        idx = dt.index(max(dt))
        print(f"  Khuon mat lon nhat: #{idx+1} ({faces[idx].rect[2]}x{faces[idx].rect[3]} px)")
    print(sep72)


# ──────────────────────────────────────────────────────────────
# 12. DOC / GHI ANH (HO TRO UNICODE)
# ──────────────────────────────────────────────────────────────

def doc_anh(duong_dan: str) -> Optional[np.ndarray]:
    """Doc anh an toan voi duong dan Unicode tren moi OS."""
    for _ in range(3):
        try:
            data = np.fromfile(duong_dan, dtype=np.uint8)
            if data.size == 0:
                time.sleep(0.2)
                continue
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except OSError:
            pass
        time.sleep(0.2)
    return None


def ghi_anh(duong_dan: str, anh: np.ndarray) -> bool:
    """Ghi anh an toan voi duong dan Unicode."""
    ext = os.path.splitext(duong_dan)[1].lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp"}:
        ext = ".jpg"
    ok, buf = cv2.imencode(ext, anh)
    if not ok:
        return False
    try:
        buf.tofile(duong_dan)
        return True
    except OSError:
        return False


# ──────────────────────────────────────────────────────────────
# 13. XU LY MOT FILE ANH
# ──────────────────────────────────────────────────────────────

def xu_ly_mot_anh(
    duong_dan: str,
    cascades : dict,
    output_path : Optional[str] = None,
    show    : bool = False,
) -> int:
    """
    Xu ly 1 file anh day du:
      - Doc anh
      - Chay pipeline nhan dien
      - In bao cao
      - Luu ket qua
      - Hien thi cua so (tuy chon)

    Returns:
        So khuon mat tim duoc, hoac -1 neu loi
    """
    anh = doc_anh(duong_dan)
    if anh is None:
        print(f"  [LOI] Khong doc duoc: {duong_dan}")
        return -1

    H, W  = anh.shape[:2]
    ten   = os.path.basename(duong_dan)
    print(f"\n  +-- File : {ten}  ({W}x{H} px)")

    faces   = nhan_dien_khuon_mat(anh, cascades)
    ket_qua = ve_ket_qua(anh, faces)

    print(f"  |   Ket qua : {len(faces)} khuon mat")
    in_bao_cao(faces, anh.shape, ten)

    # HIEN THI TRUOC
    if True:
        try:
            cv2.imshow(f"NHOM 12 – {ten}", ket_qua)
            print("  [HIEN THI] Nhan phim bat ky de dong...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print("  [HIEN THI] Khong co GUI.")

    # SAU DO MOI LUU
    if output_path is None:
        goc, duoi = os.path.splitext(duong_dan)
        duoi_luu = duoi if duoi.lower() in (".jpg", ".jpeg", ".png", ".bmp") else ".jpg"
        output_path = goc + "_nhom12_ketqua" + duoi_luu

    if ghi_anh(output_path, ket_qua):
        print(f"  [LUU] -> {output_path}")
    else:
        print(f"  [LOI] Khong ghi duoc: {output_path}")
    return len(faces)

# ──────────────────────────────────────────────────────────────
# 14. CHE DO THEO DOI THU MUC
# ──────────────────────────────────────────────────────────────

def _la_file_anh(ten: str) -> bool:
    return os.path.splitext(ten)[1].lower() in \
        {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _file_san_sang(duong_dan: str) -> bool:
    """Kiem tra file da copy xong va co kich thuoc on dinh truoc khi doc."""
    try:
        size1 = os.path.getsize(duong_dan)
    except OSError:
        return False
    if size1 <= 0:
        return False
    time.sleep(0.25)
    try:
        size2 = os.path.getsize(duong_dan)
    except OSError:
        return False
    return size1 == size2 and size2 > 0


def theo_doi_thu_muc(
    thu_muc_vao : str,
    thu_muc_ra  : str,
    cascades    : dict,
    chu_ky      : float = 2.0,
) -> None:
    """Lien tuc quet thu muc, xu ly anh moi tu dong."""
    if not os.path.isabs(thu_muc_vao):
        thu_muc_vao = os.path.join(os.path.dirname(os.path.abspath(__file__)), thu_muc_vao)
    if not os.path.isabs(thu_muc_ra):
        thu_muc_ra = os.path.join(os.path.dirname(os.path.abspath(__file__)), thu_muc_ra)

    os.makedirs(thu_muc_vao, exist_ok=True)
    os.makedirs(thu_muc_ra, exist_ok=True)
    da_xu_ly: set = set()

    print(f"\n[THEO DOI] Thu muc nhan: {thu_muc_vao}")
    print(f"           Thu muc ra  : {thu_muc_ra}")
    print(f"           Chu ky quet : {chu_ky:.1f}s")
    print("           Dat anh vao thu muc nhan anh de xu ly tu dong.")
    print("           Nhan Ctrl+C de dung.\n")

    while True:
        for ten_file in sorted(os.listdir(thu_muc_vao)):
            if not _la_file_anh(ten_file):
                continue
            src = os.path.join(thu_muc_vao, ten_file)
            if src in da_xu_ly:
                continue
            if not _file_san_sang(src):
                continue

            goc, duoi = os.path.splitext(ten_file)
            dst = os.path.join(thu_muc_ra, goc + "_nhom12_ketqua" + (duoi or ".jpg"))
            print(f"\n[MOI] {src}")
            so_mat = xu_ly_mot_anh(src, cascades, output_path=dst, show=False)
            if so_mat >= 0:
                da_xu_ly.add(src)

        time.sleep(chu_ky)


# ──────────────────────────────────────────────────────────────
# 15. TAO ANH DEMO NOI BO
# ──────────────────────────────────────────────────────────────

def tao_anh_demo() -> str:
    """Ve 2 khuon mat gian luoc, dam bao dac trung Haar co the bat."""
    H, W   = 480, 720
    canvas = np.ones((H, W, 3), dtype=np.uint8)
    for r in range(H):
        v = int(162 + r * 0.04)
        canvas[r] = [v+10, v+5, v]

    def ve_mat(cx, cy, s=1.0):
        def p(v): return int(v * s)
        # Da
        cv2.ellipse(canvas, (cx, cy), (p(82), p(102)), 0, 0, 360, (190, 165, 135), -1)
        cv2.ellipse(canvas, (cx, cy), (p(82), p(102)), 0, 0, 360, (150, 120, 90),  2)
        # Tran sang
        cv2.ellipse(canvas, (cx, cy-p(45)), (p(72), p(52)), 0, 180, 360, (210, 185, 158), -1)
        # Toc
        cv2.ellipse(canvas, (cx, cy-p(52)), (p(87), p(68)), 0, 180, 360, (45, 28, 12), -1)
        cv2.rectangle(canvas, (cx-p(87), cy-p(100)), (cx+p(87), cy-p(52)), (45,28,12), -1)
        # Long may toi
        for bx, sg in [(cx-p(27), -1), (cx+p(27), 1)]:
            cv2.ellipse(canvas, (bx, cy-p(29)), (p(19),p(5)), sg*12, 0, 180, (55,35,18), -1)
        # Mat (vung toi -> dac trung mat-go ma)
        for ex in [cx-p(27), cx+p(27)]:
            cv2.ellipse(canvas, (ex, cy-p(16)), (p(19),p(13)), 0, 0, 360, (255,255,255), -1)
            cv2.circle( canvas, (ex, cy-p(16)),  p(9),          (35,20,10), -1)
            cv2.circle( canvas, (ex-p(3), cy-p(19)), p(3),       (255,255,255), -1)
        # Go ma sang
        for mx in [cx-p(50), cx+p(50)]:
            cv2.circle(canvas, (mx, cy+p(8)), p(18), (200,170,140), -1)
        # Song mui
        pts = np.array([[cx,cy-p(6)],[cx-p(10),cy+p(19)],[cx+p(10),cy+p(19)]], np.int32)
        cv2.polylines(canvas, [pts], True, (150,115,88), 1)
        # Moi (vung toi -> dac trung moi-cam)
        cv2.ellipse(canvas, (cx, cy+p(33)), (p(24),p(10)), 0, 0, 180, (145,75,72), -1)
        cv2.ellipse(canvas, (cx, cy+p(33)), (p(24),p(8)),  0, 0, 180, (160,85,80), -1)
        # Cam sang
        cv2.ellipse(canvas, (cx, cy+p(57)), (p(32),p(18)), 0, 0, 180, (195,168,142), -1)

    ve_mat(195, 250, s=1.0)
    ve_mat(510, 235, s=0.88)

    cv2.putText(canvas, "NHOM 12 – Anh demo (2 khuon mat)",
                (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (25,25,25), 2)
    cv2.putText(canvas, "Viola-Jones + Haar dac trung hinh hoc",
                (14, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (50,50,50), 1)

    duong_dan = "nhom12_anh_demo.jpg"
    cv2.imwrite(duong_dan, canvas)
    print(f"  [DEMO] Da tao anh mau: {duong_dan}")
    return duong_dan


# ──────────────────────────────────────────────────────────────
# 16. MAIN
# ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog        = "nhom12.py",
        description = (
            "NHOM 12 – Nhan dien khuon mat bang dac trung hinh hoc co ban\n"
            "Phuong phap: Viola-Jones Haar Cascade + Xac nhan hinh hoc\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Vi du:\n"
            "  python nhom12.py                          # hoi duong dan\n"
            "  python nhom12.py --image anh.jpg --show   # co cua so\n"
            "  python nhom12.py --demo                   # anh mau\n"
            "  python nhom12.py --watch-folder ANH/      # theo doi thu muc\n"
        ),
    )
    nhom = parser.add_mutually_exclusive_group()
    nhom.add_argument("--image",  "-i", type=str, metavar="FILE",
                      help="Duong dan file anh dau vao")
    nhom.add_argument("--demo",   "-d", action="store_true",
                      help="Tao va xu ly anh mau noi bo")
    nhom.add_argument("--watch-folder", "-w", type=str, metavar="DIR",
                      default=None,
                      help="Theo doi thu muc, tu dong xu ly anh moi (mac dinh: ANH/)")
    parser.add_argument("--out",    "-o",  type=str, default=None,
                        help="Duong dan file ket qua (mac dinh: ten_nhom12_ketqua.jpg)")
    parser.add_argument("--output-folder", type=str, default="ket_qua",
                        help="Thu muc luu ket qua khi dung --watch-folder")
    parser.add_argument("--show",  "-s",  action="store_true",
                        help="Hien thi cua so ket qua (can moi truong co GUI)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Chu ky quet (giay) khi dung --watch-folder")
    parser.add_argument("--scale",     type=float, default=FACE_SCALE,
                        metavar="SF",  help=f"scaleFactor (mac dinh {FACE_SCALE})")
    parser.add_argument("--neighbors", type=int,   default=FACE_NEIGHBORS,
                        metavar="MN",  help=f"minNeighbors (mac dinh {FACE_NEIGHBORS})")
    return parser.parse_args()

def main() -> None:
    print()
    print("+" + "=" * 62 + "+")
    print("|  NHOM 12 – NHAN DIEN KHUON MAT BANG DAC TRUNG HINH HOC      |")
    print("|  Xac dinh vung chua khuon mat trong anh                      |")
    print("|  Viola-Jones Haar Cascade  +  Xac nhan hinh hoc co ban       |")
    print(f"|  Thu vien : OpenCV {cv2.__version__:<43}|")
    print("+" + "=" * 62 + "+")

    args = parse_args()

    if args.watch_folder is None:
        args.watch_folder = "ẢNH"

    # Ap dung tham so tuy chinh
    global FACE_SCALE, FACE_NEIGHBORS
    FACE_SCALE     = args.scale
    FACE_NEIGHBORS = args.neighbors

    # Khoi tao cascade
    print("\n[KHOI TAO] Nap Haar Cascade classifiers...")
    cascades = khoi_tao_cascade()
    print(f"  -> Da nap: {', '.join(cascades.keys())}\n")

    # ── Che do theo doi thu muc ──
    if args.watch_folder is not None:
        try:
            theo_doi_thu_muc(
                args.watch_folder,
                args.output_folder,
                cascades,
                args.interval,
            )
        except KeyboardInterrupt:
            print("\n[DUNG] Da ket thuc theo doi.")
        return

    # ── Che do demo ──
    if args.demo:
        print("[CHE DO] Demo – tao anh mau noi bo\n")
        dp = tao_anh_demo()
        xu_ly_mot_anh(dp, cascades, output_path=args.out, show=args.show)
        print("\n[XONG] – Nhom 12\n")
        return

    # ── Che do file anh ──
    if args.image:
        if not os.path.isfile(args.image):
            print(f"[LOI] Khong tim thay file: {args.image}")
            return
        print(f"[CHE DO] Xu ly file anh: {args.image}\n")
        xu_ly_mot_anh(args.image, cascades, output_path=args.out, show=args.show)
        print("\n[XONG] – Nhom 12\n")
        return

    # ── Che do nhap tuong tac ──
    print("[CHE DO] Nhap tuong tac\n")
    print("  Nhap duong dan file anh can xu ly.")
    print("  Go 'demo'  de dung anh mau noi bo.")
    print("  Go 'thoat' de ket thuc.")
    print("-" * 50)

    while True:
        try:
            nhap = input("\n  Duong dan anh >> ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\n  Da thoat.")
            break

        if not nhap:
            continue
        if nhap.lower() in ("thoat", "quit", "exit", "q"):
            break
        if nhap.lower() in ("demo", "d"):
            dp = tao_anh_demo()
            xu_ly_mot_anh(dp, cascades, show=args.show)
            continue
        if not os.path.isfile(nhap):
            print(f"  [LOI] Khong tim thay: {nhap}")
            print("  Thu lai hoac go 'demo'.")
            continue

        xu_ly_mot_anh(nhap, cascades, show=args.show)

        tiep = input("\n  Xu ly anh khac? [Enter=Co / n=Khong]: ").strip().lower()
        if tiep in ("n", "no", "khong", "k"):
            break
        print("  Nhap duong dan anh tiep theo:")

    print("\n[XONG] Chuong trinh ket thuc. – Nhom 12\n")


if __name__ == "__main__":
    main()