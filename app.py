import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Medición con visión artificial", layout="wide")

# ---------------- UTILIDADES ----------------

def pil_to_rgb(image):
    return np.array(image.convert("RGB"))

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def auto_otsu(gray):
    _, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    r1 = np.mean(th1 > 0)
    r2 = np.mean(th2 > 0)

    return th1 if abs(r1 - 0.1) < abs(r2 - 0.1) else th2

def clean_mask(mask, min_area=300):
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)

    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            clean[labels == i] = 255

    return clean

def get_contours(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def classify_objects(contours):
    fiber = None
    coin = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue

        x,y,w,h = cv2.boundingRect(c)
        aspect = max(w/h, h/w)

        peri = cv2.arcLength(c, True)
        circ = 4*np.pi*area/(peri*peri) if peri>0 else 0

        if circ > 0.7:
            coin = c
        elif aspect > 2:
            fiber = c

    return fiber, coin

def skeleton_length(mask):
    skel = cv2.ximgproc.thinning(mask) if hasattr(cv2, "ximgproc") else mask
    return np.sum(skel > 0)

# ---------------- UI ----------------

st.title("Medición de fibra con visión artificial")

uploaded = st.file_uploader("Carga una imagen", type=["jpg","png","jpeg"])

if uploaded is None:
    st.stop()

img = pil_to_rgb(Image.open(uploaded))

# ---------------- PROCESAMIENTO ----------------

gray = grayscale(img)
blur = cv2.GaussianBlur(gray, (5,5), 0)
th = auto_otsu(blur)

kernel = np.ones((3,3), np.uint8)

closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

clean = clean_mask(opened)

# máscara final
mask = clean

# overlay
overlay = img.copy()
cnts = get_contours(mask)
cv2.drawContours(overlay, cnts, -1, (0,255,0), 2)

# ---------------- VALIDACIÓN ----------------

st.subheader("Vista previa")
st.image(overlay)

decision = st.radio(
    "¿La detección es correcta?",
    ["Sí, continuar", "No, cargar otra imagen"]
)

if decision != "Sí, continuar":
    st.stop()

# ---------------- MEDICIÓN ----------------

fiber, coin = classify_objects(cnts)

if fiber is None or coin is None:
    st.error("No se detectó correctamente fibra o moneda")
    st.stop()

# pedir diámetro real
diam_mm = st.number_input("Ingrese diámetro real de la moneda (mm)", value=20.0)

# diámetro moneda en px
(x,y), radius = cv2.minEnclosingCircle(coin)
diam_px = radius * 2

# escala
px_per_mm = diam_px / diam_mm

# fibra
fiber_area = cv2.contourArea(fiber)
fiber_perimeter = cv2.arcLength(fiber, True)

# longitud (aprox skeleton)
fiber_mask = np.zeros_like(mask)
cv2.drawContours(fiber_mask, [fiber], -1, 255, -1)

length_px = skeleton_length(fiber_mask)

# diámetro promedio
diam_px_fiber = fiber_area / length_px if length_px > 0 else 0

# convertir a mm
length_mm = length_px / px_per_mm
diam_mm_fiber = diam_px_fiber / px_per_mm
perimeter_mm = fiber_perimeter / px_per_mm

# ---------------- RESULTADOS ----------------

st.subheader("Resultados")

st.write(f"Escala: {px_per_mm:.2f} px/mm")

st.write(f"Longitud fibra: {length_mm:.2f} mm")
st.write(f"Diámetro promedio fibra: {diam_mm_fiber:.2f} mm")
st.write(f"Perímetro fibra: {perimeter_mm:.2f} mm")

# mostrar máscara final en blanco
mask_white = cv2.bitwise_not(mask)
st.image(mask_white, caption="Máscara final")
