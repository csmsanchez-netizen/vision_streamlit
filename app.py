
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.morphology import skeletonize


st.set_page_config(page_title="Evaluación, procesamiento y medición", layout="wide")


# -------------------------------------------------
# Utilidades
# -------------------------------------------------
def pil_to_rgb(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"))


def resize_keep_aspect(img: np.ndarray, max_width: int = 900) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    scale = max_width / w
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def grayscale(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)


def auto_crop_left(img_rgb: np.ndarray, crop_left_pct: int):
    if crop_left_pct <= 0:
        return img_rgb.copy(), 0
    h, w = img_rgb.shape[:2]
    crop_px = int(w * crop_left_pct / 100.0)
    cropped = img_rgb[:, crop_px:]
    return cropped, crop_px


def auto_otsu_foreground(gray_blur: np.ndarray):
    _, th_bin = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, th_inv = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ratio_bin = float(np.mean(th_bin > 0))
    ratio_inv = float(np.mean(th_inv > 0))

    target = 0.10
    score_bin = abs(ratio_bin - target)
    score_inv = abs(ratio_inv - target)

    if score_bin < score_inv:
        return th_bin, {"ratio": ratio_bin, "polarity": "THRESH_BINARY"}
    return th_inv, {"ratio": ratio_inv, "polarity": "THRESH_BINARY_INV"}


def remove_small_components(mask: np.ndarray, min_area: int = 300) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            clean[labels == i] = 255
    return clean


def remove_border_touching_components(mask: np.ndarray, min_area: int = 50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    removed = 0
    h, w = mask.shape

    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        touches_border = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)

        if touches_border and area >= min_area:
            removed += 1
            continue

        clean[labels == i] = 255

    return clean, removed


def component_metrics(mask: np.ndarray):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    metrics = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        comp = np.zeros_like(mask)
        comp[labels == i] = 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        area_cnt = cv2.contourArea(cnt)
        circularity = 4 * np.pi * area_cnt / (peri * peri) if peri > 0 else 0.0
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        metrics.append(
            {
                "label": i,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "circularity": float(circularity),
                "aspect_ratio": float(aspect_ratio),
            }
        )
    return metrics


def overlay_components(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = img_rgb.copy()
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, (0, 255, 0), 3)
    return overlay


def evaluate_photo_quality(gray: np.ndarray, initial_mask: np.ndarray, final_mask: np.ndarray):
    fg = gray[final_mask > 0]
    bg = gray[final_mask == 0]
    contrast = abs(float(np.mean(fg)) - float(np.mean(bg))) if len(fg) and len(bg) else 0.0

    h, w = initial_mask.shape
    border = np.zeros_like(initial_mask, dtype=bool)
    border_margin_h = max(5, int(0.03 * h))
    border_margin_w = max(5, int(0.03 * w))
    border[:border_margin_h, :] = True
    border[-border_margin_h:, :] = True
    border[:, :border_margin_w] = True
    border[:, -border_margin_w:] = True

    total_fg_initial = int(np.sum(initial_mask > 0))
    border_fg_initial = int(np.sum((initial_mask > 0) & border))
    border_noise_ratio = border_fg_initial / max(total_fg_initial, 1)

    comps = component_metrics(final_mask)
    component_count = len(comps)
    elongated_count = sum(1 for c in comps if c["aspect_ratio"] >= 2.5)
    circular_count = sum(1 for c in comps if c["circularity"] >= 0.65)

    reasons = []
    ok = True

    if contrast < 35:
        ok = False
        reasons.append("Contraste bajo entre objeto y fondo.")
    if component_count < 2:
        ok = False
        reasons.append("No se aislaron al menos dos objetos principales.")
    if elongated_count < 1:
        ok = False
        reasons.append("No se detectó claramente un objeto alargado como la fibra.")
    if circular_count < 1:
        ok = False
        reasons.append("No se detectó claramente un objeto circular como la referencia.")
    if border_noise_ratio > 0.35:
        ok = False
        reasons.append("Existe demasiado ruido o interferencia pegada a los bordes.")

    verdict = "APTA" if ok else "REPETIR FOTO"

    return {
        "verdict": verdict,
        "contrast": contrast,
        "border_noise_ratio": border_noise_ratio,
        "component_count": component_count,
        "elongated_count": elongated_count,
        "circular_count": circular_count,
        "reasons": reasons,
    }


def process_image(img_rgb: np.ndarray, min_area: int = 300, crop_left_pct: int = 0):
    cropped_rgb, crop_px = auto_crop_left(img_rgb, crop_left_pct)

    gray = grayscale(cropped_rgb)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu_mask, otsu_meta = auto_otsu_foreground(blur)

    kernel = np.ones((3, 3), np.uint8)

    closed = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    cleaned_small = remove_small_components(opened, min_area=min_area)
    final_mask, removed_border = remove_border_touching_components(cleaned_small, min_area=50)

    final_mask_white_bg = cv2.bitwise_not(final_mask)

    overlay = overlay_components(cropped_rgb, final_mask)
    quality = evaluate_photo_quality(gray, otsu_mask, final_mask)

    return {
        "cropped_rgb": cropped_rgb,
        "crop_px": crop_px,
        "gray": gray,
        "blur": blur,
        "otsu_mask": otsu_mask,
        "closed": closed,
        "opened": opened,
        "cleaned_small": cleaned_small,
        "final_mask": final_mask,
        "final_mask_white_bg": final_mask_white_bg,
        "overlay": overlay,
        "quality": quality,
        "removed_border_components": removed_border,
        "otsu_meta": otsu_meta,
    }


def human_adjustments_text(meta: dict, removed_border: int, min_area: int, crop_left_pct: int):
    items = []
    if crop_left_pct > 0:
        items.append(f"1. Recorte lateral izquierdo del {crop_left_pct}% para eliminar ruido pegado al borde.")
        n = 2
    else:
        n = 1

    items.extend([
        f"{n}. Conversión a escala de grises para eliminar información de color innecesaria.",
        f"{n+1}. Suavizado gaussiano para reducir pequeñas variaciones y ruido local.",
        f"{n+2}. Umbralización automática de Otsu usando polaridad {meta['polarity']} para separar objetos del fondo.",
        f"{n+3}. Cierre morfológico para volver más continuas las figuras.",
        f"{n+4}. Apertura morfológica ligera para eliminar pequeñas formaciones adicionales.",
        f"{n+5}. Eliminación de componentes pequeñas menores a {min_area} píxeles para retirar ruido aislado.",
        f"{n+6}. Eliminación de componentes que tocan los bordes de la imagen. Componentes removidas: {removed_border}.",
        f"{n+7}. Inversión de la máscara final solo para visualización, mostrando objetos negros sobre fondo blanco.",
    ])
    return items


# -------------------------------------------------
# Medición
# -------------------------------------------------
def classify_main_objects(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None

    candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area <= 0:
            continue
        x, y, w, h = cv2.boundingRect(c)
        peri = cv2.arcLength(c, True)
        circularity = 4 * np.pi * area / (peri * peri) if peri > 0 else 0.0
        aspect_ratio = max(w / max(h, 1), h / max(w, 1))
        candidates.append({
            "contour": c,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "circularity": circularity,
            "aspect_ratio": aspect_ratio,
        })

    if len(candidates) < 2:
        return None, None

    # Moneda: el objeto más circular de área razonable
    coin = max(candidates, key=lambda d: (d["circularity"], d["area"]))

    # Fibra: el objeto más alargado que no sea la moneda
    others = [d for d in candidates if d is not coin]
    if not others:
        return None, None
    fiber = max(others, key=lambda d: (d["aspect_ratio"], d["area"]))

    return fiber["contour"], coin["contour"]


def coin_diameter_px(coin_contour: np.ndarray) -> float:
    (_, _), radius = cv2.minEnclosingCircle(coin_contour)
    return float(2 * radius)


def fiber_perimeter_px(fiber_contour: np.ndarray) -> float:
    return float(cv2.arcLength(fiber_contour, True))


def fiber_area_px2(fiber_contour: np.ndarray) -> float:
    return float(cv2.contourArea(fiber_contour))


def fiber_skeleton_length_px(mask: np.ndarray, fiber_contour: np.ndarray) -> float:
    fiber_mask = np.zeros_like(mask)
    cv2.drawContours(fiber_mask, [fiber_contour], -1, 255, thickness=cv2.FILLED)

    binary = fiber_mask > 0
    skel = skeletonize(binary)

    length_px = float(np.count_nonzero(skel))
    return length_px, (skel.astype(np.uint8) * 255)


def fiber_avg_diameter_px(area_px2: float, length_px: float) -> float:
    if length_px <= 0:
        return 0.0
    return float(area_px2 / length_px)


def overlay_measurement_objects(img_rgb: np.ndarray, fiber_contour: np.ndarray, coin_contour: np.ndarray) -> np.ndarray:
    out = img_rgb.copy()
    cv2.drawContours(out, [fiber_contour], -1, (0, 255, 0), 3)
    cv2.drawContours(out, [coin_contour], -1, (255, 0, 0), 3)
    return out


# -------------------------------------------------
# Interfaz
# -------------------------------------------------
st.title("Evaluación previa, procesamiento y medición")
st.caption(
    "Carga una foto, evalúa si es adecuada, valida la detección y luego calcula escala y medidas usando la referencia circular."
)

with st.sidebar:
    st.header("Parámetros")
    min_area = st.slider("Área mínima para conservar componentes", 50, 3000, 300, 50)
    crop_left_pct = st.slider("Recorte izquierdo (%)", 0, 30, 0, 1)
    max_width = st.slider("Ancho máximo de visualización", 500, 1600, 900, 50)

uploaded = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Carga una imagen para comenzar.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
img_rgb = pil_to_rgb(image)
results = process_image(img_rgb, min_area=min_area, crop_left_pct=crop_left_pct)
quality = results["quality"]

st.subheader("Evaluación previa de la foto")

col_q1, col_q2, col_q3, col_q4 = st.columns(4)
col_q1.metric("Veredicto", quality["verdict"])
col_q2.metric("Contraste", f"{quality['contrast']:.1f}")
col_q3.metric("Objetos útiles", str(quality["component_count"]))
col_q4.metric("Ruido en bordes", f"{quality['border_noise_ratio']:.2f}")

if quality["verdict"] == "APTA":
    st.success("La foto es adecuada para continuar con el análisis.")
else:
    st.error("La foto no es adecuada. Se recomienda tomar otra antes de medir.")
    if quality["reasons"]:
        st.markdown("**Motivos detectados:**")
        for r in quality["reasons"]:
            st.write(f"- {r}")

st.subheader("Ajustes morfológicos aplicados")
for item in human_adjustments_text(results["otsu_meta"], results["removed_border_components"], min_area, crop_left_pct):
    st.write(item)

st.subheader("Vista previa para validación")
st.image(
    resize_keep_aspect(results["overlay"], max_width),
    caption="Detección sobre imagen original",
    use_container_width=True,
)

decision = st.radio(
    "¿Está de acuerdo con esta detección o desea cargar una nueva imagen?",
    ["Sí, estoy de acuerdo", "No, cargaré una nueva imagen"],
    horizontal=True,
)

if decision == "No, cargaré una nueva imagen":
    st.warning("Cargue una nueva imagen para repetir el análisis.")
    st.stop()

st.success("Imagen aceptada. Se muestra la máscara limpia final para análisis.")

st.subheader("Resultado final: máscara limpia")
st.image(
    resize_keep_aspect(results["final_mask_white_bg"], max_width),
    caption="Máscara limpia final (objetos negros sobre fondo blanco)",
    clamp=True,
    use_container_width=True,
)

st.subheader("Calibración con la moneda")
coin_real_diameter_mm = st.number_input(
    "Ingrese el diámetro real de la moneda o referencia circular (mm)",
    min_value=0.001,
    value=17.5,
    step=0.1,
    format="%.3f",
)

fiber_contour, coin_contour = classify_main_objects(results["final_mask"])

if fiber_contour is None or coin_contour is None:
    st.error("No se detectó correctamente la fibra y la referencia circular sobre la máscara final.")
    st.stop()

coin_px = coin_diameter_px(coin_contour)
px_per_mm = coin_px / coin_real_diameter_mm
mm_per_px = 1.0 / px_per_mm if px_per_mm > 0 else 0.0

fiber_area = fiber_area_px2(fiber_contour)
fiber_perimeter = fiber_perimeter_px(fiber_contour)
fiber_length, fiber_skel = fiber_skeleton_length_px(results["final_mask"], fiber_contour)
fiber_avg_diam = fiber_avg_diameter_px(fiber_area, fiber_length)

fiber_length_mm = fiber_length / px_per_mm if px_per_mm > 0 else 0.0
fiber_perimeter_mm = fiber_perimeter / px_per_mm if px_per_mm > 0 else 0.0
fiber_avg_diam_mm = fiber_avg_diam / px_per_mm if px_per_mm > 0 else 0.0
coin_px_mm = coin_px / px_per_mm if px_per_mm > 0 else 0.0

st.subheader("Resultados de medición")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Escala", f"{px_per_mm:.3f} px/mm")
m2.metric("Longitud fibra", f"{fiber_length_mm:.3f} mm")
m3.metric("Diámetro promedio fibra", f"{fiber_avg_diam_mm:.3f} mm")
m4.metric("Perímetro fibra", f"{fiber_perimeter_mm:.3f} mm")

st.dataframe(
    {
        "Magnitud": [
            "Diámetro referencia",
            "Longitud fibra",
            "Diámetro promedio fibra",
            "Perímetro fibra",
            "Área fibra",
        ],
        "Píxeles": [
            round(coin_px, 3),
            round(fiber_length, 3),
            round(fiber_avg_diam, 3),
            round(fiber_perimeter, 3),
            round(fiber_area, 3),
        ],
        "Milímetros": [
            round(coin_px_mm, 3),
            round(fiber_length_mm, 3),
            round(fiber_avg_diam_mm, 3),
            round(fiber_perimeter_mm, 3),
            round(fiber_area / (px_per_mm ** 2), 3) if px_per_mm > 0 else 0.0,
        ],
    },
    hide_index=True,
    use_container_width=True,
)

st.subheader("Visualización de medición")
measure_overlay = overlay_measurement_objects(results["cropped_rgb"], fiber_contour, coin_contour)

c1, c2 = st.columns(2)
with c1:
    st.image(
        resize_keep_aspect(measure_overlay, max_width),
        caption="Fibra en verde y referencia circular en azul",
        use_container_width=True,
    )
with c2:
    st.image(
        resize_keep_aspect(cv2.bitwise_not(fiber_skel), max_width),
        caption="Esqueleto de la fibra (negro sobre fondo blanco)",
        clamp=True,
        use_container_width=True,
    )

with st.expander("Ver pasos intermedios del procesamiento"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Imagen original o recortada**")
        st.image(resize_keep_aspect(results["cropped_rgb"], max_width), use_container_width=True)
    with c2:
        st.markdown("**Escala de grises**")
        st.image(resize_keep_aspect(results["gray"], max_width), clamp=True, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Suavizado gaussiano**")
        st.image(resize_keep_aspect(results["blur"], max_width), clamp=True, use_container_width=True)
    with c4:
        st.markdown(f"**Umbralización de Otsu** ({results['otsu_meta']['polarity']})")
        st.image(resize_keep_aspect(results["otsu_mask"], max_width), clamp=True, use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("**Cierre morfológico**")
        st.image(resize_keep_aspect(results["closed"], max_width), clamp=True, use_container_width=True)
    with c6:
        st.markdown("**Apertura morfológica**")
        st.image(resize_keep_aspect(results["opened"], max_width), clamp=True, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.markdown("**Limpieza por área mínima**")
        st.image(resize_keep_aspect(results["cleaned_small"], max_width), clamp=True, use_container_width=True)
    with c8:
        st.markdown("**Superposición final**")
        st.image(resize_keep_aspect(results["overlay"], max_width), use_container_width=True)
