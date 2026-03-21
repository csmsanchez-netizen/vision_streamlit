import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Evaluación y procesamiento morfológico", layout="wide")


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


def remove_small_components(mask: np.ndarray, min_area: int = 300):
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
        reasons.append("No se detectó claramente un objeto circular como la moneda.")
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


def process_image(img_rgb: np.ndarray, min_area: int = 300):
    gray = grayscale(img_rgb)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu_mask, otsu_meta = auto_otsu_foreground(blur)

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(otsu_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_small = remove_small_components(closed, min_area=min_area)
    final_mask, removed_border = remove_border_touching_components(cleaned_small, min_area=50)

    overlay = overlay_components(img_rgb, final_mask)
    quality = evaluate_photo_quality(gray, otsu_mask, final_mask)

    return {
        "gray": gray,
        "blur": blur,
        "otsu_mask": otsu_mask,
        "otsu_meta": otsu_meta,
        "closed": closed,
        "cleaned_small": cleaned_small,
        "final_mask": final_mask,
        "overlay": overlay,
        "quality": quality,
        "removed_border_components": removed_border,
    }


def human_adjustments_text(meta: dict, removed_border: int, min_area: int):
    return [
        "1. Conversión a escala de grises para eliminar información de color innecesaria.",
        "2. Suavizado gaussiano para reducir pequeñas variaciones y ruido local.",
        f"3. Umbralización automática de Otsu usando polaridad {meta['polarity']} para separar objetos del fondo.",
        "4. Cierre morfológico para cerrar pequeños huecos y volver más continuas las figuras.",
        f"5. Eliminación de componentes pequeñas menores a {min_area} píxeles para retirar ruido aislado.",
        f"6. Eliminación de componentes que tocan los bordes de la imagen. Componentes removidas: {removed_border}.",
    ]


st.title("Evaluación previa y procesamiento morfológico")
st.caption(
    "Carga una foto y la app evaluará si es adecuada para medición. "
    "Luego mostrará el pipeline morfológico y el resultado final."
)

with st.sidebar:
    st.header("Parámetros")
    min_area = st.slider("Área mínima para conservar componentes", 50, 3000, 300, 50)
    max_width = st.slider("Ancho máximo de visualización", 500, 1600, 900, 50)

uploaded = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Carga una imagen para comenzar.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
img_rgb = pil_to_rgb(image)
results = process_image(img_rgb, min_area=min_area)
quality = results["quality"]

st.subheader("Evaluación previa de la foto")
col_q1, col_q2, col_q3, col_q4 = st.columns(4)
col_q1.metric("Veredicto", quality["verdict"])
col_q2.metric("Contraste", f"{quality['contrast']:.1f}")
col_q3.metric("Objetos útiles", str(quality["component_count"]))
col_q4.metric("Ruido en bordes", f"{quality['border_noise_ratio']:.2f}")

if quality["verdict"] == "APTA":
    st.success("La foto es adecuada para continuar con medición automática básica.")
else:
    st.error("La foto no es adecuada. Se recomienda tomar otra antes de medir.")
    if quality["reasons"]:
        st.markdown("**Motivos detectados:**")
        for r in quality["reasons"]:
            st.write(f"- {r}")

st.subheader("Ajustes morfológicos aplicados")
for item in human_adjustments_text(results["otsu_meta"], results["removed_border_components"], min_area):
    st.write(item)

st.subheader("Resultados visuales")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Imagen original**")
    st.image(resize_keep_aspect(img_rgb, max_width), use_container_width=True)
with c2:
    st.markdown("**Resultado final: máscara limpia**")
    st.image(resize_keep_aspect(results["final_mask"], max_width), clamp=True, use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    st.markdown("**Escala de grises**")
    st.image(resize_keep_aspect(results["gray"], max_width), clamp=True, use_container_width=True)
with c4:
    st.markdown("**Suavizado gaussiano**")
    st.image(resize_keep_aspect(results["blur"], max_width), clamp=True, use_container_width=True)

c5, c6 = st.columns(2)
with c5:
    st.markdown(f"**Umbralización de Otsu** ({results['otsu_meta']['polarity']})")
    st.image(resize_keep_aspect(results["otsu_mask"], max_width), clamp=True, use_container_width=True)
with c6:
    st.markdown("**Cierre morfológico**")
    st.image(resize_keep_aspect(results["closed"], max_width), clamp=True, use_container_width=True)

c7, c8 = st.columns(2)
with c7:
    st.markdown("**Limpieza por área mínima**")
    st.image(resize_keep_aspect(results["cleaned_small"], max_width), clamp=True, use_container_width=True)
with c8:
    st.markdown("**Resultado final superpuesto**")
    st.image(resize_keep_aspect(results["overlay"], max_width), use_container_width=True)

st.markdown(
    """
### Interpretación
- Si la foto es **APTA**, ya tienes una base razonable para continuar con medición automática.
- Si la foto dice **REPETIR FOTO**, normalmente conviene mejorar fondo, iluminación y orientación de la cámara antes de medir.
"""
)
