
import math
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas


st.set_page_config(page_title="Medición manual sobre imagen", layout="wide")


# -----------------------------
# Utilidades geométricas
# -----------------------------
def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def polyline_length(points: List[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(distance(points[i], points[i + 1]) for i in range(len(points) - 1))


def polygon_area(points: List[Tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    x = np.array([p[0] for p in points], dtype=float)
    y = np.array([p[1] for p in points], dtype=float)
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(points: List[Tuple[float, float]], closed: bool = True) -> float:
    if len(points) < 2:
        return 0.0
    per = polyline_length(points)
    if closed:
        per += distance(points[-1], points[0])
    return per


def extract_points_from_path(path_data) -> List[Tuple[float, float]]:
    """
    path_data suele venir como lista estilo Fabric.js:
    [['M', x, y], ['Q', cx, cy, x, y], ['L', x, y], ...]
    Aproximamos tomando los puntos finales de cada segmento.
    """
    points = []
    if not isinstance(path_data, list):
        return points

    for cmd in path_data:
        if not isinstance(cmd, list) or len(cmd) < 3:
            continue
        tag = cmd[0]
        if tag == "M" and len(cmd) >= 3:
            points.append((float(cmd[1]), float(cmd[2])))
        elif tag == "L" and len(cmd) >= 3:
            points.append((float(cmd[1]), float(cmd[2])))
        elif tag == "Q" and len(cmd) >= 5:
            points.append((float(cmd[3]), float(cmd[4])))
        elif tag == "C" and len(cmd) >= 7:
            points.append((float(cmd[5]), float(cmd[6])))
        elif tag == "Z":
            pass
    return points


def object_to_line_length_px(obj: dict) -> Optional[float]:
    if obj.get("type") != "line":
        return None
    x1 = float(obj.get("x1", 0))
    y1 = float(obj.get("y1", 0))
    x2 = float(obj.get("x2", 0))
    y2 = float(obj.get("y2", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))
    return distance((x1 * scale_x, y1 * scale_y), (x2 * scale_x, y2 * scale_y))


def object_to_circle_diameter_px(obj: dict) -> Optional[float]:
    if obj.get("type") != "circle":
        return None
    radius = float(obj.get("radius", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))
    # Si la figura se deformó, usamos promedio de diámetros X/Y
    diameter_x = 2 * radius * scale_x
    diameter_y = 2 * radius * scale_y
    return float((diameter_x + diameter_y) / 2.0)


def object_to_path_points(obj: dict) -> List[Tuple[float, float]]:
    obj_type = obj.get("type")
    left = float(obj.get("left", 0))
    top = float(obj.get("top", 0))
    scale_x = float(obj.get("scaleX", 1))
    scale_y = float(obj.get("scaleY", 1))

    if obj_type == "path":
        raw_points = extract_points_from_path(obj.get("path", []))
        pts = [(left + x * scale_x, top + y * scale_y) for x, y in raw_points]
        return pts

    if obj_type == "line":
        x1 = float(obj.get("x1", 0)) * scale_x
        y1 = float(obj.get("y1", 0)) * scale_y
        x2 = float(obj.get("x2", 0)) * scale_x
        y2 = float(obj.get("y2", 0)) * scale_y
        return [(x1, y1), (x2, y2)]

    return []


def first_object_of_type(objects: List[dict], accepted_types: List[str]) -> Optional[dict]:
    for obj in objects:
        if obj.get("type") in accepted_types:
            return obj
    return None


def get_canvas_objects(canvas_result) -> List[dict]:
    if canvas_result is None or canvas_result.json_data is None:
        return []
    return canvas_result.json_data.get("objects", [])


# -----------------------------
# Interfaz
# -----------------------------
st.title("Medición manual sobre imagen")
st.caption(
    "Esta versión evita la detección automática de contornos. "
    "El usuario dibuja manualmente la referencia y las geometrías a medir."
)

uploaded = st.file_uploader("Carga una imagen", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Carga una imagen para comenzar.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
img_w, img_h = image.size

max_canvas_width = st.sidebar.slider("Ancho máximo del canvas", 500, 1400, 1000, 50)
display_scale = min(max_canvas_width / img_w, 1.0)
canvas_w = int(img_w * display_scale)
canvas_h = int(img_h * display_scale)

st.sidebar.markdown("### Recomendaciones")
st.sidebar.write(
    "- Usa **línea** para una longitud conocida.\n"
    "- Usa **círculo** para una moneda o figura circular.\n"
    "- Usa **trazo libre** para curvas o contornos."
)

st.subheader("Paso 1: referencia de escala")
ref_mode = st.radio(
    "Tipo de referencia",
    ["Longitud conocida (línea)", "Diámetro conocido (círculo)"],
    horizontal=True
)

ref_size_mm = st.number_input(
    "Tamaño real de la referencia (mm)",
    min_value=0.001,
    value=24.0,
    step=0.1,
    format="%.3f"
)

ref_tool = "line" if "Longitud" in ref_mode else "circle"

st.write("Dibuja **solo una** referencia sobre la imagen.")
ref_canvas = st_canvas(
    fill_color="rgba(255, 0, 0, 0.15)",
    stroke_width=3,
    stroke_color="#ff0000",
    background_image=image,
    update_streamlit=True,
    height=canvas_h,
    width=canvas_w,
    drawing_mode=ref_tool,
    key="ref_canvas",
    display_toolbar=True,
)

ref_objects = get_canvas_objects(ref_canvas)

if len(ref_objects) == 0:
    st.warning("Aún no has dibujado la referencia.")
    st.stop()

ref_obj = first_object_of_type(ref_objects, ["line", "circle"])
if ref_obj is None:
    st.error("No se encontró un objeto válido para la referencia.")
    st.stop()

if ref_obj.get("type") == "line":
    ref_px_display = object_to_line_length_px(ref_obj)
else:
    ref_px_display = object_to_circle_diameter_px(ref_obj)

if ref_px_display is None or ref_px_display <= 0:
    st.error("No fue posible calcular la referencia en píxeles.")
    st.stop()

# Convertir de pixeles del canvas a pixeles reales de la imagen
ref_px_real = ref_px_display / display_scale
px_per_mm = ref_px_real / ref_size_mm
mm_per_px = 1.0 / px_per_mm

st.success(f"Escala calculada: {px_per_mm:.4f} px/mm  |  {mm_per_px:.4f} mm/px")

st.subheader("Paso 2: longitud de la figura objetivo")
length_mode = st.radio(
    "Modo de longitud",
    ["Línea recta", "Curva (trazo libre)"],
    horizontal=True
)
length_tool = "line" if "Línea" in length_mode else "freedraw"

st.write("Dibuja la longitud del objeto. Si es curva, recórrela con trazo libre.")
length_canvas = st_canvas(
    fill_color="rgba(0, 0, 255, 0.05)",
    stroke_width=3,
    stroke_color="#0066ff",
    background_image=image,
    update_streamlit=True,
    height=canvas_h,
    width=canvas_w,
    drawing_mode=length_tool,
    key="length_canvas",
    display_toolbar=True,
)

length_objects = get_canvas_objects(length_canvas)
length_px_real = None

if len(length_objects) > 0:
    length_obj = length_objects[-1]
    if length_obj.get("type") == "line":
        length_px_display = object_to_line_length_px(length_obj)
        if length_px_display:
            length_px_real = length_px_display / display_scale
    elif length_obj.get("type") == "path":
        pts = object_to_path_points(length_obj)
        if len(pts) >= 2:
            length_px_display = polyline_length(pts)
            length_px_real = length_px_display / display_scale

st.subheader("Paso 3: diámetro")
st.write("Dibuja una línea atravesando el diámetro que quieres reportar.")
diam_canvas = st_canvas(
    fill_color="rgba(0, 255, 0, 0.05)",
    stroke_width=3,
    stroke_color="#00aa00",
    background_image=image,
    update_streamlit=True,
    height=canvas_h,
    width=canvas_w,
    drawing_mode="line",
    key="diam_canvas",
    display_toolbar=True,
)

diam_objects = get_canvas_objects(diam_canvas)
diam_px_real = None

if len(diam_objects) > 0:
    diam_obj = first_object_of_type(diam_objects, ["line"])
    if diam_obj:
        diam_px_display = object_to_line_length_px(diam_obj)
        if diam_px_display:
            diam_px_real = diam_px_display / display_scale

st.subheader("Paso 4: área y perímetro")
st.write(
    "Traza manualmente el contorno cerrado de la figura con **trazo libre**. "
    "Mientras más fiel sea el trazo, mejor la estimación."
)
area_canvas = st_canvas(
    fill_color="rgba(255, 165, 0, 0.08)",
    stroke_width=3,
    stroke_color="#ff8800",
    background_image=image,
    update_streamlit=True,
    height=canvas_h,
    width=canvas_w,
    drawing_mode="freedraw",
    key="area_canvas",
    display_toolbar=True,
)

area_objects = get_canvas_objects(area_canvas)
area_px2_real = None
perimeter_px_real = None

if len(area_objects) > 0:
    area_obj = area_objects[-1]
    if area_obj.get("type") == "path":
        pts = object_to_path_points(area_obj)
        if len(pts) >= 3:
            area_px2_display = polygon_area(pts)
            perimeter_px_display = polygon_perimeter(pts, closed=True)
            area_px2_real = area_px2_display / (display_scale ** 2)
            perimeter_px_real = perimeter_px_display / display_scale

st.markdown("---")
st.subheader("Resultados")

results = []

if length_px_real is not None:
    results.append({
        "Magnitud": "Longitud",
        "Pixeles": round(length_px_real, 3),
        "Milímetros": round(length_px_real / px_per_mm, 3),
    })

if diam_px_real is not None:
    results.append({
        "Magnitud": "Diámetro",
        "Pixeles": round(diam_px_real, 3),
        "Milímetros": round(diam_px_real / px_per_mm, 3),
    })

if area_px2_real is not None:
    results.append({
        "Magnitud": "Área",
        "Pixeles": round(area_px2_real, 3),
        "Milímetros": round(area_px2_real / (px_per_mm ** 2), 3),
    })

if perimeter_px_real is not None:
    results.append({
        "Magnitud": "Perímetro",
        "Pixeles": round(perimeter_px_real, 3),
        "Milímetros": round(perimeter_px_real / px_per_mm, 3),
    })

if results:
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)

    cols = st.columns(min(4, len(results)))
    metric_map = {row["Magnitud"]: row["Milímetros"] for row in results}
    i = 0
    for key in ["Longitud", "Diámetro", "Área", "Perímetro"]:
        if key in metric_map:
            unit = "mm²" if key == "Área" else "mm"
            cols[i].metric(key, f"{metric_map[key]:.3f} {unit}")
            i += 1
else:
    st.info("Dibuja las geometrías para ver resultados.")

st.markdown(
    """
    ### Cómo usar esta versión
    1. Dibuja una **referencia** de tamaño conocido.
    2. Dibuja la **longitud** del objeto, incluso si es curva.
    3. Dibuja una línea para el **diámetro**.
    4. Traza manualmente el **contorno** para estimar área y perímetro.

    ### Nota
    Esta versión es más manual, pero te da control total cuando la segmentación automática no funciona bien.
    """
)
