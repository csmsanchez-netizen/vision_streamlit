from skimage.morphology import skeletonize


st.set_page_config(page_title="Medición morfológica con Streamlit", layout="wide")


# ---------------------------
# Utilidades de imagen
# ---------------------------
def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_for_display(image: np.ndarray, max_width: int = 900) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def ensure_uint8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    gray = np.clip(gray, 0, 255)
    return gray.astype(np.uint8)


def order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def min_area_rect_dimensions(contour: np.ndarray) -> Tuple[float, float]:
    rect = cv2.minAreaRect(contour)
    (_, _), (w, h), _ = rect
    return float(max(w, h)), float(min(w, h))


def contour_circular_diameter_px(contour: np.ndarray) -> float:
    (_, _), radius = cv2.minEnclosingCircle(contour)
    return float(2 * radius)


def contour_mask(shape: Tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    return mask


def contour_metrics_px(contour: np.ndarray, gray_shape: Tuple[int, int]) -> Dict[str, float]:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    major, minor = min_area_rect_dimensions(contour)
    eq_diameter = float(np.sqrt(4 * area / np.pi)) if area > 0 else 0.0
    circular_diameter = contour_circular_diameter_px(contour)
    x, y, w, h = cv2.boundingRect(contour)
    return {
        "area_px2": area,
        "perimeter_px": perimeter,
        "major_px": major,
        "minor_px": minor,
        "bbox_w_px": float(w),
        "bbox_h_px": float(h),
        "eq_diameter_px": eq_diameter,
        "enclosing_diameter_px": circular_diameter,
        "bbox_x": int(x),
        "bbox_y": int(y),
    }


def draw_numbered_contours(image_bgr: np.ndarray, contours: List[np.ndarray], selected_idx: int | None = None) -> np.ndarray:
    canvas = image_bgr.copy()
    for idx, contour in enumerate(contours):
        color = (0, 255, 0) if idx == selected_idx else (255, 0, 0)
        cv2.drawContours(canvas, [contour], -1, color, 2)
        m = cv2.moments(contour)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
        cv2.putText(canvas, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    return canvas


# ---------------------------
# Procesamiento morfológico
# ---------------------------
@dataclass
class Candidate:
    name: str
    mask: np.ndarray
    contours: List[np.ndarray]
    score: float
    notes: str


def preprocess_candidates(image_bgr: np.ndarray, min_area_px: int = 150) -> List[Candidate]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = ensure_uint8(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    candidates: List[Candidate] = []

    def postprocess(mask: np.ndarray, name: str, notes: str) -> None:
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)

        variants = [
            (mask, name + " | base", notes + " | base"),
            (cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3), name + " | open3", notes + " | apertura 3x3"),
            (cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5), name + " | close5", notes + " | cierre 5x5"),
            (cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3), cv2.MORPH_CLOSE, kernel5), name + " | open3+close5", notes + " | apertura + cierre"),
            (cv2.dilate(mask, kernel3, iterations=1), name + " | dilate", notes + " | dilatación"),
        ]

        for variant_mask, variant_name, variant_notes in variants:
            cnts, _ = cv2.findContours(variant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered = [c for c in cnts if cv2.contourArea(c) >= min_area_px]
            score = score_candidate(filtered, gray.shape)
            candidates.append(Candidate(variant_name, variant_mask, filtered, score, variant_notes))

    # Otsu normal e invertido
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    postprocess(otsu, "Otsu", "Umbralización automática")
    postprocess(otsu_inv, "Otsu invertido", "Umbralización automática invertida")

    # Adaptive threshold normal e invertido
    adaptive = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 5)
    adaptive_inv = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 31, 5)
    postprocess(adaptive, "Adaptativo", "Umbralización adaptativa gaussiana")
    postprocess(adaptive_inv, "Adaptativo invertido", "Umbralización adaptativa gaussiana invertida")

    # Canny + cierre + relleno
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(edges)
    for c in cnts:
        if cv2.contourArea(c) >= min_area_px:
            cv2.drawContours(filled, [c], -1, 255, thickness=cv2.FILLED)
    postprocess(filled, "Canny relleno", "Canny + cierre + relleno de contornos")

    # Ordenar por score descendente
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def score_candidate(contours: List[np.ndarray], shape: Tuple[int, int]) -> float:
    if not contours:
        return 0.0
    image_area = shape[0] * shape[1]
    areas = sorted([cv2.contourArea(c) for c in contours], reverse=True)
    largest_ratio = areas[0] / image_area
    contour_count_penalty = abs(len(contours) - 3) * 0.1  # preferencia moderada por pocos objetos útiles
    area_balance = min(largest_ratio * 8, 1.8)
    total_ratio = min(sum(areas) / image_area * 4, 1.2)
    return round(area_balance + total_ratio - contour_count_penalty, 4)


# ---------------------------
# Longitud sobre esqueleto
# ---------------------------
def skeleton_length_px_from_contour(contour: np.ndarray, shape: Tuple[int, int]) -> float:
    mask = contour_mask(shape, contour)
    binary = mask > 0
    skel = skeletonize(binary)

    coords = np.argwhere(skel)
    if len(coords) < 2:
        major, _ = min_area_rect_dimensions(contour)
        return major

    coord_to_idx = {tuple(pt): i for i, pt in enumerate(coords)}
    neighbors = [[] for _ in range(len(coords))]

    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),           (0, 1),
                  (1, -1),  (1, 0),  (1, 1)]

    for i, (r, c) in enumerate(coords):
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            j = coord_to_idx.get((nr, nc))
            if j is not None:
                w = math.sqrt(2) if dr != 0 and dc != 0 else 1.0
                neighbors[i].append((j, w))

    degrees = [len(n) for n in neighbors]
    endpoints = [i for i, d in enumerate(degrees) if d == 1]

    def dijkstra(start: int) -> Tuple[List[float], int]:
        dist = [float("inf")] * len(coords)
        dist[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            current_dist, node = heapq.heappop(pq)
            if current_dist > dist[node]:
                continue
            for nxt, w in neighbors[node]:
                nd = current_dist + w
                if nd < dist[nxt]:
                    dist[nxt] = nd
                    heapq.heappush(pq, (nd, nxt))
        farthest = int(np.nanargmax(np.array(dist, dtype=np.float64)))
        return dist, farthest

    # Si no hay endpoints (forma cerrada o esqueleto complejo), usar un nodo arbitrario.
    start = endpoints[0] if endpoints else 0
    _, far1 = dijkstra(start)
    dist2, far2 = dijkstra(far1)

    length = dist2[far2]
    if not np.isfinite(length) or length <= 0:
        major, _ = min_area_rect_dimensions(contour)
        return major
    return float(length)


def compute_measurements(contour: np.ndarray, shape: Tuple[int, int], px_per_mm: float) -> Dict[str, float]:
    metrics = contour_metrics_px(contour, shape)
    length_px = skeleton_length_px_from_contour(contour, shape)
    avg_diameter_px = (metrics["area_px2"] / length_px) if length_px > 0 else 0.0

    return {
        "length_px": length_px,
        "length_mm": length_px / px_per_mm,
        "avg_diameter_px": avg_diameter_px,
        "avg_diameter_mm": avg_diameter_px / px_per_mm,
        "area_px2": metrics["area_px2"],
        "area_mm2": metrics["area_px2"] / (px_per_mm ** 2),
        "perimeter_px": metrics["perimeter_px"],
        "perimeter_mm": metrics["perimeter_px"] / px_per_mm,
        "eq_diameter_px": metrics["eq_diameter_px"],
        "eq_diameter_mm": metrics["eq_diameter_px"] / px_per_mm,
        "major_px": metrics["major_px"],
        "major_mm": metrics["major_px"] / px_per_mm,
        "minor_px": metrics["minor_px"],
        "minor_mm": metrics["minor_px"] / px_per_mm,
    }


# ---------------------------
# Interfaz
# ---------------------------
st.title("Medición morfológica con imagen de referencia")
st.caption(
    "Carga una imagen, prueba automáticamente varios procesamientos morfológicos, valida el mejor, "
    "elige un contorno de referencia y luego mide otra figura incluso si es curva."
)

with st.sidebar:
    st.header("Parámetros")
    min_area_px = st.slider("Área mínima de contorno (px²)", 20, 5000, 150, 10)
    max_display_width = st.slider("Ancho máximo de visualización", 500, 1600, 950, 50)

uploaded = st.file_uploader("Carga una imagen (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if "candidate_idx" not in st.session_state:
    st.session_state.candidate_idx = 0
if "approved_candidate" not in st.session_state:
    st.session_state.approved_candidate = None
if "prev_file_key" not in st.session_state:
    st.session_state.prev_file_key = None

if uploaded is not None:
    file_key = f"{uploaded.name}-{uploaded.size}"
    if st.session_state.prev_file_key != file_key:
        st.session_state.candidate_idx = 0
        st.session_state.approved_candidate = None
        st.session_state.prev_file_key = file_key

    image_pil = Image.open(uploaded)
    image_bgr = pil_to_bgr(image_pil)
    candidates = preprocess_candidates(image_bgr, min_area_px=min_area_px)

    if not candidates:
        st.error("No se pudieron generar candidatos de procesamiento.")
        st.stop()

    st.subheader("Imagen original")
    st.image(resize_for_display(bgr_to_rgb(image_bgr), max_display_width), channels="RGB", use_container_width=True)

    st.subheader("Paso 1: revisión del procesamiento morfológico")
    candidate_count = len(candidates)

    col_nav1, col_nav2, col_nav3 = st.columns([1, 1, 2])
    with col_nav1:
        if st.button("Procesamiento anterior", disabled=st.session_state.candidate_idx <= 0):
            st.session_state.candidate_idx -= 1
            st.rerun()
    with col_nav2:
        if st.button("Siguiente procesamiento", disabled=st.session_state.candidate_idx >= candidate_count - 1):
            st.session_state.candidate_idx += 1
            st.rerun()
    with col_nav3:
        st.write(f"Candidato {st.session_state.candidate_idx + 1} de {candidate_count}")

    selected_candidate = candidates[st.session_state.candidate_idx]

    if selected_candidate.contours:
        overlay = draw_numbered_contours(image_bgr, selected_candidate.contours)
    else:
        overlay = image_bgr.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Procesamiento:** {selected_candidate.name}")
        st.caption(f"Notas: {selected_candidate.notes} | Score automático: {selected_candidate.score}")
        st.image(
            resize_for_display(selected_candidate.mask, max_width=max_display_width),
            caption="Máscara binaria resultante",
            use_container_width=True,
            clamp=True,
        )
    with col2:
        st.image(
            resize_for_display(bgr_to_rgb(overlay), max_width=max_display_width),
            caption="Contornos detectados y numerados",
            use_container_width=True,
        )

    if selected_candidate.contours:
        contour_rows = []
        for idx, c in enumerate(selected_candidate.contours):
            m = contour_metrics_px(c, selected_candidate.mask.shape)
            contour_rows.append({
                "id": idx,
                "área_px2": round(m["area_px2"], 1),
                "perímetro_px": round(m["perimeter_px"], 1),
                "long_mayor_px": round(m["major_px"], 1),
                "long_menor_px": round(m["minor_px"], 1),
                "diam_enclosing_px": round(m["enclosing_diameter_px"], 1),
                "diam_eq_px": round(m["eq_diameter_px"], 1),
            })
        st.dataframe(pd.DataFrame(contour_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("Este procesamiento no detectó contornos filtrados por área. Prueba otro candidato.")

    col_ok1, col_ok2 = st.columns([1, 1])
    with col_ok1:
        if st.button("✅ Usar este procesamiento", disabled=not bool(selected_candidate.contours)):
            st.session_state.approved_candidate = st.session_state.candidate_idx
            st.rerun()
    with col_ok2:
        if st.button("🔄 Rechazar y seguir probando"):
            if st.session_state.candidate_idx < candidate_count - 1:
                st.session_state.candidate_idx += 1
            st.rerun()

    if st.session_state.approved_candidate is not None:
        approved_candidate = candidates[st.session_state.approved_candidate]
        st.success(f"Procesamiento aprobado: {approved_candidate.name}")

        st.subheader("Paso 2: seleccionar contorno de referencia")
        ref_idx = st.selectbox(
            "Elige el ID del contorno que actuará como referencia",
            options=list(range(len(approved_candidate.contours))),
            format_func=lambda x: f"Contorno {x}",
            key="ref_idx",
        )

        ref_mode = st.radio(
            "Tipo de referencia",
            options=["Longitud conocida", "Diámetro de figura circular"],
            horizontal=True,
        )
        ref_value_mm = st.number_input(
            "Tamaño real de la referencia (mm)",
            min_value=0.001,
            value=24.0,
            step=0.1,
            format="%.3f",
        )

        ref_contour = approved_candidate.contours[ref_idx]
        ref_metrics = contour_metrics_px(ref_contour, approved_candidate.mask.shape)

        if ref_mode == "Longitud conocida":
            ref_px = ref_metrics["major_px"]
            ref_descriptor = "longitud mayor del contorno de referencia"
        else:
            ref_px = ref_metrics["enclosing_diameter_px"]
            ref_descriptor = "diámetro de la circunferencia envolvente mínima"

        if ref_px <= 0:
            st.error("No fue posible calcular el tamaño en píxeles de la referencia.")
            st.stop()

        px_per_mm = ref_px / ref_value_mm
        mm_per_px = 1.0 / px_per_mm

        st.info(
            f"Escala calculada: **{px_per_mm:.4f} px/mm** "
            f"(**{mm_per_px:.4f} mm/px**) usando la {ref_descriptor}."
        )

        st.subheader("Paso 3: seleccionar el contorno a medir")
        target_options = [i for i in range(len(approved_candidate.contours)) if i != ref_idx]
        if not target_options:
            st.warning("Solo hay un contorno útil detectado. Necesitas al menos otro contorno para medir.")
            st.stop()

        target_idx = st.selectbox(
            "Elige el ID del contorno objetivo",
            options=target_options,
            format_func=lambda x: f"Contorno {x}",
            key="target_idx",
        )

        target_contour = approved_candidate.contours[target_idx]

        overlay_target = draw_numbered_contours(image_bgr, approved_candidate.contours, selected_idx=target_idx)
        st.image(
            resize_for_display(bgr_to_rgb(overlay_target), max_width=max_display_width),
            caption="Contorno objetivo resaltado",
            use_container_width=True,
        )

        measurements = compute_measurements(target_contour, approved_candidate.mask.shape, px_per_mm)

        st.subheader("Resultados")
        result_cols = st.columns(4)
        result_cols[0].metric("Longitud estimada", f"{measurements['length_mm']:.3f} mm")
        result_cols[1].metric("Diámetro promedio", f"{measurements['avg_diameter_mm']:.3f} mm")
        result_cols[2].metric("Área", f"{measurements['area_mm2']:.3f} mm²")
        result_cols[3].metric("Perímetro", f"{measurements['perimeter_mm']:.3f} mm")

        details_df = pd.DataFrame([
            {"magnitud": "Longitud de esqueleto", "valor_px": measurements["length_px"], "valor_mm": measurements["length_mm"]},
            {"magnitud": "Diámetro promedio (área / longitud)", "valor_px": measurements["avg_diameter_px"], "valor_mm": measurements["avg_diameter_mm"]},
            {"magnitud": "Área", "valor_px": measurements["area_px2"], "valor_mm": measurements["area_mm2"]},
            {"magnitud": "Perímetro", "valor_px": measurements["perimeter_px"], "valor_mm": measurements["perimeter_mm"]},
            {"magnitud": "Diámetro equivalente circular", "valor_px": measurements["eq_diameter_px"], "valor_mm": measurements["eq_diameter_mm"]},
            {"magnitud": "Longitud mayor por rectángulo mínimo", "valor_px": measurements["major_px"], "valor_mm": measurements["major_mm"]},
            {"magnitud": "Longitud menor por rectángulo mínimo", "valor_px": measurements["minor_px"], "valor_mm": measurements["minor_mm"]},
        ])
        st.dataframe(
            details_df.style.format({"valor_px": "{:.3f}", "valor_mm": "{:.3f}"}),
            use_container_width=True,
            hide_index=True
        )

        st.markdown(
            """
            **Cómo interpreta la app las medidas**
            - **Longitud**: se estima sobre el esqueleto del objeto, por eso puede seguir figuras curvas.
            - **Diámetro promedio**: se calcula como área / longitud, útil para fibras o elementos alargados.
            - **Área**: se obtiene del contorno segmentado.
            - **Perímetro**: se obtiene del contorno externo.
            """
        )
else:
    st.info("Carga una imagen para comenzar.")

