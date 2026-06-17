import os
import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI


APP_TITLE = "Generador de informes PIE"
DEFAULT_MODEL = "gpt-4.1-mini"
BASE_DIR = Path(__file__).resolve().parent

PROFESSIONALS = ["Psicología", "Psicopedagogía", "Fonoaudiología"]

TESTS_BY_PROFESSIONAL = {
    "Psicología": ["WISC-V", "WAIS-IV", "ICAP", "Otro"],
    "Psicopedagogía": ["Evalua", "Tabulatest", "CLP", "Pruebas pedagogicas internas", "Otro"],
    "Fonoaudiología": ["TECAL", "STSG", "TEPROSIF-R", "TAR", "EFA", "Protocolo pragmatico", "Otro"],
}

PSYCHOLOGY_COLUMNS = [
    "Escala",
    "Suma de puntajes equivalentes",
    "Puntaje compuesto",
    "Percentil",
    "Intervalo de confianza 95%",
    "Rango descriptivo",
]

PSYCHOPEDAGOGY_COLUMNS = [
    "Area evaluada",
    "Puntaje bruto",
    "Cantidad de errores",
    "Puntaje estandar/equivalente",
    "Percentil",
    "Intervalo de confianza 95%",
    "Rango descriptivo",
    "Nivel de desempeno",
    "Observacion cualitativa",
]

SPEECH_COLUMNS = [
    "Test",
    "Area evaluada",
    "Puntaje obtenido",
    "Cantidad de errores",
    "Percentil / DS",
    "Intervalo de confianza 95%",
    "Rango descriptivo",
    "Rendimiento",
    "Nivel de desempeno",
    "Observacion clinica",
]

DERIVED_FIELDS = [
    "puntaje_compuesto",
    "percentil",
    "intervalo_confianza",
    "rango_descriptivo",
    "nivel_desempeno",
]

TEST_BAREMO_FILES = {
    "WISC-V": BASE_DIR / "baremos/psicologia/wisc_v.json",
    "WAIS-IV": BASE_DIR / "baremos/psicologia/wais_iv.json",
    "Evalua": BASE_DIR / "baremos/psicopedagogia/evalua.json",
    "Tabulatest": BASE_DIR / "baremos/psicopedagogia/tabulatest.json",
    "TECAL": BASE_DIR / "baremos/fonoaudiologia/tecal.json",
    "STSG": BASE_DIR / "baremos/fonoaudiologia/stsg.json",
    "TEPROSIF-R": BASE_DIR / "baremos/fonoaudiologia/teprosif_r.json",
}

TABULATEST_TEMPLATE_PATH = BASE_DIR / "data/tabulatest/tabulatest_templates.json"
TABULATEST_COURSE_LABELS = {
    "prekinder": "Prekínder",
    "kinder": "Kínder",
    "1_basico": "1° Básico",
    "2_basico": "2° Básico",
    "3_basico": "3° Básico",
    "4_basico": "4° Básico",
    "5_basico": "5° Básico",
    "6_basico": "6° Básico",
    "7_basico": "7° Básico",
    "8_basico": "8° Básico",
    "1_medio": "I° Medio",
    "2_medio": "II° Medio",
}
TABULATEST_LABEL_TO_KEY = {label: key for key, label in TABULATEST_COURSE_LABELS.items()}

TABULATEST_ADMIN_COLUMNS = [
    "curso",
    "area",
    "puntaje_bruto",
    "percentil",
    "nivel_desempeno",
    "descripcion",
]

PSYCHOPEDAGOGY_AREAS = [
    "Lectura",
    "Comprension lectora",
    "Escritura",
    "Ortografia",
    "Calculo",
    "Resolucion de problemas",
    "Memoria y atencion",
]

TABULATEST_COLUMNS = [
    "Curso",
    "Área",
    "Subtest",
    "Puntaje máximo",
    "Puntaje obtenido",
    "Porcentaje de logro",
    "Nivel de desempeño",
]

SPEECH_AREAS = [
    "Lenguaje comprensivo",
    "Lenguaje expresivo",
    "Fonologia",
    "Articulacion",
    "Morfosintaxis",
    "Semantica",
    "Pragmatica",
]

REPORT_TITLES = {
    "Psicología": "Informe Psicologico Escolar PIE",
    "Psicopedagogía": "Informe Psicopedagogico PIE",
    "Fonoaudiología": "Informe Fonoaudiologico PIE",
}


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def informed(value):
    if value is None:
        return "no informado"
    clean = str(value).strip()
    if clean.lower() == "nan":
        return "no informado"
    return clean if clean else "no informado"


def normalize_text(value):
    return informed(value).lower().replace("_", " ").replace("-", " ").strip()


def empty_conversion():
    return {
        "puntaje_compuesto": "no informado",
        "percentil": "no informado",
        "intervalo_confianza": "no informado",
        "rango_descriptivo": "no informado",
        "nivel_desempeno": "no informado",
    }


def load_baremo(test):
    baremo_path = TEST_BAREMO_FILES.get(test)
    if not baremo_path or not baremo_path.exists():
        return None
    try:
        with baremo_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return None


def load_tabulatest_templates():
    if not TABULATEST_TEMPLATE_PATH.exists():
        return {}, f"No se encontro el archivo {TABULATEST_TEMPLATE_PATH}."
    try:
        with TABULATEST_TEMPLATE_PATH.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError:
        return {}, "El archivo tabulatest_templates.json tiene errores de formato y no pudo leerse."
    except OSError:
        return {}, "No fue posible abrir tabulatest_templates.json."

    templates = payload.get("templates")
    if not isinstance(templates, dict):
        return {}, "El archivo tabulatest_templates.json no contiene una clave 'templates' valida."
    return templates, ""


def get_tabulatest_course_options(templates):
    available = []
    for key, label in TABULATEST_COURSE_LABELS.items():
        if key in templates:
            available.append(label)
    return available


def has_tabulatest_scores(df):
    records = df.to_dict("records") if isinstance(df, pd.DataFrame) else []
    return any(parse_number(row.get("Puntaje obtenido")) is not None for row in records)


def normalize_tabulatest_df(df):
    normalized = df.copy()
    for column in TABULATEST_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = None
    normalized = normalized[TABULATEST_COLUMNS]
    normalized["Puntaje máximo"] = normalized["Puntaje máximo"].apply(parse_number)
    normalized["Puntaje obtenido"] = normalized["Puntaje obtenido"].apply(parse_number)
    return normalized


def calculate_tabulatest_df(df):
    normalized = normalize_tabulatest_df(df)
    calculated_records = calculate_tabulatest_records(normalized.to_dict("records"))
    calculated_df = pd.DataFrame(calculated_records, columns=TABULATEST_COLUMNS)
    calculated_df["Puntaje obtenido"] = calculated_df["Puntaje obtenido"].apply(parse_number)
    calculated_df["Porcentaje de logro"] = calculated_df["Porcentaje de logro"].apply(
        lambda value: parse_number(value) if parse_number(value) is not None else None
    )
    calculated_df["Nivel de desempeño"] = calculated_df["Nivel de desempeño"].replace("no informado", "")
    return calculated_df


def tabulatest_dataframes_equal(left, right):
    left_normalized = normalize_tabulatest_df(left).fillna("")
    right_normalized = normalize_tabulatest_df(right).fillna("")
    return left_normalized.equals(right_normalized)


def get_tabulatest_session_df(course_label, templates):
    session_key = f"tabulatest_df_{course_label}"
    if session_key not in st.session_state:
        st.session_state[session_key] = calculate_tabulatest_df(make_tabulatest_rows(course_label, templates))
    return st.session_state[session_key]


def render_tabulatest_course_tabs(templates):
    labels = list(TABULATEST_COURSE_LABELS.values())
    tabs = st.tabs(labels)
    selected_df = None
    selected_course = ""
    first_available_df = None
    first_available_course = ""

    for tab, label in zip(tabs, labels):
        with tab:
            course_key = TABULATEST_LABEL_TO_KEY.get(label, label)
            if course_key not in templates:
                st.warning(
                    "No hay plantilla cargada para este curso en tabulatest_templates.json. "
                    "Puede agregarla al archivo JSON sin modificar el codigo de la aplicacion."
                )
                continue

            session_df_key = f"tabulatest_df_{label}"
            editor_key = f"tabulatest_editor_{label}"
            current_df = get_tabulatest_session_df(label, templates)
            edited = st.data_editor(
                current_df,
                num_rows="fixed",
                use_container_width=True,
                hide_index=True,
                column_order=TABULATEST_COLUMNS,
                disabled=[column for column in TABULATEST_COLUMNS if column != "Puntaje obtenido"],
                column_config={
                    "Curso": st.column_config.TextColumn("Curso"),
                    "Área": st.column_config.TextColumn("Área"),
                    "Subtest": st.column_config.TextColumn("Subtest"),
                    "Puntaje máximo": st.column_config.NumberColumn("Puntaje máximo", min_value=0.0, step=1.0),
                    "Puntaje obtenido": st.column_config.NumberColumn(
                        "Puntaje obtenido",
                        min_value=0.0,
                        step=1.0,
                        required=False,
                        help="Ingrese solo el puntaje obtenido. Puede dejarlo vacio.",
                    ),
                    "Porcentaje de logro": st.column_config.NumberColumn(
                        "Porcentaje de logro",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        format="%.1f%%",
                    ),
                    "Nivel de desempeño": st.column_config.TextColumn("Nivel de desempeño"),
                },
                key=editor_key,
            )
            updated_df = calculate_tabulatest_df(edited)
            if not tabulatest_dataframes_equal(current_df, updated_df):
                st.session_state[session_df_key] = updated_df
                st.rerun()
            st.session_state[session_df_key] = updated_df

            if first_available_df is None:
                first_available_df = updated_df
                first_available_course = label

            if selected_df is None and has_tabulatest_scores(updated_df):
                selected_df = updated_df
                selected_course = label

    if selected_df is not None:
        return selected_df, selected_course
    if first_available_df is not None:
        return first_available_df, first_available_course

    empty_df = pd.DataFrame([{column: "" for column in TABULATEST_COLUMNS}], columns=TABULATEST_COLUMNS)
    return empty_df, ""


def match_field(row_value, expected_value):
    expected = normalize_text(expected_value)
    if expected in ["", "no informado", "todos", "todas", "*"]:
        return True
    return normalize_text(row_value) == expected


def convertir_puntajes(test, edad, curso, escala, puntaje_base):
    baremo = load_baremo(test)
    if not baremo:
        return empty_conversion()

    conversions = baremo.get("conversiones", [])
    for item in conversions:
        if not match_field(edad, item.get("edad", "*")):
            continue
        if not match_field(curso, item.get("curso", "*")):
            continue
        expected_scale = item.get("escala", item.get("area", "*"))
        if not match_field(escala, expected_scale):
            continue
        expected_score = item.get("puntaje_base", item.get("puntaje_bruto", ""))
        if not match_field(puntaje_base, expected_score):
            continue

        return {
            "puntaje_compuesto": informed(item.get("puntaje_compuesto")),
            "percentil": informed(item.get("percentil")),
            "intervalo_confianza": informed(item.get("intervalo_confianza")),
            "rango_descriptivo": informed(item.get("rango_descriptivo", item.get("descripcion"))),
            "nivel_desempeno": informed(item.get("nivel_desempeno")),
        }

    return empty_conversion()


def get_base_score(row):
    for field in [
        "Suma de puntajes equivalentes",
        "Puntaje bruto",
        "Puntaje obtenido",
        "Cantidad de errores",
        "Otro puntaje base",
    ]:
        value = row.get(field)
        if informed(value) != "no informado":
            return value
    return ""


def get_scale_name(row):
    for field in ["Escala", "Area evaluada", "Test"]:
        value = row.get(field)
        if informed(value) != "no informado":
            return value
    return ""


def has_loaded_baremo(test):
    baremo = load_baremo(test)
    return bool(baremo and baremo.get("conversiones"))


def load_tabulatest_admin_df():
    baremo = load_baremo("Tabulatest") or {"conversiones": []}
    rows = []
    for item in baremo.get("conversiones", []):
        rows.append(
            {
                "curso": informed(item.get("curso")),
                "area": informed(item.get("area", item.get("escala"))),
                "puntaje_bruto": informed(item.get("puntaje_bruto", item.get("puntaje_base"))),
                "percentil": informed(item.get("percentil")),
                "nivel_desempeno": informed(item.get("nivel_desempeno")),
                "descripcion": informed(item.get("descripcion", item.get("rango_descriptivo"))),
            }
        )
    if not rows:
        rows = [{column: "" for column in TABULATEST_ADMIN_COLUMNS}]
    return pd.DataFrame(rows, columns=TABULATEST_ADMIN_COLUMNS)


def save_tabulatest_baremos(df):
    records = []
    for row in dataframe_records(df):
        curso = informed(row.get("curso"))
        area = informed(row.get("area"))
        puntaje_bruto = informed(row.get("puntaje_bruto"))
        if curso == "no informado" and area == "no informado" and puntaje_bruto == "no informado":
            continue
        records.append(
            {
                "edad": "*",
                "curso": curso,
                "area": area,
                "escala": area,
                "puntaje_bruto": puntaje_bruto,
                "puntaje_base": puntaje_bruto,
                "percentil": informed(row.get("percentil")),
                "nivel_desempeno": informed(row.get("nivel_desempeno")),
                "descripcion": informed(row.get("descripcion")),
                "rango_descriptivo": informed(row.get("descripcion")),
                "puntaje_compuesto": "no informado",
                "intervalo_confianza": "no informado",
            }
        )

    payload = {
        "test": "Tabulatest",
        "descripcion": "Baremo Tabulatest ingresado manualmente por el equipo PIE.",
        "conversiones": records,
    }
    path = TEST_BAREMO_FILES["Tabulatest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    return len(records)


def apply_baremos_to_records(test, edad, curso, records):
    converted_records = []
    any_conversion = False
    for row in records:
        converted = dict(row)
        conversion = convertir_puntajes(test, edad, curso, get_scale_name(row), get_base_score(row))

        if conversion["puntaje_compuesto"] != "no informado":
            converted["Puntaje compuesto"] = conversion["puntaje_compuesto"]
            converted["Puntaje estandar/equivalente"] = conversion["puntaje_compuesto"]
        if conversion["percentil"] != "no informado":
            converted["Percentil"] = conversion["percentil"]
            converted["Percentil / DS"] = conversion["percentil"]
        if conversion["intervalo_confianza"] != "no informado":
            converted["Intervalo de confianza 95%"] = conversion["intervalo_confianza"]
        if conversion["rango_descriptivo"] != "no informado":
            converted["Rango descriptivo"] = conversion["rango_descriptivo"]
            converted["Rendimiento"] = conversion["rango_descriptivo"]
        if conversion["nivel_desempeno"] != "no informado":
            converted["Nivel de desempeno"] = conversion["nivel_desempeno"]
            converted["Rendimiento"] = conversion["nivel_desempeno"]

        if any(value != "no informado" for value in conversion.values()):
            any_conversion = True
        converted_records.append(converted)

    return converted_records, any_conversion


def parse_number(value):
    text = informed(value)
    if text == "no informado":
        return None
    cleaned = text.replace(",", ".")
    number_chars = []
    for char in cleaned:
        if char.isdigit() or char in [".", "-"]:
            number_chars.append(char)
    if not number_chars:
        return None
    try:
        return float("".join(number_chars))
    except ValueError:
        return None


def escape_html(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def classify_standard_score(score):
    if score is None:
        return "no informado", "No se cuenta con puntaje compuesto suficiente para establecer un rango interpretativo."
    if score >= 130:
        return "muy superior", "El resultado se ubica en un rango muy superior, asociado a recursos cognitivos altamente desarrollados."
    if score >= 120:
        return "superior", "El resultado se ubica en un rango superior, lo que constituye una fortaleza cognitiva relevante."
    if score >= 110:
        return "promedio alto", "El resultado se ubica en un rango promedio alto, sugiriendo recursos favorecedores para el aprendizaje."
    if score >= 90:
        return "promedio", "El resultado se encuentra dentro de parametros esperados para su grupo de referencia."
    if score >= 80:
        return "promedio bajo", "El resultado se ubica en un rango promedio bajo, por lo que puede requerir mediacion en tareas de mayor complejidad."
    if score >= 70:
        return "limite", "El resultado se ubica en rango limite, orientando la necesidad de apoyos sistematicos y ajustes pedagogicos."
    return "muy bajo", "El resultado se ubica en un rango muy bajo, sugiriendo necesidad de apoyos intensivos y seguimiento profesional."


def classify_percentile(percentile):
    if percentile is None:
        return "no informado", "No se ingreso percentil, por lo que no se establece ubicacion normativa."
    if percentile >= 75:
        return "alto", "El desempeno se ubica sobre lo esperado y puede considerarse una fortaleza relativa."
    if percentile >= 25:
        return "esperado", "El desempeno se ubica dentro de parametros esperados para la demanda evaluada."
    if percentile >= 10:
        return "bajo", "El desempeno se ubica bajo lo esperado, sugiriendo necesidad de apoyo focalizado."
    return "muy bajo", "El desempeno se ubica significativamente bajo lo esperado, orientando apoyos intensivos y seguimiento."


def interpret_category(value, area):
    text = informed(value).lower()
    if text == "no informado":
        return f"En {area}, el rendimiento se encuentra no informado."
    if any(word in text for word in ["adecuado", "normal", "promedio", "esperado", "alto", "superior", "logrado"]):
        return f"En {area}, la categoria informada sugiere un desempeno funcional o acorde a lo esperado."
    if any(word in text for word in ["desarrollo", "medio", "emergente", "parcial"]):
        return f"En {area}, la categoria informada indica habilidades en proceso de consolidacion."
    if any(word in text for word in ["bajo", "descendido", "insuficiente", "riesgo", "dificultad"]):
        return f"En {area}, la categoria informada evidencia un desempeno descendido y necesidad de apoyo especifico."
    if any(word in text for word in ["muy bajo", "severo", "significativo", "deficit"]):
        return f"En {area}, la categoria informada orienta necesidades de apoyo significativo."
    return f"En {area}, la categoria debe ser integrada al juicio profesional responsable."


def make_psychology_rows(test):
    second_scale = "Visoespacial" if test == "WISC-V" else "Razonamiento perceptual"
    rows = [
        "Comprension verbal",
        second_scale,
        "Razonamiento fluido",
        "Memoria de trabajo",
        "Velocidad de procesamiento",
        "Escala total / CIT",
    ]
    return pd.DataFrame([{column: "" for column in PSYCHOLOGY_COLUMNS} | {"Escala": row} for row in rows])


def make_psychopedagogy_rows():
    return pd.DataFrame(
        [{column: "" for column in PSYCHOPEDAGOGY_COLUMNS} | {"Area evaluada": area} for area in PSYCHOPEDAGOGY_AREAS]
    )


def tabulatest_level(percentage):
    if percentage is None:
        return "no informado"
    if percentage >= 80:
        return "Logrado"
    if percentage >= 50:
        return "Medianamente Logrado"
    return "Por Lograr"


def make_tabulatest_rows(course_label, templates):
    course_key = TABULATEST_LABEL_TO_KEY.get(course_label, course_label)
    template = templates.get(course_key, [])
    rows = []
    for item in template:
        rows.append(
            {
                "Curso": course_label,
                "Área": informed(item.get("area")),
                "Subtest": informed(item.get("subtest")),
                "Puntaje máximo": parse_number(item.get("puntaje_maximo")),
                "Puntaje obtenido": None,
                "Porcentaje de logro": None,
                "Nivel de desempeño": "",
            }
        )
    if not rows:
        rows.append({column: "" for column in TABULATEST_COLUMNS} | {"Curso": course_label})
    return pd.DataFrame(rows, columns=TABULATEST_COLUMNS)


def calculate_tabulatest_records(records):
    calculated = []
    for row in records:
        converted = dict(row)
        obtained = parse_number(row.get("Puntaje obtenido"))
        maximum = parse_number(row.get("Puntaje máximo"))
        if obtained is not None and maximum and maximum > 0:
            percentage = round((obtained / maximum) * 100, 1)
            converted["Porcentaje de logro"] = percentage
            converted["Nivel de desempeño"] = tabulatest_level(percentage)
        else:
            converted["Porcentaje de logro"] = "no informado"
            converted["Nivel de desempeño"] = "no informado"
        calculated.append(converted)
    return calculated


def is_early_childhood_course(course):
    return informed(course) in ["Prekínder", "Kínder"]


def tabulatest_broad_area(row, course):
    text = normalize_text(f"{row.get('Área', '')} {row.get('Subtest', '')}")
    if any(keyword in text for keyword in ["calculo", "numeracion", "problema", "matemat", "numerico"]):
        return "Área Matemática"
    if any(keyword in text for keyword in ["lectura", "lector", "escritura", "ortografia", "grafomotricidad"]):
        return "Área Lectura y Escritura"
    if is_early_childhood_course(course) and any(keyword in text for keyword in ["pensamiento numerico", "numero"]):
        return "Área Matemática"
    return "Área Cognitiva"


def tabulatest_level_phrase(level):
    if level == "Logrado":
        return "adecuado"
    if level == "Medianamente Logrado":
        return "en desarrollo"
    if level == "Por Lograr":
        return "descendido"
    return "no informado"


def tabulatest_area_domain_summary(rows, course):
    text = normalize_text(" ".join(f"{row.get('Área', '')} {row.get('Subtest', '')}" for row in rows))
    if is_early_childhood_course(course):
        domains = []
        if any(keyword in text for keyword in ["percepcion visual", "organizacion perceptiva", "patron", "figura"]):
            domains.append("percepción visual y organización perceptiva")
        if any(keyword in text for keyword in ["percepcion auditiva", "rima", "palabra"]):
            domains.append("percepción auditiva y discriminación verbal")
        if any(keyword in text for keyword in ["razonamiento verbal", "analogia", "descripcion", "lenguaje"]):
            domains.append("lenguaje y razonamiento verbal")
        if any(keyword in text for keyword in ["psicomotricidad", "traza", "copia", "recorta"]):
            domains.append("psicomotricidad fina")
        if any(keyword in text for keyword in ["lectura", "escritura", "grafomotricidad"]):
            domains.append("habilidades precursoras del lenguaje escrito")
        if any(keyword in text for keyword in ["numerico", "numero", "calculo"]):
            domains.append("nociones lógico-matemáticas iniciales")
        return ", ".join(dict.fromkeys(domains)) or "habilidades precursoras del aprendizaje"

    domains = []
    if any(keyword in text for keyword in ["memoria", "atencion"]):
        domains.append("atención y memoria de trabajo")
    if any(keyword in text for keyword in ["series", "clasificacion", "analogias", "razonamiento"]):
        domains.append("razonamiento y establecimiento de relaciones")
    if any(keyword in text for keyword in ["percept", "espacial", "organizacion"]):
        domains.append("organización perceptiva y habilidades visoespaciales")
    if any(keyword in text for keyword in ["lectura", "comprension", "exactitud"]):
        domains.append("comprensión y precisión lectora")
    if any(keyword in text for keyword in ["escritura", "ortografia", "grafomotricidad"]):
        domains.append("escritura, ortografía y control grafomotor")
    if any(keyword in text for keyword in ["calculo", "numeracion", "problema"]):
        domains.append("cálculo, numeración y resolución de problemas")
    return ", ".join(dict.fromkeys(domains)) or "habilidades evaluadas"


def tabulatest_area_paragraph(area, average, level, rows, course):
    level_phrase = tabulatest_level_phrase(level)
    domains = tabulatest_area_domain_summary(rows, course)
    early = is_early_childhood_course(course)

    if area == "Área Cognitiva":
        if early:
            if level == "Logrado":
                return (
                    "En el área cognitiva se aprecia un desempeño adecuado en habilidades precursoras vinculadas "
                    f"con {domains}. Este funcionamiento favorece la exploración, el juego con propósito, la comprensión "
                    "de relaciones simples entre estímulos y la disposición para enfrentar experiencias de aprendizaje propias del nivel."
                )
            if level == "Medianamente Logrado":
                return (
                    "En el área cognitiva se observa un desempeño en desarrollo en habilidades precursoras relacionadas "
                    f"con {domains}. El rendimiento sugiere que la estudiante requiere mediación para organizar la información, "
                    "establecer relaciones entre estímulos y sostener estrategias de resolución acordes a su etapa evolutiva."
                )
            return (
                "En el área cognitiva se evidencia un desempeño descendido en habilidades precursoras asociadas "
                f"a {domains}. Esto puede interferir en la exploración activa, la identificación de patrones, la comprensión "
                "de relaciones básicas y la utilización de recursos cognitivos iniciales para resolver situaciones de aprendizaje."
            )
        if level == "Logrado":
            return (
                "En el área cognitiva se observa un desempeño adecuado, con recursos funcionales para abordar tareas que "
                f"exigen {domains}. Este perfil favorece la comprensión de consignas, la organización de la información y el uso "
                "de estrategias para enfrentar demandas escolares."
            )
        if level == "Medianamente Logrado":
            return (
                "En el área cognitiva se aprecia un desempeño en desarrollo, especialmente en procesos vinculados con "
                f"{domains}. Este funcionamiento puede requerir apoyos para sostener la atención, organizar la información y "
                "aplicar estrategias de resolución de manera consistente."
            )
        return (
            "En el área cognitiva se observa un desempeño descendido en procesos asociados a "
            f"{domains}. Esta condición puede afectar la organización de la información, la flexibilidad para establecer relaciones "
            "y la resolución de tareas escolares que demandan control cognitivo."
        )

    if area == "Área Lectura y Escritura":
        if early:
            if level == "Logrado":
                return (
                    "En el área de lectura y escritura emergente se identifican recursos adecuados en habilidades precursoras "
                    f"relacionadas con {domains}. Estos logros constituyen una base favorable para el desarrollo progresivo del lenguaje, "
                    "la conciencia de símbolos y la aproximación inicial al material escrito desde experiencias lúdicas."
                )
            if level == "Medianamente Logrado":
                return (
                    "En el área de lectura y escritura emergente se observa un desempeño en desarrollo en habilidades precursoras "
                    f"vinculadas con {domains}. Se sugiere fortalecer la mediación mediante experiencias de lenguaje, juego verbal, "
                    "discriminación auditiva y coordinación grafomotriz fina."
                )
            return (
                "En el área de lectura y escritura emergente se evidencia un desempeño descendido en habilidades precursoras "
                f"asociadas a {domains}. Esto puede dificultar la aproximación inicial a símbolos, sonidos del lenguaje, trazos y "
                "relaciones entre imágenes, palabras y significados."
            )
        if level == "Logrado":
            return (
                "En el área de lectura y escritura se aprecia un desempeño adecuado en habilidades asociadas a "
                f"{domains}. Este funcionamiento favorece el acceso a la información escrita, la comprensión de consignas y la expresión "
                "de aprendizajes mediante lenguaje escrito."
            )
        if level == "Medianamente Logrado":
            return (
                "En el área de lectura y escritura se observa un desempeño en desarrollo en procesos vinculados con "
                f"{domains}. Esto puede incidir en la comprensión de textos, la precisión al leer, la organización de ideas y la calidad "
                "de las respuestas escritas."
            )
        return (
            "En el área de lectura y escritura se evidencia un desempeño descendido en habilidades relacionadas con "
            f"{domains}. Esta situación puede limitar el acceso autónomo a la información escrita, la elaboración de respuestas y la "
            "participación en actividades que requieren comprensión y expresión escrita."
        )

    if early:
        if level == "Logrado":
            return (
                "En el área matemática se observan recursos adecuados para abordar nociones lógico-matemáticas iniciales, "
                f"particularmente en {domains}. Este desempeño favorece la exploración de cantidades, secuencias y relaciones simples "
                "a través de experiencias concretas y juego guiado."
            )
        if level == "Medianamente Logrado":
            return (
                "En el área matemática se aprecia un desempeño en desarrollo en nociones lógico-matemáticas iniciales, "
                f"asociadas a {domains}. Se requiere fortalecer la manipulación de material concreto, la comparación, la seriación y "
                "la verbalización de relaciones simples."
            )
        return (
            "En el área matemática se evidencia un desempeño descendido en nociones lógico-matemáticas iniciales relacionadas con "
            f"{domains}. Esto puede afectar la comprensión temprana de cantidad, secuencia y relación, por lo que resulta pertinente "
            "favorecer experiencias concretas, breves y altamente mediadas."
        )
    if level == "Logrado":
        return (
            "En el área matemática se aprecia un desempeño adecuado en habilidades vinculadas con "
            f"{domains}. Este funcionamiento favorece la resolución de situaciones numéricas, la comprensión de procedimientos y la "
            "aplicación de estrategias ante problemas escolares."
        )
    if level == "Medianamente Logrado":
        return (
            "En el área matemática se observa un desempeño en desarrollo en habilidades relacionadas con "
            f"{domains}. Esto puede requerir apoyos para consolidar procedimientos, comprender relaciones numéricas y seleccionar "
            "estrategias pertinentes ante problemas."
        )
    return (
        "En el área matemática se evidencia un desempeño descendido en procesos asociados a "
        f"{domains}. Esta condición puede interferir en la precisión del cálculo, la comprensión de relaciones numéricas y la resolución "
        "de problemas que exigen análisis de información."
    )


def interpret_tabulatest_table(records):
    valid_rows = [row for row in records if informed(row.get("Nivel de desempeño")) != "no informado"]
    if not valid_rows:
        return {
            "analisis_cuantitativo": "La tabla de analisis cuantitativo no presenta puntajes suficientes para calcular porcentajes de logro.",
            "analisis_cualitativo": "No se registran puntajes suficientes para interpretar el desempeno por areas.",
            "sintesis_diagnostica": "Se requiere completar los puntajes obtenidos para formular una sintesis psicopedagogica.",
            "conclusion": "No es posible establecer conclusiones de desempeno sin resultados cuantificables.",
            "recomendaciones": {
                "establecimiento": "- Completar la aplicacion y registro de puntajes antes de tomar decisiones de apoyo.",
                "equipo_aula": "- Revisar la informacion pendiente junto al equipo PIE.",
                "estudiante": "- No aplica hasta contar con resultados suficientes.",
                "familia": "- Informar que el analisis queda pendiente de resultados completos.",
                "otros": "- Validar la tabla con la profesional responsable.",
            },
            "areas": {},
        }

    grouped = {}
    for row in valid_rows:
        area = tabulatest_broad_area(row, row.get("Curso"))
        grouped.setdefault(area, []).append(row)

    area_texts = []
    area_summaries = {}

    for area, rows in grouped.items():
        percentages = [parse_number(row.get("Porcentaje de logro")) for row in rows]
        percentages = [value for value in percentages if value is not None]
        average = round(sum(percentages) / len(percentages), 1) if percentages else None
        global_level = tabulatest_level(average)
        course = informed(rows[0].get("Curso"))
        domains = tabulatest_area_domain_summary(rows, course)
        paragraph = tabulatest_area_paragraph(area, average, global_level, rows, course)

        area_summaries[area] = {
            "promedio": average,
            "nivel_global": global_level,
            "habilidades_consideradas": domains,
            "interpretacion": paragraph,
        }
        area_texts.append(f"{area}\n{paragraph}")

    general_average = round(
        sum(parse_number(row.get("Porcentaje de logro")) or 0 for row in valid_rows) / len(valid_rows),
        1,
    )
    global_level = tabulatest_level(general_average)
    areas_need_support = [
        area for area, details in area_summaries.items() if details.get("nivel_global") in ["Por Lograr", "Medianamente Logrado"]
    ]
    support_focus = ", ".join(areas_need_support) if areas_need_support else "mantencion y enriquecimiento de habilidades ya consolidadas"

    return {
        "analisis_cuantitativo": (
            f"La tabla cuantitativa registra un promedio general de logro de {general_average}%, "
            f"correspondiente a un desempeno global {tabulatest_level_phrase(global_level)}."
        ),
        "analisis_cualitativo": "\n\n".join(area_texts),
        "sintesis_diagnostica": (
            f"El rendimiento global se ubica en un nivel {tabulatest_level_phrase(global_level)}, con necesidad de orientar "
            f"la respuesta psicopedagogica hacia {support_focus}. La interpretacion debe integrarse con antecedentes de aula, "
            "historia escolar y observacion profesional."
        ),
        "conclusion": (
            "Los resultados permiten orientar apoyos psicopedagogicos focalizados por area, priorizando mediaciones ajustadas "
            "al nivel educativo y seguimiento sistematico de la respuesta del estudiante frente a las estrategias implementadas."
        ),
        "recomendaciones": {
            "establecimiento": "- Asegurar tiempos de apoyo PIE y seguimiento periodico de avances por area evaluada.\n- Facilitar coordinacion entre profesional PIE, docentes y familia para ajustar estrategias.",
            "equipo_aula": "- Planificar experiencias graduadas, con modelamiento, apoyos visuales y verificacion de comprension.\n- Ajustar la mediacion segun la respuesta del estudiante y registrar avances de manera sistematica.",
            "estudiante": "- Favorecer metas breves, alcanzables y vinculadas con experiencias de exito.\n- Reforzar estrategias de organizacion, atencion y resolucion de tareas mediante pasos claros.",
            "familia": "- Mantener rutinas breves de acompanamiento en el hogar, utilizando material concreto, lectura compartida o juegos de razonamiento segun el nivel.\n- Reforzar positivamente el esfuerzo y comunicar avances o dificultades al equipo escolar.",
            "otros": "- Revisar estos resultados junto a antecedentes cualitativos, observacion en aula e historia escolar antes de definir apoyos finales.",
        },
        "areas": area_summaries,
    }


def make_speech_rows(test):
    return pd.DataFrame(
        [{column: "" for column in SPEECH_COLUMNS} | {"Test": test, "Area evaluada": area} for area in SPEECH_AREAS]
    )


def dataframe_records(df):
    clean_df = df.fillna("")
    return clean_df.to_dict(orient="records")


def table_to_prompt(records):
    lines = []
    for index, row in enumerate(records, start=1):
        values = "; ".join(f"{key}: {informed(value)}" for key, value in row.items())
        lines.append(f"Fila {index}: {values}")
    return "\n".join(lines) if lines else "no informado"


def html_table_from_records(records, columns):
    valid_records = [
        row for row in records
        if any(informed(row.get(column)) != "no informado" for column in columns)
    ]
    if not valid_records:
        return ""
    header = "".join(f"<th>{escape_html(column)}</th>" for column in columns)
    body_rows = []
    for row in valid_records:
        cells = "".join(f"<td>{escape_html(informed(row.get(column)))}</td>" for column in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def summarize_context(context):
    themes = []
    combined = " ".join(informed(value).lower() for value in context.values())
    if any(word in combined for word in ["atencion", "concentr", "distra"]):
        themes.append("autorregulacion atencional")
    if any(word in combined for word in ["ansiedad", "frustra", "emoc", "nerv"]):
        themes.append("variables socioemocionales que pueden incidir en el desempeno")
    if any(word in combined for word in ["apoyo", "pie", "adecuacion", "refuerzo"]):
        themes.append("presencia o necesidad de apoyos educativos")
    if any(word in combined for word in ["familia", "hogar", "cuidador"]):
        themes.append("factores familiares relevantes para la continuidad de apoyos")
    if not themes:
        return "Los antecedentes contextuales se integran como informacion complementaria para el analisis profesional."
    return "Los antecedentes contextuales sugieren considerar " + ", ".join(themes) + " en la planificacion de apoyos."


def interpret_psychology_table(records):
    interpretations = []
    cit_row = None
    for row in records:
        scale = informed(row.get("Escala"))
        score = parse_number(row.get("Puntaje compuesto"))
        percentile = parse_number(row.get("Percentil"))
        score_range, score_text = classify_standard_score(score)
        percentile_range, percentile_text = classify_percentile(percentile)

        if "cit" in scale.lower() or "escala total" in scale.lower():
            cit_row = (score, score_range, score_text, percentile, percentile_range, percentile_text)
        elif score is not None or percentile is not None:
            interpretations.append(
                f"En {scale}, el puntaje compuesto se clasifica en rango {score_range}. {score_text} "
                f"El percentil se interpreta como desempeno {percentile_range}. {percentile_text}"
            )

    if cit_row:
        score, score_range, score_text, percentile, percentile_range, percentile_text = cit_row
        lead = (
            f"El funcionamiento intelectual global, estimado a partir de la Escala total/CIT, "
            f"se ubica en rango {score_range}. {score_text}"
        )
        if percentile is not None:
            lead += f" El percentil asociado se interpreta como desempeno {percentile_range}. {percentile_text}"
        interpretations.insert(0, lead)
    else:
        interpretations.insert(0, "El CIT o escala total se encuentra no informado, por lo que no se clasifica el funcionamiento intelectual global.")

    return " ".join(interpretations)


def interpret_psychopedagogy_table(records):
    interpretations = []
    strengths = []
    needs = []
    for row in records:
        area = informed(row.get("Area evaluada"))
        percentile = parse_number(row.get("Percentil"))
        level = informed(row.get("Nivel de desempeno"))
        percentile_range, percentile_text = classify_percentile(percentile)
        category_text = interpret_category(level, area)

        if percentile is not None or level != "no informado":
            interpretations.append(
                f"En {area}, el percentil se interpreta como desempeno {percentile_range}. "
                f"{percentile_text} {category_text}"
            )
            if percentile_range in ["alto", "esperado"] or any(word in level.lower() for word in ["adecuado", "esperado", "alto", "logrado"]):
                strengths.append(area.lower())
            if percentile_range in ["bajo", "muy bajo"] or any(word in level.lower() for word in ["bajo", "descendido", "riesgo", "insuficiente"]):
                needs.append(area.lower())

    if not interpretations:
        interpretations.append("Los resultados psicopedagogicos se encuentran no informados.")

    strengths_text = ", ".join(strengths) if strengths else "no informado"
    needs_text = ", ".join(needs) if needs else "no informado"
    return " ".join(interpretations), strengths_text, needs_text


def interpret_speech_table(records):
    interpretations = []
    strengths = []
    needs = []
    for row in records:
        area = informed(row.get("Area evaluada"))
        percentile_ds = informed(row.get("Percentil / DS"))
        performance = informed(row.get("Rendimiento"))
        category_text = interpret_category(performance, area.lower())
        percentile = parse_number(percentile_ds)
        percentile_range, percentile_text = classify_percentile(percentile)

        if performance != "no informado" or percentile_ds != "no informado":
            if percentile is not None:
                interpretations.append(
                    f"En {area}, el indicador normativo se interpreta como desempeno {percentile_range}. "
                    f"{percentile_text} {category_text}"
                )
            else:
                interpretations.append(category_text)

            if any(word in performance.lower() for word in ["adecuado", "normal", "esperado", "alto"]):
                strengths.append(area.lower())
            if any(word in performance.lower() for word in ["bajo", "descendido", "riesgo", "alterado", "deficit"]):
                needs.append(area.lower())

    if not interpretations:
        interpretations.append("Los resultados fonoaudiologicos especificos se encuentran no informados.")

    strengths_text = ", ".join(strengths) if strengths else "no informado"
    needs_text = ", ".join(needs) if needs else "no informado"
    return " ".join(interpretations), strengths_text, needs_text


def interpretar_resultados(test, tabla_resultados):
    if test in ["WISC-V", "WAIS-IV", "ICAP"]:
        quantitative = interpret_psychology_table(tabla_resultados)
        return {
            "analisis_cuantitativo": quantitative,
            "analisis_cualitativo": "El perfil psicologico escolar se interpreta integrando puntajes derivados disponibles, dispersion entre escalas y antecedentes contextuales, sin copiar observaciones literalmente.",
            "sintesis_diagnostica": "Los resultados orientan una comprension tecnica del funcionamiento cognitivo y adaptativo en contexto escolar, considerando recursos y necesidades de apoyo.",
            "conclusion": "La interpretacion debe ser validada por la profesional responsable y utilizada como insumo para decisiones educativas.",
            "recomendaciones": "Ajustar mediaciones, tiempos, instrucciones y seguimiento de acuerdo con el perfil observado.",
        }

    if test == "Tabulatest":
        return interpret_tabulatest_table(tabla_resultados)

    if test in ["Evalua", "CLP", "Pruebas pedagogicas internas"]:
        quantitative, strengths, needs = interpret_psychopedagogy_table(tabla_resultados)
        return {
            "analisis_cuantitativo": quantitative,
            "analisis_cualitativo": f"Las fortalezas academicas relativas se observan en: {strengths}. Las necesidades de apoyo se focalizan en: {needs}.",
            "sintesis_diagnostica": "El perfil psicopedagogico permite identificar barreras y facilitadores para el aprendizaje, orientando mediaciones pedagogicas especificas.",
            "conclusion": "El desempeno debe analizarse en conjunto con historia escolar, apoyos actuales y respuesta a la intervencion.",
            "recomendaciones": "Implementar apoyos graduados, seguimiento de progreso y estrategias explicitas en las areas descendidas.",
        }

    quantitative, strengths, needs = interpret_speech_table(tabla_resultados)
    return {
        "analisis_cuantitativo": quantitative,
        "analisis_cualitativo": f"Las fortalezas comunicativas relativas se observan en: {strengths}. Las necesidades de apoyo se focalizan en: {needs}.",
        "sintesis_diagnostica": "El perfil fonoaudiologico escolar se organiza en torno al lenguaje, habla, comunicacion funcional y participacion comunicativa.",
        "conclusion": "Los resultados orientan apoyos fonoaudiologicos escolares que deben ser validados por la profesional responsable.",
        "recomendaciones": "Favorecer apoyos visuales, verificacion de comprension, tiempo de respuesta y oportunidades comunicativas funcionales.",
    }


def build_prompt(data):
    title = REPORT_TITLES[data["professional"]]
    interpreted = interpretar_resultados(data["test"], data["records"])
    tabulatest_rules = ""
    if data.get("test") == "Tabulatest":
        tabulatest_rules = """
Reglas especificas Tabulatest:
- No enumeres todos los subtests en un parrafo largo.
- Mantén el analisis cuantitativo como referencia a la tabla.
- Redacta el analisis cualitativo por areas.
- Separa sugerencias en establecimiento, equipo de aula, estudiante, familia y otros.
- Si el curso es Prekinder o Kinder, usa lenguaje pertinente a educacion parvularia.
""".strip()
    return f"""
Actua como profesional {data["professional"]} con experiencia en Programa de Integracion Escolar (PIE) en Chile.

Redacta un {title} profesional, claro y tecnico.

Reglas obligatorias:
- No copies literalmente lo ingresado por el usuario.
- Interpreta puntajes, percentiles, DS, categorias y niveles de desempeno.
- Nunca inventes puntajes ni resultados.
- Si un dato esta vacio, escribe "no informado".
- Usa lenguaje tecnico propio de {data["professional"]}.
- La interpretacion final debe ser validada por la profesional responsable.
{tabulatest_rules}

Datos del estudiante:
- Nombre: {informed(data["student_name"])}
- Curso: {informed(data["course"])}
- Edad: {informed(data["age"])}
- Diagnostico o antecedente principal: {informed(data["diagnosis"])}

Profesional: {data["professional"]}
Test aplicado: {data["test"]}

Tabla de resultados:
{table_to_prompt(data["records"])}

Interpretacion automatica calculada por la aplicacion:
- Analisis cuantitativo: {interpreted["analisis_cuantitativo"]}
- Analisis cualitativo: {interpreted["analisis_cualitativo"]}
- Sintesis diagnostica: {interpreted["sintesis_diagnostica"]}
- Conclusion: {interpreted["conclusion"]}
- Recomendaciones: {interpreted["recomendaciones"]}

Contextualizacion del estudiante:
- Antecedentes escolares relevantes: {informed(data["context"]["school_history"])}
- Antecedentes familiares relevantes: {informed(data["context"]["family_history"])}
- Observacion conductual durante la evaluacion: {informed(data["context"]["behavior_observation"])}
- Fortalezas observadas: {informed(data["context"]["observed_strengths"])}
- Necesidades de apoyo: {informed(data["context"]["support_needs"])}
- Apoyos actuales: {informed(data["context"]["current_supports"])}
- Observaciones del profesional: {informed(data["context"]["professional_observations"])}

Genera un informe personalizado con:
1. Identificacion
2. Contextualizacion del estudiante
3. Instrumento aplicado
4. Analisis e interpretacion de resultados
5. Sintesis profesional
6. Conclusiones
7. Sugerencias para establecimiento
8. Sugerencias para equipo de aula y/o familia
""".strip()


def generate_openai_report(client, model, data):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Eres una asistente de redaccion profesional para informes PIE en Chile. "
                    "Debes interpretar datos objetivos sin inventar resultados ni copiar observaciones literalmente."
                ),
            },
            {"role": "user", "content": build_prompt(data)},
        ],
    )
    return response.output_text


def tabulatest_records_for_ai(resultados):
    records = []
    for row in resultados:
        records.append(
            {
                "Área": informed(row.get("Área")),
                "Subtest": informed(row.get("Subtest")),
                "Puntaje máximo": informed(row.get("Puntaje máximo")),
                "Puntaje obtenido": informed(row.get("Puntaje obtenido")),
                "Porcentaje de logro": informed(row.get("Porcentaje de logro")),
                "Nivel de desempeño": informed(row.get("Nivel de desempeño")),
            }
        )
    return records


def tabulatest_area_summaries_for_ai(resultados):
    interpreted = interpretar_resultados("Tabulatest", resultados)
    summaries = []
    for area, details in interpreted.get("areas", {}).items():
        summaries.append(
            {
                "Área": area,
                "Promedio por área": informed(details.get("promedio")),
                "Nivel de desempeño por área": informed(details.get("nivel_global")),
                "Habilidades consideradas": informed(details.get("habilidades_consideradas")),
            }
        )
    return summaries


def tabulatest_level_guidance(course):
    course_text = informed(course)
    if course_text in ["Prekínder", "Kínder"]:
        return {
            "nivel_educativo": "Educación parvularia",
            "enfatizar": [
                "desarrollo de habilidades precursoras",
                "desarrollo psicomotor",
                "lenguaje",
                "funciones cognitivas básicas",
                "juego y exploración",
            ],
            "evitar_mencionar": [
                "currículo formal",
                "producción escrita",
                "autonomía curricular",
            ],
        }
    if course_text in ["I° Medio", "II° Medio"]:
        return {
            "nivel_educativo": "Enseñanza media",
            "enfatizar": [
                "procesos de aprendizaje complejos",
                "comprensión de información",
                "organización y autonomía académica",
                "funciones cognitivas aplicadas al aprendizaje",
            ],
            "evitar_mencionar": [],
        }
    return {
        "nivel_educativo": "Educación básica",
        "enfatizar": [
            "comprensión lectora",
            "escritura",
            "pensamiento matemático",
            "funciones cognitivas",
        ],
        "evitar_mencionar": [],
    }


def build_tabulatest_openai_prompt(resultados, datos_estudiante, contexto):
    course = informed(datos_estudiante.get("course"))
    payload = {
        "datos_estudiante": {
            "nombre": informed(datos_estudiante.get("student_name")),
            "curso": course,
            "edad": informed(datos_estudiante.get("age")),
            "diagnostico_o_antecedente_principal": informed(datos_estudiante.get("diagnosis")),
        },
        "contexto_estudiante": {
            "antecedentes_escolares_relevantes": informed(contexto.get("school_history")),
            "antecedentes_familiares_relevantes": informed(contexto.get("family_history")),
            "observacion_conductual_durante_la_evaluacion": informed(contexto.get("behavior_observation")),
            "fortalezas_observadas": informed(contexto.get("observed_strengths")),
            "necesidades_de_apoyo": informed(contexto.get("support_needs")),
            "apoyos_actuales": informed(contexto.get("current_supports")),
            "observaciones_del_profesional": informed(contexto.get("professional_observations")),
        },
        "tabla_tabulatest_calculada": tabulatest_records_for_ai(resultados),
        "resumen_por_area": tabulatest_area_summaries_for_ai(resultados),
        "orientaciones_por_nivel_educativo": tabulatest_level_guidance(course),
    }
    return f"""
Redacta un informe psicopedagogico profesional para PIE Chile. No enumeres los resultados. Interpreta los desempenos observados. Relaciona los resultados con posibles implicancias en el aprendizaje. Utiliza lenguaje tecnico psicopedagogico. Adapta el lenguaje al nivel educativo del estudiante.

Debes basarte exclusivamente en los datos entregados. No inventes antecedentes, puntajes ni diagnosticos. Interpreta los resultados por area, considerando porcentajes de logro, promedio por area y niveles de desempeno por area. Usa lenguaje tecnico, claro, respetuoso y propio de informes psicopedagogicos PIE reales.

El informe debe incluir exactamente estas secciones:

I. Identificacion
II. Contextualizacion del estudiante
III. Instrumento aplicado
IV. Analisis cuantitativo
V. Analisis cualitativo por areas
VI. Sintesis profesional
VII. Conclusion
VIII. Sugerencias:
- Al establecimiento educacional
- Al equipo de aula
- Al estudiante
- A la familia
- Otros

Reglas obligatorias:
- No copies la tabla como parrafo largo.
- No enumeres subtests como listado de resultados.
- Redacta por areas amplias: Area Cognitiva, Area Lectura y Escritura, Area Matematica.
- Cada area debe quedar en parrafos profesionales e integrados, no en listas.
- Interpreta los desempenos; no repitas mecanicamente puntajes ni porcentajes.
- Relaciona cada area con implicancias posibles en el aprendizaje y la participacion escolar.
- Omite subtests sin puntaje obtenido.
- No inventes puntajes faltantes ni completes datos ausentes.
- Evita frases genericas, repetitivas o de plantilla.
- No uses expresiones como "el perfil Tabulatest permite identificar" ni introducciones vacias.
- No uses expresiones como "Subtests logrados", "Subtests por lograr" ni "Este desempeno entrega antecedentes relevantes".
- Mantén un lenguaje psicopedagogico profesional, pertinente a informes PIE de Chile.

Adecuacion por nivel educativo:
- Si el curso corresponde a Prekinder o Kinder, habla de habilidades precursoras, desarrollo psicomotor, lenguaje, funciones cognitivas basicas, juego y exploracion. No menciones curriculo formal, produccion escrita ni autonomia curricular.
- Si el curso corresponde a Educacion Basica, considera comprension lectora, escritura, pensamiento matematico y funciones cognitivas.
- Si el curso corresponde a Ensenanza Media, considera procesos de aprendizaje complejos, comprension de informacion, organizacion y autonomia academica.

Datos disponibles en formato JSON:
{json.dumps(payload, ensure_ascii=False, indent=2)}
""".strip()


def generate_openai_tabulatest_report(client, model, resultados, datos_estudiante, contexto):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Eres una profesional experta en redaccion de informes psicopedagogicos PIE en Chile. "
                    "Debes interpretar datos objetivos sin inventar antecedentes, puntajes ni diagnosticos. "
                    "Tu redaccion debe ser contextualizada, tecnica y fluida, similar a un informe psicopedagogico real. "
                    "Evita plantillas genericas, listas extensas de subtests y repeticiones mecanicas de resultados."
                ),
            },
            {"role": "user", "content": build_tabulatest_openai_prompt(resultados, datos_estudiante, contexto)},
        ],
    )
    return response.output_text


def local_psychology_report(data):
    interpreted = interpretar_resultados(data["test"], data["records"])
    context_summary = summarize_context(data["context"])
    return f"""
1. Identificacion

Estudiante: {informed(data["student_name"])}
Curso: {informed(data["course"])}
Edad: {informed(data["age"])}
Antecedente principal: {informed(data["diagnosis"])}

2. Contextualizacion del estudiante

{context_summary} Se consideran antecedentes escolares, familiares, apoyos actuales y observacion conductual como variables que pueden modular el desempeno evaluativo y la participacion escolar.

3. Instrumento aplicado

Se informa la aplicacion de {data["test"]}. Los datos vacios se consignan como no informado y no se incorporan puntajes no ingresados.

4. Analisis e interpretacion de resultados

Analisis cuantitativo: {interpreted["analisis_cuantitativo"]}

Analisis cualitativo: {interpreted["analisis_cualitativo"]}

5. Sintesis profesional

{interpreted["sintesis_diagnostica"]}

6. Conclusiones

{interpreted["conclusion"]}

7. Sugerencias para establecimiento

{interpreted["recomendaciones"]}

8. Sugerencias para equipo de aula y/o familia

- Entregar instrucciones claras, secuenciadas y verificables.
- Reforzar rutinas, organizacion y estrategias de autorregulacion.
- Favorecer acompanamiento proporcional al nivel de autonomia del estudiante.
""".strip()


def local_psychopedagogy_report(data):
    interpreted = interpretar_resultados(data["test"], data["records"])
    context_summary = summarize_context(data["context"])
    return f"""
1. Identificacion

Estudiante: {informed(data["student_name"])}
Curso: {informed(data["course"])}
Edad: {informed(data["age"])}
Antecedente principal: {informed(data["diagnosis"])}

2. Contextualizacion del estudiante

{context_summary} La informacion contextual permite comprender barreras, facilitadores y condiciones de participacion en el proceso de aprendizaje.

3. Instrumento aplicado

Se informa la aplicacion de {data["test"]}. Todo puntaje o nivel no ingresado se considera no informado.

4. Analisis e interpretacion de resultados

Analisis cuantitativo: {interpreted["analisis_cuantitativo"]}

Analisis cualitativo: {interpreted["analisis_cualitativo"]}

5. Sintesis profesional

{interpreted["sintesis_diagnostica"]}

6. Conclusiones

{interpreted["conclusion"]} La interpretacion final debe ser validada por la profesional responsable.

7. Sugerencias para establecimiento

{interpreted["recomendaciones"]}

8. Sugerencias para equipo de aula y/o familia

- Fragmentar tareas extensas y verificar comprension de instrucciones.
- Reforzar explicitamente estrategias de lectura, escritura, calculo u organizacion segun corresponda.
- Mantener rutinas de estudio y retroalimentacion positiva.
""".strip()


def generar_informe_tabulatest(resultados, datos_estudiante, contexto, model=DEFAULT_MODEL):
    client = get_openai_client()
    if client is not None:
        return generate_openai_tabulatest_report(client, model, resultados, datos_estudiante, contexto)

    interpreted = interpretar_resultados("Tabulatest", resultados)
    context_summary = summarize_context(contexto)
    course = informed(datos_estudiante.get("course"))
    early_childhood = course in ["Prekínder", "Kínder"]
    level_context = (
        "educacion parvularia, considerando habilidades precursoras del aprendizaje, exploracion guiada, desarrollo psicomotor, lenguaje y disposicion frente a experiencias pedagogicas"
        if early_childhood
        else "educacion escolar, considerando demandas curriculares, autonomia academica y desempeno en tareas instrumentales"
    )
    recommendations = interpreted["recomendaciones"]

    return f"""
1. Identificacion

Estudiante: {informed(datos_estudiante.get("student_name"))}
Curso: {course}
Edad: {informed(datos_estudiante.get("age"))}
Antecedente principal: {informed(datos_estudiante.get("diagnosis"))}

2. Contextualizacion del estudiante

{context_summary} La interpretacion se realiza en el marco de {level_context}, integrando resultados objetivos con antecedentes escolares y observacion profesional.

3. Instrumento aplicado

Se informa aplicacion de Tabulatest. La tabla de analisis cuantitativo se mantiene como registro de puntaje obtenido, puntaje maximo, porcentaje de logro y nivel de desempeno por subtest.

4. Analisis cuantitativo

{interpreted["analisis_cuantitativo"]}

5. Analisis cualitativo por areas

{interpreted["analisis_cualitativo"]}

6. Sintesis

{interpreted["sintesis_diagnostica"]}

7. Conclusion

{interpreted["conclusion"]} La interpretacion final debe ser validada por la profesional responsable.

8. Sugerencias

Al establecimiento educacional
{recommendations["establecimiento"]}

Al equipo de aula
{recommendations["equipo_aula"]}

Al estudiante
{recommendations["estudiante"]}

A la familia
{recommendations["familia"]}

Otros
{recommendations["otros"]}
""".strip()


def local_speech_report(data):
    interpreted = interpretar_resultados(data["test"], data["records"])
    context_summary = summarize_context(data["context"])
    return f"""
1. Identificacion

Estudiante: {informed(data["student_name"])}
Curso: {informed(data["course"])}
Edad: {informed(data["age"])}
Antecedente principal: {informed(data["diagnosis"])}

2. Contextualizacion del estudiante

{context_summary} Estos antecedentes se integran al analisis del desempeno comunicativo-linguistico en contexto escolar.

3. Instrumento aplicado

Se informa la aplicacion de {data["test"]}. Los resultados no ingresados se consignan como no informado.

4. Analisis e interpretacion de resultados

Analisis cuantitativo: {interpreted["analisis_cuantitativo"]}

Analisis cualitativo: {interpreted["analisis_cualitativo"]}

5. Sintesis profesional

{interpreted["sintesis_diagnostica"]}

6. Conclusiones

{interpreted["conclusion"]} La interpretacion final debe ser validada por la profesional responsable.

7. Sugerencias para establecimiento

{interpreted["recomendaciones"]}

8. Sugerencias para equipo de aula y/o familia

- Apoyar instrucciones con claves visuales y verificacion de comprension.
- Dar tiempo suficiente para responder y organizar el discurso.
- Reforzar vocabulario, narracion y conversacion funcional en contextos naturales.
""".strip()


def generate_local_report(data):
    if data["professional"] == "Psicopedagogía" and data["test"] == "Tabulatest":
        datos_estudiante = {
            "student_name": data.get("student_name"),
            "course": data.get("course"),
            "age": data.get("age"),
            "diagnosis": data.get("diagnosis"),
        }
        return generar_informe_tabulatest(data["records"], datos_estudiante, data["context"])
    if data["professional"] == "Psicología":
        return local_psychology_report(data)
    if data["professional"] == "Psicopedagogía":
        return local_psychopedagogy_report(data)
    return local_speech_report(data)


def build_word_html(data, generated_text):
    safe_generated = escape_html(generated_text).replace("\n", "<br>")
    title = REPORT_TITLES[data["professional"]]
    notice_text = (
        "El informe generado por IA debe ser revisado y validado por la profesional responsable."
        if data.get("generated_with_ai")
        else "Este texto debe ser revisado, ajustado y validado antes de usarse. La interpretacion final debe ser validada por la profesional responsable."
    )
    tabulatest_table = ""
    if data.get("test") == "Tabulatest":
        tabulatest_table = (
            "<h2>Tabla de analisis cuantitativo Tabulatest</h2>"
            + html_table_from_records(data.get("records", []), TABULATEST_COLUMNS)
        )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{escape_html(title)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #111827; line-height: 1.45; }}
    h1 {{ text-align: center; font-size: 20pt; }}
    h2 {{ font-size: 13pt; margin-top: 22px; border-bottom: 1px solid #999; padding-bottom: 4px; }}
    p {{ margin: 7px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin: 10px 0 18px; }}
    th, td {{ border: 1px solid #777; padding: 6px; text-align: left; vertical-align: top; }}
    th {{ background: #eeeeee; }}
    .notice {{ border: 1px solid #b45309; background: #fff7ed; padding: 10px; }}
  </style>
</head>
<body>
  <h1>{escape_html(title)}</h1>
  <p><strong>Fecha:</strong> {date.today().strftime("%d-%m-%Y")}</p>
  <p><strong>Estudiante:</strong> {escape_html(informed(data["student_name"]))}</p>
  <p><strong>Curso:</strong> {escape_html(informed(data["course"]))}</p>
  <p><strong>Edad:</strong> {escape_html(informed(data["age"]))}</p>
  <p><strong>Profesional:</strong> {escape_html(data["professional"])}</p>
  <p><strong>Test aplicado:</strong> {escape_html(data["test"])}</p>
  <p class="notice"><strong>Advertencia:</strong> {escape_html(notice_text)}</p>
  {tabulatest_table}
  <h2>Informe generado</h2>
  <p>{safe_generated}</p>
</body>
</html>"""


def render_selector():
    st.markdown(
        """
        <style>
        div[role="radiogroup"] {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }
        div[role="radiogroup"] label {
            border: 1px solid #d0d5dd;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
            min-height: 58px;
        }
        div[role="radiogroup"] label:has(input:checked) {
            border-color: #265c56;
            background: #edf7f4;
            color: #183f3b;
            font-weight: 700;
        }
        @media (max-width: 760px) {
            div[role="radiogroup"] {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    return st.radio("Seleccionar profesional", PROFESSIONALS, horizontal=True)


def render_student_fields():
    st.subheader("Datos generales")
    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Nombre del estudiante")
    with col2:
        course = st.text_input("Curso")
    with col3:
        age = st.text_input("Edad")
    diagnosis = st.text_area("Diagnostico o antecedente principal", height=80)
    return {
        "student_name": student_name,
        "course": course,
        "age": age,
        "diagnosis": diagnosis,
    }


def render_context_fields():
    st.subheader("Contextualizacion del estudiante")
    col1, col2 = st.columns(2)
    with col1:
        school_history = st.text_area("Antecedentes escolares relevantes", height=90)
        behavior_observation = st.text_area("Observacion conductual durante la evaluacion", height=90)
        support_needs = st.text_area("Necesidades de apoyo", height=90)
        professional_observations = st.text_area("Observaciones del profesional", height=90)
    with col2:
        family_history = st.text_area("Antecedentes familiares relevantes", height=90)
        observed_strengths = st.text_area("Fortalezas observadas", height=90)
        current_supports = st.text_area("Apoyos actuales", height=90)
    return {
        "school_history": school_history,
        "family_history": family_history,
        "behavior_observation": behavior_observation,
        "observed_strengths": observed_strengths,
        "support_needs": support_needs,
        "current_supports": current_supports,
        "professional_observations": professional_observations,
    }


def render_results_table(professional, test):
    st.subheader("Tabla de resultados")
    if professional == "Psicopedagogía" and test == "Tabulatest":
        templates, template_error = load_tabulatest_templates()
        if template_error:
            st.error(template_error)
            st.info("Revise el archivo JSON y vuelva a cargar la aplicacion. La app no se detendra.")
            empty_df = pd.DataFrame([{column: "" for column in TABULATEST_COLUMNS}], columns=TABULATEST_COLUMNS)
            return empty_df, ""

        st.caption(f"Leyendo plantilla desde: {TABULATEST_TEMPLATE_PATH}")
        st.caption("Modulo Tabulatest: la profesional solo ingresa el puntaje obtenido.")
        edited, selected_course = render_tabulatest_course_tabs(templates)
        return edited, selected_course

    if has_loaded_baremo(test):
        st.success("Baremo cargado para este test. La app intentara completar puntajes derivados al generar.")
    else:
        st.info(
            "Aun no hay baremo cargado para este test. Puede ingresar manualmente puntaje compuesto, "
            "percentil, intervalo de confianza, rango descriptivo y nivel de desempeno en la tabla."
        )

    if professional == "Psicología":
        base_df = make_psychology_rows(test)
        columns = PSYCHOLOGY_COLUMNS
    elif professional == "Psicopedagogía":
        base_df = make_psychopedagogy_rows()
        columns = PSYCHOPEDAGOGY_COLUMNS
    else:
        base_df = make_speech_rows(test)
        columns = SPEECH_COLUMNS

    edited = st.data_editor(
        base_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_order=columns,
        key=f"table_{professional}_{test}",
    )
    return edited, ""


def render_baremos_admin():
    st.title("Gestion de Baremos")
    st.warning(
        "Seccion exclusiva para administradores. Ingrese solo baremos validados por el equipo PIE. "
        "La aplicacion no usa internet ni inventa conversiones."
    )
    st.subheader("Tabulatest")
    st.caption("Crear baremos por curso, area y test. Los datos se guardan en baremos/psicopedagogia/tabulatest.json.")

    admin_df = load_tabulatest_admin_df()
    edited_df = st.data_editor(
        admin_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_order=TABULATEST_ADMIN_COLUMNS,
        key="admin_tabulatest_baremos",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Guardar baremo Tabulatest", type="primary"):
            saved = save_tabulatest_baremos(edited_df)
            st.success(f"Baremo Tabulatest guardado con {saved} conversiones.")
    with col2:
        st.info(
            "Para aplicar una conversion, deben coincidir curso, area evaluada y puntaje bruto. "
            "Si no coinciden, la profesional podra ingresar los derivados manualmente."
        )

    if TEST_BAREMO_FILES["Tabulatest"].exists():
        st.download_button(
            "Descargar JSON Tabulatest",
            data=TEST_BAREMO_FILES["Tabulatest"].read_text(encoding="utf-8"),
            file_name="tabulatest.json",
            mime="application/json",
        )


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=":memo:", layout="wide")
    with st.sidebar:
        section = st.radio("Seccion", ["Informes", "Gestion de Baremos"])

    if section == "Gestion de Baremos":
        render_baremos_admin()
        return

    st.title(APP_TITLE)
    st.warning(
        "El informe generado debe ser revisado por la profesional responsable. "
        "La aplicacion interpreta datos objetivos, pero no reemplaza el juicio profesional."
    )

    professional = render_selector()
    if st.session_state.get("active_professional") != professional:
        st.session_state["active_professional"] = professional
        st.session_state.pop("generated_report", None)
        st.session_state.pop("report_data", None)

    with st.sidebar:
        st.header("Configuracion")
        model = st.text_input("Modelo OpenAI", value=DEFAULT_MODEL)
        if os.getenv("OPENAI_API_KEY"):
            st.success("OPENAI_API_KEY detectada.")
        else:
            st.info("No se detecto OPENAI_API_KEY. Se usaran plantillas locales.")

    student_data = render_student_fields()
    test = st.selectbox("Test aplicado", TESTS_BY_PROFESSIONAL[professional])
    edited_table, tabulatest_course = render_results_table(professional, test)
    raw_records = dataframe_records(edited_table)

    if professional == "Psicopedagogía" and test == "Tabulatest":
        preview_records = calculate_tabulatest_records(raw_records)
        st.subheader("Tabla de analisis cuantitativo Tabulatest")
        st.dataframe(pd.DataFrame(preview_records), use_container_width=True)

    context = render_context_fields()
    submitted = st.button("Generar informe", type="primary")

    if submitted:
        if professional == "Psicopedagogía" and test == "Tabulatest":
            converted_records = calculate_tabulatest_records(raw_records)
            any_conversion = True
            if informed(tabulatest_course) != "no informado":
                student_data["course"] = tabulatest_course
        else:
            converted_records, any_conversion = apply_baremos_to_records(
                test,
                student_data["age"],
                student_data["course"],
                raw_records,
            )
        data = {
            "professional": professional,
            "test": test,
            "records": converted_records,
            "context": context,
            **student_data,
        }

        missing = [field for field in ["student_name", "course", "age", "diagnosis"] if not informed(data[field]) != "no informado"]
        if missing:
            st.error("Completa al menos nombre, curso, edad y diagnostico o antecedente principal.")
            st.stop()

        if professional == "Psicopedagogía" and test == "Tabulatest":
            st.success("Tabla Tabulatest calculada automaticamente.")
            st.dataframe(pd.DataFrame(converted_records), use_container_width=True)
        elif has_loaded_baremo(test) and any_conversion:
            st.success("Se aplicaron conversiones automaticas desde el baremo cargado.")
            st.dataframe(pd.DataFrame(converted_records), use_container_width=True)
        else:
            st.info(
                "No existe baremo cargado para este test/edad/curso. Se usaran los puntajes derivados "
                "ingresados manualmente; los campos vacios quedaran como no informado."
            )

        client = get_openai_client()
        if professional == "Psicopedagogía" and test == "Tabulatest":
            datos_estudiante_tabulatest = {
                "student_name": data.get("student_name"),
                "course": data.get("course"),
                "age": data.get("age"),
                "diagnosis": data.get("diagnosis"),
            }
            if client is None:
                with st.spinner("Generando informe Tabulatest con interpretacion local..."):
                    generated_text = generar_informe_tabulatest(
                        data["records"],
                        datos_estudiante_tabulatest,
                        data["context"],
                        model.strip() or DEFAULT_MODEL,
                    )
                st.info("Informe Tabulatest generado en modo local, sin API Key.")
                data["generated_with_ai"] = False
            else:
                with st.spinner("Generando informe Tabulatest con OpenAI..."):
                    try:
                        generated_text = generar_informe_tabulatest(
                            data["records"],
                            datos_estudiante_tabulatest,
                            data["context"],
                            model.strip() or DEFAULT_MODEL,
                        )
                    except Exception as exc:
                        st.error("No se pudo generar el informe Tabulatest con OpenAI.")
                        st.exception(exc)
                        st.stop()
                data["generated_with_ai"] = True
                st.warning("El informe generado por IA debe ser revisado y validado por la profesional responsable.")
        elif client is None:
            with st.spinner("Generando informe con interpretacion local..."):
                generated_text = generate_local_report(data)
            st.info("Informe generado en modo local, sin API Key.")
            data["generated_with_ai"] = False
        else:
            with st.spinner("Generando informe con OpenAI..."):
                try:
                    generated_text = generate_openai_report(client, model.strip() or DEFAULT_MODEL, data)
                except Exception as exc:
                    st.error("No se pudo generar el informe con OpenAI.")
                    st.exception(exc)
                    st.stop()
            data["generated_with_ai"] = True

        st.session_state["generated_report"] = generated_text
        st.session_state["report_data"] = data

    if "generated_report" in st.session_state:
        st.subheader("Informe generado")
        if st.session_state.get("report_data", {}).get("generated_with_ai"):
            st.warning("El informe generado por IA debe ser revisado y validado por la profesional responsable.")
        edited_report = st.text_area(
            "Texto editable",
            value=st.session_state["generated_report"],
            height=540,
        )
        st.session_state["generated_report"] = edited_report

        report_data = st.session_state["report_data"]
        file_student = informed(report_data["student_name"]).replace(" ", "_")
        file_professional = report_data["professional"].lower()
        word_html = build_word_html(report_data, edited_report)
        st.download_button(
            "Descargar informe editable en Word",
            data=word_html.encode("utf-8"),
            file_name=f"informe_{file_professional}_{file_student}.doc",
            mime="application/msword",
        )


if __name__ == "__main__":
    main()
