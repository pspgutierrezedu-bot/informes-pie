# Generador de informes PIE

Aplicación web en Python con Streamlit para generar informes psicopedagógicos, psicológicos y fonoaudiológicos para contexto PIE en Chile. La app puede funcionar con OpenAI si existe `OPENAI_API_KEY`, y mantiene generación local como respaldo cuando no hay API key.

## Características

- Formulario de datos del estudiante y contextualización.
- Módulo Tabulatest con tablas por curso desde `data/tabulatest/tabulatest_templates.json`.
- Cálculo automático de porcentaje de logro y nivel de desempeño.
- Generación de informe editable.
- Exportación a Word.
- Soporte para uso con o sin `OPENAI_API_KEY`.

## Estructura del proyecto

```text
.
app.py
requirements.txt
README.md
render.yaml

data/
└── tabulatest/
    ├── tabulatest_templates.json
    └── tabulatest_templates.xlsx

baremos/

## Variable de entorno

La aplicación lee la API key desde:

```text
OPENAI_API_KEY
```

No escribas la API key dentro de `app.py`, `README.md`, ni ningún archivo del repositorio. Si la variable no existe, la aplicación usa plantillas locales.

## Ejecución local

```bash
python -m streamlit run app.py
```

## Despliegue en Streamlit Cloud

1. Sube este proyecto a un repositorio de GitHub.
2. En Streamlit Cloud, crea una nueva app desde ese repositorio.
3. Selecciona `app.py` como archivo principal.
4. En la configuración de secrets de Streamlit Cloud, agrega:

```toml
OPENAI_API_KEY = "tu_api_key"
```

5. Si no configuras `OPENAI_API_KEY`, la app igualmente funcionará con generación local.

## Despliegue en Render

Este repositorio incluye `render.yaml` para crear un Web Service.

Configuración esperada:

- Build command: `pip install -r requirements.txt`
- Start command: `python -m streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

En Render, configura la variable de entorno:

```text
OPENAI_API_KEY=tu_api_key
```

Si no agregas esa variable, la app usará plantillas locales.

## Datos Tabulatest

Las plantillas de curso se leen desde:

```text
data/tabulatest/tabulatest_templates.json
```

Para agregar cursos o subtests, modifica ese JSON manteniendo la estructura existente. La app carga las plantillas desde el repositorio, por lo que el archivo debe estar incluido en GitHub.

## Nota profesional

Todo informe generado por IA o por plantillas locales debe ser revisado, ajustado y validado por la profesional responsable antes de su uso.
