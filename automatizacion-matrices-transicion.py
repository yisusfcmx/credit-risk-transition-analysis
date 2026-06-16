import os
import time
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.colab import auth
import gspread
from google.auth import default
from gspread_dataframe import set_with_dataframe

# ==========================================
# CONFIGURACIÓN Y VARIABLES DE ENTORNO
# ==========================================
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "tu_proyecto_gcp")
BQ_SOURCE_TABLE = os.getenv("BQ_SOURCE_TABLE", "dataset.tabla_origen")
SHEET_REPORT_NAME = os.getenv("SHEET_REPORT_NAME", "REPORTE_MATRIZ_TRANSICION_MOCK")
START_DATE = os.getenv("START_DATE", "2025-12-01")
END_DATE = os.getenv("END_DATE", "2026-02-28")

# ==========================================
# AUTENTICACIÓN Y CONEXIÓN
# ==========================================
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
client = bigquery.Client(project=GCP_PROJECT_ID)

# Inicialización de hoja de cálculo
try:
    sh_seg = gc.open(SHEET_REPORT_NAME)
except gspread.exceptions.SpreadsheetNotFound:
    sh_seg = gc.create(SHEET_REPORT_NAME)

# ==========================================
# EXTRACCIÓN DE DATOS (ETL - EXTRACT)
# ==========================================
query_seg = f"""
SELECT
    REGEXP_REPLACE(TRIM(CAST(id_registro AS STRING)), r'^0+', '') AS id_registro,
    LAST_DAY(CAST(fecha_corte AS DATE)) AS fecha_norm,
    estado_actual,
    LAG(estado_actual) OVER (
        PARTITION BY REGEXP_REPLACE(TRIM(CAST(id_registro AS STRING)), r'^0+', '') 
        ORDER BY LAST_DAY(CAST(fecha_corte AS DATE)) ASC
    ) AS estado_anterior,
    CAST(metrica_valor AS FLOAT64) AS metrica_valor,
    CAST(atributo_a AS STRING) AS atributo_a,
    CAST(atributo_b AS STRING) AS atributo_b,
    CAST(atributo_c AS STRING) AS atributo_c
FROM `{BQ_SOURCE_TABLE}`
WHERE fecha_corte >= '{START_DATE}' AND fecha_corte <= '{END_DATE}'
"""

print("Extrayendo datos desde BigQuery...")
df_raw = client.query(query_seg).to_dataframe()

# ==========================================
# TRANSFORMACIÓN (ETL - TRANSFORM)
# ==========================================
def limpiar_texto(txt):
    """Normaliza texto: mayúsculas, sin espacios extra y sin tildes."""
    if txt is None: return "NULO"
    return str(txt).upper().strip().replace('Á','A').replace('É','E').replace('Í','I').replace('Ó','O').replace('Ú','U')

# Limpieza de columnas categóricas
df_raw['attr_a_clean'] = df_raw['atributo_a'].apply(limpiar_texto)
df_raw['attr_b_clean'] = df_raw['atributo_b'].apply(limpiar_texto)
df_raw['attr_c_clean'] = df_raw['atributo_c'].apply(limpiar_texto)
df_raw['metrica_valor'] = df_raw['metrica_valor'].fillna(0)

# Filtrado de registros válidos para el cálculo de transición
df = df_raw.dropna(subset=['estado_anterior', 'estado_actual']).copy()
df['fecha_dt'] = pd.to_datetime(df['fecha_norm'])

# Categorización para mantener el orden en matrices
estados_cat = sorted(df['estado_actual'].astype(str).unique())
df['estado_actual'] = pd.Categorical(df['estado_actual'].astype(str), categories=estados_cat)
df['estado_anterior'] = pd.Categorical(df['estado_anterior'].astype(str), categories=estados_cat)

meses_lista = [m for m in sorted(df['fecha_dt'].dt.to_period('M').unique()) if str(m) >= '2024-01' and str(m) <= '2026-12']

def segmentar_datos(data):
    """
    Divide el dataframe en segmentos de negocio predefinidos.
    (Lógica abstraída para sala limpia)
    """
    seg_1 = data[data['attr_a_clean'] == 'TIPO_1']
    seg_2 = data[(data['attr_a_clean'] == 'TIPO_2') & (data['attr_b_clean'] == 'CATEGORIA_X') & (data['attr_c_clean'] != 'GRUPO_EXCLUIDO')]
    seg_3 = data[(data['attr_c_clean'] == 'GRUPO_EXCLUIDO') | ((data['attr_a_clean'] == 'TIPO_2') & (data['attr_b_clean'] == 'CATEGORIA_Y'))]
    
    return [
        (seg_1, "1. SEGMENTO_PRIMARIO"), 
        (seg_2, "2. SEGMENTO_SECUNDARIO_ESPECIAL"), 
        (seg_3, "3. SEGMENTO_TERCIARIO_MIXTO")
    ]

# ==========================================
# CARGA Y REPORTERÍA (ETL - LOAD)
# ==========================================
for mes in meses_lista:
    exitoso_mes = False
    intentos_mes = 0
    
    while not exitoso_mes and intentos_mes < 5:
        try:
            print(f"Progreso: Procesando mes {mes}")
            df_mes = df[df['fecha_dt'].dt.to_period('M') == mes].copy()
            if df_mes.empty:
                print(f"Sin datos para {mes}"); break

            sheet_name = str(mes)
            try:
                ws = sh_seg.add_worksheet(title=sheet_name, rows="4000", cols="50")
            except:
                ws = sh_seg.worksheet(sheet_name)
                ws.clear()

            segmentos = segmentar_datos(df_mes)
            row_pointer = 1
            
            for df_sub, titulo in segmentos:
                ws.update(range_name=f"A{row_pointer}", values=[[f"=== {titulo} ==="]])
                row_pointer += 2

                if not df_sub.empty:
                    # Cálculo de Matrices de Transición
                    m_exp = pd.pivot_table(df_sub, values='metrica_valor', index='estado_anterior', columns='estado_actual', aggfunc='sum', fill_value=0, dropna=False, margins=True, observed=False)
                    m_prob = (m_exp / m_exp.iloc[-1]).fillna(0) * 100
                    m_cnt = pd.pivot_table(df_sub, values='id_registro', index='estado_anterior', columns='estado_actual', aggfunc='count', fill_value=0, dropna=False, margins=True, observed=False)

                    # Resumen detallado
                    df_det = df_sub.groupby(['attr_b_clean', 'attr_a_clean']).agg(
                        TOTAL_METRICA=('metrica_valor', 'sum'), 
                        CONTEO_N=('id_registro', 'count')
                    ).reset_index().sort_values(by='TOTAL_METRICA', ascending=False)

                    # Escritura en la hoja de cálculo
                    for m_s, t_s in [(m_exp, "VOLUMEN ($)"), (m_prob, "PROBABILIDAD DE TRANSICIÓN (%)"), (m_cnt, "CONTEO DE REGISTROS (N)")] :
                        ws.update(range_name=f"A{row_pointer}:I{row_pointer}", values=[[f"MATRIZ {t_s}", "", "", "", "", "", "", "", "DESGLOSE POR ATRIBUTOS"]])
                        set_with_dataframe(ws, m_s.reset_index(), row=row_pointer + 1, col=1)
                        set_with_dataframe(ws, df_det, row=row_pointer + 1, col=9)
                        row_pointer += max(len(m_s), len(df_det)) + 6
                        
                        # Prevención de Rate Limiting (Google Sheets API)
                        time.sleep(2)
                else:
                    ws.update(range_name=f"A{row_pointer}", values=[["SIN DATOS PARA ESTE SEGMENTO"]])
                    row_pointer += 5
                row_pointer += 2

            exitoso_mes = True
            print(f"Mes {mes} completado con éxito.")
            time.sleep(10) # Enfriamiento entre hojas
            
        except Exception as e:
            # Lógica de Exponential Backoff para HTTP 429 (Too Many Requests)
            if "429" in str(e):
                intentos_mes += 1
                tiempo_espera = 60 * intentos_mes
                print(f"API Rate Limit alcanzado. Reintentando en {tiempo_espera}s... (Intento {intentos_mes}/5)")
                time.sleep(tiempo_espera)
            else: 
                print(f"Error crítico durante procesamiento: {e}")
                break

print("Proceso de Matriz de Transición Finalizado.")
