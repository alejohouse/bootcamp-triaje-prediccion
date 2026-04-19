"""
Script de mapeo NLP entre DxSindromatico (hospital) y síntomas (xlsx).

Estrategia de 3 capas:
1. TF-IDF coseno: similitud semántica de palabras compartidas
2. RapidFuzz token_sort_ratio: similitud de cadena tolerante a reorden
3. Reglas semánticas manuales: conocimiento médico para diagnósticos
   cuyo nombre no contiene síntomas obvios (traumas, accidentes, etc.)

Salidas:
  outputs/mapeo_dx_sintomas.csv  — tabla para inspección humana
  Imprime dict Python listo para copiar a config.py
"""
import os
import re
import sys
import unicodedata

import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import ENFERMEDAD_XLSX, MORBILIDAD_XLSX, OUTPUTS_DIR

# ============================================================
# UTILIDADES DE TEXTO
# ============================================================

STOPWORDS_ES = {
    "de", "la", "el", "en", "no", "con", "sin", "por", "para", "a",
    "y", "o", "e", "los", "las", "un", "una", "unos", "unas",
    "del", "al", "se", "su", "sus", "que", "es", "son", "fue",
    "esta", "este", "otro", "otra", "otros", "otras",
    "parte", "sitio", "forma", "causa", "tipo", "lugar",
    "especificado", "especificada", "especificacion", "especificados",
    "cualquier", "organismo", "anatomico", "nivel",
}


def normalizar(texto: str) -> str:
    """Minúsculas, sin tildes, sin puntuación, sin stopwords."""
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    tokens = [t for t in texto.split() if t not in STOPWORDS_ES and len(t) > 1]
    return " ".join(tokens)


# ============================================================
# REGLAS SEMÁNTICAS MANUALES (ampliadas)
# Clave = fragmento del DxSindromatico (se busca por Jaccard/fuzzy)
# Valor = lista de síntomas del xlsx que aplican
# ============================================================

REGLAS = {
    # ── Traumas / fracturas / heridas ──────────────────────────────────
    "TRAUMA": ["dolor muscular", "dolor de espalda", "heridas",
               "moretones"],
    "FRACTURA": ["dolor en la mano o en los dedos",
                 "dolor en el pie o en los dedos del pie",
                 "dolor en la pierna", "dolor en el brazo"],
    "HERIDA": ["sangrado intermenstrual", "heridas", "moretones"],
    "CONTUSION": ["moretones", "dolor muscular"],
    "ESGUINCE": ["dolor articular", "hinchazón de la articulación"],
    "LUXACION": ["dolor articular", "hinchazón de la articulación"],
    "AMPUTACION": ["heridas", "sangrado intermenstrual"],
    "LACERACIÓN Y HEMATOMA": ["moretones", "heridas"],
    # ── Cardiorrespiratorio ────────────────────────────────────────────
    "INFARTO": ["dolor agudo en el pecho", "opresión en el pecho",
                "dificultad para respirar", "palpitaciones"],
    "ANGINA": ["dolor agudo en el pecho", "opresión en el pecho",
               "palpitaciones"],
    "PARO": ["dificultad para respirar", "arritmia"],
    "FALLA CARDIACA": ["dificultad para respirar", "palpitaciones", "fatiga"],
    "INSUFICIENCIA CARDIACA": ["dificultad para respirar", "palpitaciones"],
    "TAQUICARDIA": ["palpitaciones", "arritmia", "respirando rápido"],
    "BRADICARDIA": ["palpitaciones", "mareo", "dificultad para respirar"],
    "BLOQUEO AURICULAR": ["arritmia", "palpitaciones", "mareo"],
    "CIANOSIS": ["dificultad para respirar", "respirando rápido"],
    "APNEA": ["dificultad para respirar"],
    "FALLA RESPIRATORIA": ["dificultad para respirar", "respirando rápido"],
    "INSUFICIENCIA RESPIRATORIA": ["dificultad para respirar", "respirando rápido"],
    "DISNEA": ["dificultad para respirar", "opresión en el pecho",
               "respirando rápido"],
    "CRISIS ASMATICA": ["dificultad para respirar", "opresión en el pecho", "tos"],
    "ENFERMEDAD PULMONAR": ["dificultad para respirar", "tos", "pus en el esputo"],
    # ── Sangrados ─────────────────────────────────────────────────────
    "HEMATEMESIS": ["vomitar sangre", "náuseas", "vómitos"],
    "HEMATURIA": ["sangre en la orina", "ardor al orinar"],
    "HEMOPTISIS": ["pus en el esputo", "tos", "dificultad para respirar"],
    "HEMORRAGIA": ["sangrado intermenstrual", "debilidad", "mareo"],
    "EPISTAXIS": ["hemorragia nasal", "sangrado en la boca"],
    "SANGRADO RECTAL": ["sangrado rectal", "sangre en las heces"],
    "EMESIS CUNCHO": ["vomitar sangre", "náuseas"],
    # ── Neurológico ────────────────────────────────────────────────────
    "SINCOPE": ["desmayo", "mareo", "debilidad"],
    "LIPOTIMIA": ["desmayo", "mareo", "debilidad"],
    "CONVULSION": ["convulsiones", "pérdida del conocimiento",
                   "movimientos involuntarios anormales"],
    "CEFALEA": ["dolor de cabeza", "mareo", "náuseas"],
    "MIGRAÑA": ["dolor de cabeza", "náuseas", "manchas o nubes en la visión"],
    "VERTIGO": ["mareo", "náuseas"],
    "MENINGITIS": ["dolor de cabeza", "fiebre", "rigidez o tensión en el cuello",
                   "náuseas", "vómitos"],
    "ACCIDENTE CEREBRO VASCULAR": ["debilidad", "dificultad para hablar",
                                    "dolor de cabeza"],
    "PARESIA": ["debilidad", "movimientos involuntarios anormales",
                "dificultad para hablar"],
    "PLEJIA": ["debilidad", "movimientos involuntarios anormales"],
    "EPISODIO PARESIA": ["debilidad", "movimientos involuntarios anormales"],
    "DEMENCIA": ["confusión", "pérdida de memoria", "síntomas emocionales"],
    "DETERIORO": ["confusión", "mareo"],
    "TCE": ["dolor de cabeza", "confusión", "náuseas"],
    "TRAUMATISMO CRANEOENCEFALICO": ["dolor de cabeza", "confusión", "náuseas"],
    "POLINEUROPATIA": ["entumecimiento", "debilidad",
                       "movimientos involuntarios anormales"],
    "ISQUEMIA": ["debilidad", "dolor de cabeza", "dificultad para hablar"],
    "TINNITUS": ["zumbido en el oído", "audición disminuida"],
    # ── Gastrointestinal ───────────────────────────────────────────────
    "ABDOMEN AGUDO": ["dolor abdominal agudo", "náuseas", "vómitos"],
    "APENDICITIS": ["dolor abdominal agudo", "náuseas", "fiebre", "vómitos"],
    "PANCREATITIS": ["dolor abdominal agudo", "náuseas", "vómitos", "fiebre"],
    "COLECISTITIS": ["dolor abdominal agudo", "náuseas", "vómitos"],
    "PIROSIS": ["ardor de estómago", "náuseas"],
    "EMESIS": ["vómitos", "náuseas"],
    "GASTROENTERITIS": ["diarrea", "náuseas", "vómitos", "dolor abdominal agudo"],
    "DIARREA": ["diarrea", "náuseas", "dolor abdominal agudo"],
    "ESTREÑIMIENTO": ["estreñimiento", "dolor abdominal agudo", "flatulencia"],
    "HEPATITIS": ["ictericia", "dolor abdominal agudo", "náuseas", "fatiga"],
    "HERNIA INGUINAL": ["masa en la ingle", "dolor abdominal agudo"],
    "OTROS DOLORES ABDOMINALES": ["dolor abdominal agudo", "náuseas"],
    "ULCERA PEPTICA": ["ardor de estómago", "dolor abdominal agudo", "náuseas"],
    # ── Respiratorio ───────────────────────────────────────────────────
    "BRONQUIOLITIS": ["tos", "dificultad para respirar", "fiebre"],
    "LARINGOTRAQUEITIS": ["tos", "voz ronca", "fiebre", "dificultad para respirar"],
    "AMIGDALITIS": ["dolor de garganta", "fiebre", "dificultad para tragar"],
    "ODINOFAGIA": ["dolor de garganta", "dificultad para tragar", "fiebre"],
    "DISFONIA": ["voz ronca", "dificultad para hablar"],
    "RINORREA": ["congestión nasal", "tos", "dolor de garganta"],
    "IMPOSIBILIDAD DEGLUTIR": ["dificultad para tragar", "dolor de garganta"],
    "TOS": ["tos", "pus en el esputo", "dificultad para respirar"],
    # ── Urológico ─────────────────────────────────────────────────────
    "INFECCION VIAS URINARIAS": ["ardor al orinar", "dolor suprapúbico"],
    "PIELONEFRITIS": ["fiebre", "náuseas", "ardor al orinar"],
    "RETENCION URINARIA": ["retención de orina", "dolor suprapúbico"],
    "DISURIA": ["ardor al orinar", "dolor suprapúbico"],
    "POLAQUIURIA": ["ardor al orinar", "dolor suprapúbico"],
    "INCONTINENCIA": ["retención de orina"],
    "SECRECION URETRAL": ["secreción peneana", "ardor al orinar"],
    "INFECCION GONOCOCICA": ["secreción peneana", "ardor al orinar", "fiebre"],
    # ── Ginecológico / Obstétrico ─────────────────────────────────────
    "ABORTO": ["sangrado intermenstrual", "dolor abdominal agudo",
               "sangrado vaginal después de la menopausia"],
    "AMENAZA PARTO": ["dolor abdominal agudo"],
    "PARTO": ["dolor abdominal agudo"],
    "DISMENORREA": ["dolor menstrual", "dolor pélvico", "dolor abdominal agudo"],
    "COLICO MENSTRUAL": ["dolor menstrual", "calambres de las piernas",
                          "dolor abdominal agudo"],
    "MENSTRUACIONES IRREGULARES": ["flujo menstrual escaso",
                                    "flujo menstrual abundante",
                                    "sangrado intermenstrual"],
    "AMENORREA": ["flujo menstrual escaso", "sangrado intermenstrual"],
    "RETRASO MENSTRUAL": ["flujo menstrual escaso", "sangrado intermenstrual"],
    "VAGINITIS": ["flujo vaginal", "picazón vaginal", "dolor vaginal"],
    "VAGINISMO": ["dolor vaginal"],
    "VULVOVAGINITIS": ["flujo vaginal", "picazón vaginal", "dolor vaginal"],
    "BARTHOLINITIS": ["dolor vaginal", "fiebre", "hinchazón"],
    "ENDOMETRIOSIS": ["dolor menstrual", "dolor pélvico", "sangrado intermenstrual"],
    "SANGRADO VAGINAL": ["sangrado vaginal después de la menopausia",
                          "sangrado intermenstrual", "dolor pélvico"],
    "PRURITO VAGINAL": ["picazón vaginal", "flujo vaginal"],
    "PROLAPSO": ["dolor pélvico", "retención de orina"],
    "MIOMATOSIS": ["dolor pélvico", "flujo menstrual abundante",
                    "dolor abdominal agudo"],
    "QUISTE OVARIO": ["dolor pélvico", "dolor abdominal agudo"],
    "MASTALGIA": ["dolor agudo en el pecho", "problemas mamarios posparto"],
    "MASTITIS": ["fiebre", "problemas mamarios posparto", "dolor agudo en el pecho"],
    "FIBROADENOMA": ["problemas mamarios posparto", "dolor agudo en el pecho"],
    "EDEMA GESTACIONAL": ["retención de líquidos", "hinchazón"],
    "RETENCION PLACENTA": ["sangrado vaginal después de la menopausia",
                            "dolor abdominal agudo"],
    "INFECCION HERIDA QUIRURGICA": ["fiebre", "enrojecimiento de la piel",
                                     "pus en el esputo"],
    "DISMINUCION MOVIMIENTOS FETALES": ["dolor abdominal agudo"],
    "RECIEN NACIDO": ["dificultad para respirar", "problema de alimentación infantil",
                       "ictericia"],
    "OTROS PROBLEMAS ALIMENTACION": ["problema de alimentación infantil",
                                       "dificultad para respirar"],
    "ICTERICIA NEONATAL": ["ictericia", "falta de crecimiento"],
    # ── Dermatológico ─────────────────────────────────────────────────
    "URTICARIA": ["reacción alérgica", "picazón en la piel", "erupción cutánea"],
    "DERMATITIS": ["picazón en la piel", "erupción cutánea",
                    "enrojecimiento de la piel"],
    "CELULITIS": ["enrojecimiento de la piel", "hinchazón de la piel",
                   "dolor de piel"],
    "IMPETIGO": ["erupción cutánea", "enrojecimiento de la piel"],
    "PSORIASIS": ["picazón en la piel", "descamación de la piel",
                   "erupción cutánea"],
    "ALERGIA": ["reacción alérgica", "picazón en la piel", "erupción cutánea",
                 "congestión nasal"],
    "PRURITO": ["picazón en la piel", "erupción cutánea"],
    "VERRUGAS": ["verrugas", "lesión cutánea"],
    "ABSCESO": ["enrojecimiento de la piel", "pus en el esputo", "fiebre",
                 "hinchazón de la piel"],
    "FURUNCULO": ["pus en el esputo", "enrojecimiento de la piel", "dolor de piel"],
    "UÑA ENCARNADA": ["dolor en el pie o en los dedos del pie", "pus en el esputo"],
    "TRASTORNO UÑA": ["uñas de aspecto irregular"],
    "LUPUS": ["fatiga", "erupción cutánea", "dolor articular"],
    # ── Oftalmológico ─────────────────────────────────────────────────
    "CONJUNTIVITIS": ["lagrimeo", "ardor ocular", "secreción blanca del ojo"],
    "BLEFAROCONJUNTIVITIS": ["lagrimeo", "ardor ocular", "secreción blanca del ojo",
                              "El ojo se mueve de forma anormal."],
    "ARDOR OCULAR": ["ardor ocular", "lagrimeo"],
    "DIPLOPIA": ["visión doble"],
    "CATARATA": ["visión disminuida", "manchas o nubes en la visión"],
    "PERDIDA VISION": ["visión disminuida", "manchas o nubes en la visión"],
    "CAMBIOS VISION": ["visión disminuida", "manchas o nubes en la visión"],
    "HEMORRAGIA SUBCONJUNTIVAL": ["sangrado del ojo"],
    "TRAUMA OCULAR": ["síntomas del ojo", "sangrado del ojo", "dolor en el ojo"],
    "CUERPO EXTRAÑO OJO": ["dolor en el ojo", "lagrimeo", "ardor ocular"],
    # ── ORL ───────────────────────────────────────────────────────────
    "OTALGIA": ["pus que drena del oído", "dolor de oído"],
    "OTORREA": ["pus que drena del oído", "audición disminuida"],
    "HIPOACUSIA": ["audición disminuida"],
    "OTORRAGIA": ["sangrado del oído", "dolor de oído"],
    "CUERPO EXTRAÑO NARIZ": ["congestión nasal", "hemorragia nasal"],
    "CUERPO EXTRAÑO OIDO": ["pus que drena del oído", "audición disminuida"],
    "CUERPO EXTRAÑO FARINGE": ["dificultad para tragar", "dolor de garganta"],
    # ── Metabólico / endócrino ─────────────────────────────────────────
    "DIABETES": ["poliuria", "polifagia", "polidipsia", "fatiga"],
    "PIE DIABETICO": ["hinchazón del pie o del dedo del pie",
                       "dolor en el pie o en los dedos del pie",
                       "La piel de la pierna o el pie parece infectada."],
    "HIPERTENSION": ["dolor de cabeza", "mareo", "hemorragia nasal"],
    "HIPOTENSION": ["mareo", "desmayo", "debilidad"],
    "ANEMIA": ["fatiga", "palidez", "mareo", "debilidad"],
    "ICTERICIA": ["ictericia", "náuseas", "fatiga"],
    # ── Mental ────────────────────────────────────────────────────────
    "EPISODIO DEPRESIVO": ["depresión", "insomnio", "síntomas emocionales"],
    "ANSIEDAD": ["ansiedad y nerviosismo", "palpitaciones",
                  "dificultad para respirar"],
    "ESQUIZOFRENIA": ["síntomas depresivos o psicóticos"],
    "TRASTORNO MENTAL": ["síntomas depresivos o psicóticos", "síntomas emocionales"],
    "TRASTORNO AFECTIVO": ["síntomas depresivos o psicóticos", "insomnio"],
    "TRASTORNO PSICOTICO": ["síntomas depresivos o psicóticos"],
    "INTENTO SUICIDIO": ["síntomas emocionales", "depresión"],
    "INSOMNIO": ["insomnio", "ansiedad y nerviosismo"],
    # ── Infeccioso / General ───────────────────────────────────────────
    "SEPSIS": ["fiebre", "dificultad para respirar"],
    "SHOCK": ["dificultad para respirar", "debilidad"],
    "SINDROME FEBRIL": ["fiebre", "fatiga"],
    "SINDROME INFECCION": ["fiebre", "fatiga"],
    "FIEBRE TIFOIDEA": ["fiebre", "dolor abdominal agudo", "diarrea", "debilidad"],
    "TUBERCULOSIS": ["tos", "pus en el esputo", "fiebre"],
    "TOXOPLASMOSIS": ["fiebre", "dolor muscular", "fatiga"],
    "LEUCEMIA": ["fatiga", "fiebre", "moretones"],
    "VIH": ["fatiga", "fiebre", "pérdida de peso reciente"],
    "TROMBOSIS": ["dolor en la pierna", "hinchazón", "enrojecimiento de la piel"],
    "ENVENENAMIENTO": ["náuseas", "vómitos", "dolor abdominal agudo", "confusión"],
    "INFECCION ETS": ["secreción peneana", "ardor al orinar", "lesiones en la lengua"],
    "ENFERMEDAD PELVICA": ["dolor pélvico", "fiebre", "flujo vaginal"],
    # ── Musculoesquelético ────────────────────────────────────────────
    "ARTRALGIAS": ["dolor articular", "hinchazón de la articulación"],
    "GONARTROSIS": ["dolor de rodilla", "rigidez o tensión en la rodilla"],
    "ESPASMO MUSCULAR": ["dolor muscular", "espasmos musculares"],
    "EDEMA": ["hinchazón", "retención de líquidos"],
    "PALIDEZ": ["palidez", "fatiga", "debilidad"],
    # ── Accidentes / violencia ─────────────────────────────────────────
    "ABUSO FISICO": ["dolor muscular", "moretones"],
    "ABUSO SEXUAL": ["dolor pélvico", "síntomas emocionales"],
    "VIOLENCIA": ["dolor muscular", "heridas", "síntomas emocionales"],
    "ACCIDENTE": ["dolor", "heridas", "sangrado intermenstrual"],
    "DICTAMEN EMBRIAGUEZ": ["confusión", "mareo"],
    "ANEURISMA": ["dolor abdominal agudo", "dolor de espalda"],
}


# ============================================================
# CARGA DE DATOS
# ============================================================

def cargar_datos():
    """Carga metadatos de ambos datasets sin cargar el xlsx completo."""
    print("📂 Cargando datos...")
    df_h = pd.read_excel(MORBILIDAD_XLSX, usecols=["DxSindromatico"])
    dx_unicos = sorted(df_h["DxSindromatico"].dropna().unique().tolist())
    print(f"   DxSindromatico únicos: {len(dx_unicos)}")

    df_header = pd.read_excel(ENFERMEDAD_XLSX, nrows=0)
    sintomas = [c for c in df_header.columns if c != "enfermedad"]
    print(f"   Síntomas disponibles:  {len(sintomas)}")

    return dx_unicos, sintomas


# ============================================================
# MOTOR NLP
# ============================================================

def construir_tfidf(sintomas):
    sintomas_norm = [normalizar(s) for s in sintomas]
    vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 3),
                          min_df=1, sublinear_tf=True)
    mat = vec.fit_transform(sintomas_norm)
    return vec, mat


def buscar_por_tfidf_fuzzy(dx, vec, mat, sintomas, top_n=8,
                            thr_tfidf=0.05, thr_fuzz=55):
    """Capas 1+2: TF-IDF coseno + RapidFuzz token_sort_ratio."""
    dx_norm = normalizar(dx)
    if not dx_norm.strip():
        return []

    dx_vec = vec.transform([dx_norm])
    sims = cosine_similarity(dx_vec, mat).flatten()

    candidatos = [(sintomas[i], sims[i])
                  for i in sims.argsort()[::-1]
                  if sims[i] >= thr_tfidf][:top_n * 3]

    seleccionados = []
    for sint_orig, score_tfidf in candidatos:
        sint_norm = normalizar(sint_orig)
        score_fuzz = fuzz.token_sort_ratio(dx_norm, sint_norm)
        score_final = 0.7 * score_tfidf * 100 + 0.3 * score_fuzz
        if score_fuzz >= thr_fuzz or score_tfidf >= 0.15:
            seleccionados.append((sint_orig, score_final))

    seleccionados.sort(key=lambda x: -x[1])
    return [s for s, _ in seleccionados[:top_n]]


def aplicar_reglas_semanticas(dx):
    """Capa 3: reglas semánticas por coincidencia parcial Jaccard + fuzzy."""
    dx_norm = normalizar(dx)
    tokens_dx = set(dx_norm.split())
    acumulado = []

    for clave, sints in REGLAS.items():
        clave_norm = normalizar(clave)
        tokens_clave = set(clave_norm.split())
        if not tokens_clave:
            continue
        jaccard = len(tokens_dx & tokens_clave) / len(tokens_dx | tokens_clave)
        fuzz_score = fuzz.token_sort_ratio(dx_norm, clave_norm)
        if fuzz_score >= 55 or jaccard >= 0.35:
            acumulado.extend(sints)

    return list(dict.fromkeys(acumulado))  # deduplica, mantiene orden


def filtrar_existentes(candidatos, validos_set, validos_list):
    """Solo síntomas que existen en el xlsx (nombre exacto o fuzzy ≥ 82)."""
    resultado = []
    for cand in candidatos:
        if cand in validos_set:
            resultado.append(cand)
        else:
            m = process.extractOne(cand, validos_list,
                                   scorer=fuzz.token_sort_ratio, score_cutoff=82)
            if m:
                resultado.append(m[0])
    return list(dict.fromkeys(resultado))


def generar_mapeo(dx_unicos, sintomas):
    print("\n🔎 Generando mapeo NLP...")
    vec, mat = construir_tfidf(sintomas)
    validos_set = set(sintomas)
    validos_list = sintomas

    filas = []
    mapeo = {}

    for dx in dx_unicos:
        por_nlp = buscar_por_tfidf_fuzzy(dx, vec, mat, sintomas)
        por_regla = aplicar_reglas_semanticas(dx)
        candidatos = list(dict.fromkeys(por_nlp + por_regla))
        finales = filtrar_existentes(candidatos, validos_set, validos_list)[:10]

        mapeo[dx] = finales
        filas.append({
            "dx_sindromatico": dx,
            "n_sintomas":      len(finales),
            "sintomas":        " | ".join(finales),
        })

    return mapeo, pd.DataFrame(filas)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    dx_unicos, sintomas = cargar_datos()
    mapeo, df_mapeo = generar_mapeo(dx_unicos, sintomas)

    ruta_csv = os.path.join(OUTPUTS_DIR, "mapeo_dx_sintomas.csv")
    df_mapeo.to_csv(ruta_csv, index=False, encoding="utf-8")
    print(f"\n💾 CSV exportado: {ruta_csv}")

    con = df_mapeo[df_mapeo["n_sintomas"] > 0]
    sin = df_mapeo[df_mapeo["n_sintomas"] == 0]
    print(f"\n📊 Cobertura: {len(con)}/{len(df_mapeo)} "
          f"({len(con)/len(df_mapeo)*100:.1f}%)")
    if len(sin):
        print("   Sin mapeo:")
        for dx in sin["dx_sindromatico"]:
            print(f"     - {dx}")

    print(f"\n{'=' * 70}")
    print("# SINDROMATICO_A_SINTOMAS — pegar en src/config.py")
    print("SINDROMATICO_A_SINTOMAS = {")
    for dx, sints in mapeo.items():
        if sints:
            print(f'    "{dx}": {repr(sints)},')
    print("}")
    print("=" * 70)
