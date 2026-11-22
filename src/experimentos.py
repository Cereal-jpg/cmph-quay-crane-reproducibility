"""Código común de los experimentos (E5 en adelante), extraído de `09_experimentación_algoritmos`.

Reproduce exactamente la partición, las líneas base, los folds, los pipelines y los hiperparámetros
afinados de E1, para que cada experimento nuevo se compare contra la misma referencia. Uso desde un
notebook en `notebooks/`:

    import sys; sys.path.insert(0, "..")
    from src.experimentos import *
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

RAIZ = Path(__file__).resolve().parents[1]
PROCESSED = RAIZ / "data" / "processed"
EXPERIMENTOS = RAIZ / "reports" / "experimentos"

ESCALA = ["visita_id", "seq"]
OBJETIVO = "cmph"
SEMILLA = 42
SEMILLAS = range(10)
BASES = ["b_media", "b_servicio", "b_reciente30", "b_anterior"]
PATIO = ["pat_ocupacion_inicio", "pat_entrada_prev", "pat_salida_prev"]
CATEGORICAS = ["turno", "servicio", "operador_linea", "clase_buque", "pais"]
# E7 (notebook 13): fuera del modelo final por disponibilidad dudosa al inicio del turno
FUERA_E7 = ["cit_tomadas_imp", "n_operadores_grua", "cambio_nave", "avance_escala"]
# Numéricas del modelo final (E5 sin FUERA_E7, docs/variables_modelo_final.md), explícitas para correr sin el
# catálogo (que no está en git); reproducibilidad/01 comprueba que coinciden con predictoras() + E5 − FUERA_E7
NUM_FINAL_LISTA = [
    "n_gruas", "n_tractos", "n_rtg", "n_reach_stacker", "n_empty_handler", "n_tractos_por_grua",
    "pct_reefer", "pct_hazmat", "pct_oog", "pct_twin", "pct_sobre_cubierta", "n_bahias",
    "qc_1", "qc_2", "qc_3", "qc_4", "qc_5",
    "esc_total_ctn", "esc_transbordo", "esc_reestiba", "esc_tapas", "esc_pct_40", "esc_pct_vacio",
    "pct_carga_pendiente", "atraso_itinerario_h", "horas_cutoff_a_inicio",
    "pts_descarga", "pts_embarque", "pts_max_grua", "cit_ofertadas_imp", "cit_ofertadas_exp", "adjacent_berth_active",
    "esc_pct_descarga", "nivel_reciente_30d", "nivel_servicio_reciente",
]
# Colas largas (asimetría > 2): log1p antes de escalar, solo en los modelos que escalan
ASIMETRICAS = ["atraso_itinerario_h", "esc_transbordo", "esc_reestiba", "pct_hazmat", "pct_oog", "n_tractos_por_grua"]
CANDIDATOS = ["ridge", "lightgbm", "xgboost"]
# MAE de CV de los modelos afinados de E1 en el notebook 09 (sección 6.1): control de reproducción
MAE_CV_E1 = {"ridge": 2.112144, "lightgbm": 2.147007, "xgboost": 2.147861}


# --- Datos, partición y líneas base (09, secciones 2 y 3) ---------------------------------------

def nivel_reciente(serie, ts, dias=30):
    """Media del CMPH de los turnos ya cerrados (inicio <= ts − 8 h) en los `dias` previos."""
    ventana = serie[(serie.index <= ts - pd.Timedelta(hours=8)) & (serie.index > ts - pd.Timedelta(days=dias))]
    return ventana.mean()


def cargar(ruta=PROCESSED / "dataset_turnos.parquet"):
    """Dataset por turno con la partición 80/10/10 por escala y las líneas base, igual que el 09.

    `ruta`: parquet o DataFrame con el grano atraque × turno (p. ej. el sintético de `reproducibilidad/`)."""
    turnos = ruta.copy() if isinstance(ruta, pd.DataFrame) else pd.read_parquet(ruta)
    escalas = (turnos.groupby(ESCALA, as_index=False)
               .agg(inicio=("inicio_turno", "min"), n_turnos=(OBJETIVO, "size"))
               .sort_values(["inicio", "visita_id", "seq"], kind="stable").reset_index(drop=True))
    acum = escalas["n_turnos"].cumsum() / escalas["n_turnos"].sum()
    escalas["particion"] = pd.cut(acum, [0, .8, .9, 1], labels=["train", "val", "test"])
    turnos = turnos.merge(escalas[ESCALA + ["particion"]], on=ESCALA, how="left", validate="m:1")
    turnos = turnos.sort_values(["inicio_turno", "visita_id", "seq"], kind="stable").reset_index(drop=True)

    train = turnos[turnos["particion"] == "train"]
    turnos["b_media"] = train[OBJETIVO].mean()
    turnos["b_servicio"] = turnos["servicio"].map(train.groupby("servicio")[OBJETIVO].mean()).fillna(train[OBJETIVO].mean())
    serie = turnos.set_index("inicio_turno")[OBJETIVO].sort_index()
    turnos["b_reciente30"] = turnos["inicio_turno"].map(lambda ts: nivel_reciente(serie, ts))
    turnos["b_anterior"] = turnos.groupby(ESCALA)[OBJETIVO].shift(1).fillna(turnos["b_reciente30"])
    return turnos


def agregar_e5(turnos, ultimas=5):
    """Variables de E5 (notebook 10): nivel reciente de 30 días y media de las últimas escalas del servicio."""
    turnos["nivel_reciente_30d"] = turnos["b_reciente30"]
    esc = (turnos.groupby(ESCALA, as_index=False)
           .agg(servicio=("servicio", "first"), fin=("inicio_turno", "max"), movs=("movs", "sum"), horas=("horas_grua", "sum")))
    esc["fin"] = esc["fin"] + pd.Timedelta(hours=8)          # fin del último turno de la escala
    esc["cmph_escala"] = esc["movs"] / esc["horas"]
    esc = esc.sort_values(["fin", "visita_id", "seq"], kind="stable")
    historia = {s: (d["fin"].to_numpy(), d["cmph_escala"].to_numpy()) for s, d in esc.groupby("servicio")}

    def nivel_servicio(servicio, ts):
        if servicio not in historia:
            return np.nan
        fin, cmph = historia[servicio]
        n = np.searchsorted(fin, np.datetime64(ts), side="right")        # escalas terminadas hasta el inicio del turno
        return cmph[n - ultimas:n].mean() if n >= ultimas else np.nan

    turnos["nivel_servicio_reciente"] = [nivel_servicio(s, ts) for s, ts in zip(turnos["servicio"], turnos["inicio_turno"])]
    turnos["nivel_servicio_reciente"] = turnos["nivel_servicio_reciente"].fillna(turnos["nivel_reciente_30d"])
    return turnos


def predictoras(turnos, extra=()):
    """Numéricas de E1 (catálogo: ex-ante + plan, sin patio) más las columnas `extra`."""
    cat = pd.read_csv(RAIZ / "reports" / "catalogo_dataset_turnos.csv").query("tabla == 'turnos'")
    momento = cat.set_index("variable")["momento"]
    fuera = ESCALA + ["fecha_turno", "inicio_turno", "particion", OBJETIVO, "movs", "horas_grua"] + BASES + PATIO
    pred = [v for v in momento.index[momento.isin(["ex-ante", "plan"])] if v in turnos.columns and v not in fuera]
    return [c for c in pred if c not in CATEGORICAS] + list(extra)


def preparar_X(df, numericas):
    X = df[numericas + CATEGORICAS].copy()
    X[numericas] = X[numericas].astype(float)
    X[CATEGORICAS] = X[CATEGORICAS].astype(object).where(X[CATEGORICAS].notna(), "desconocido")
    return X


# --- Pipelines (09, sección 4.2) y afinados (sección 5) -------------------------------------------

MODELOS = {
    "ridge": (Ridge(alpha=10.0), True),
    "xgboost": (XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, min_child_weight=5,
                             subsample=0.8, colsample_bytree=0.7, n_jobs=-1, random_state=SEMILLA), False),
    "lightgbm": (LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                               random_state=SEMILLA, verbose=-1), False),
}


def preprocesador(escalar, numericas):
    imputar = lambda: SimpleImputer(strategy="median", add_indicator=True)
    escalado = [("escalar", StandardScaler())] if escalar else []
    normales = [c for c in numericas if not (escalar and c in ASIMETRICAS)]
    ramas = [
        ("num", Pipeline([("imputar", imputar()), *escalado]), normales),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False), CATEGORICAS),
    ]
    if escalar:
        ramas.append(("log", Pipeline([("imputar", imputar()),
                                       ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                                       *escalado]), [c for c in ASIMETRICAS if c in numericas]))
    return ColumnTransformer(ramas, verbose_feature_names_out=False).set_output(transform="pandas")


MEJORES = json.loads((EXPERIMENTOS / "e1_hiperparametros.json").read_text(encoding="utf-8"))


def pipeline_afinado(nombre, numericas, semilla=SEMILLA):
    """Modelo de E1 con los hiperparámetros afinados en el 09 (sección 5)."""
    estimador, escalar = MODELOS[nombre]
    pipe = Pipeline([("prep", preprocesador(escalar, numericas)), ("modelo", clone(estimador))])
    pipe.set_params(**MEJORES[nombre])
    if "random_state" in pipe["modelo"].get_params():
        pipe.set_params(modelo__random_state=semilla)
    return pipe


def tiene_azar(nombre):
    return "random_state" in MODELOS[nombre][0].get_params()


# --- Validación (09, secciones 3.1 y 4.3) ---------------------------------------------------------

def folds_temporales(df, inicio=0.5, n=5):
    """Índices (train, val) de ventana creciente, con las escalas enteras y en orden temporal."""
    esc = (df.groupby(ESCALA, as_index=False)
             .agg(inicio=("inicio_turno", "min"), n_turnos=(OBJETIVO, "size"))
             .sort_values(["inicio", "visita_id", "seq"], kind="stable"))
    acum = esc["n_turnos"].cumsum() / esc["n_turnos"].sum()
    esc["bloque"] = pd.cut(acum, [0, *np.linspace(inicio, 1, n + 1)], labels=False)
    bloque = df[ESCALA].merge(esc[ESCALA + ["bloque"]], on=ESCALA, how="left", validate="m:1")["bloque"].to_numpy()
    return [(np.flatnonzero(bloque < k), np.flatnonzero(bloque == k)) for k in range(1, n + 1)]


def bootstrap_atraques(df, pred, ref=None, B=2000, semilla=42):
    """IC 95 % del MAE de `pred`; con `ref`, IC de la mejora % de `pred` frente a `ref` (pareado)."""
    rng = np.random.default_rng(semilla)
    err = df.assign(e=(df[OBJETIVO] - df[pred]).abs(), r=(df[OBJETIVO] - df[ref]).abs() if ref else 0.0)
    g = err.groupby(ESCALA).agg(e=("e", "sum"), r=("r", "sum"), n=("e", "size"))
    idx = rng.integers(0, len(g), (B, len(g)))
    e, r, n = (g[c].to_numpy()[idx].sum(axis=1) for c in ["e", "r", "n"])
    stat = e / n if ref is None else (r - e) / r * 100
    return np.percentile(stat, [2.5, 97.5])


def mae(df, pred):
    return (df[OBJETIVO] - df[pred]).abs().mean()


def r2(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = ~np.isnan(pred)
    y, pred = y[ok], pred[ok]
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def predecir_cv(fabrica, X, y, folds, ajuste=None):
    """Predicciones fuera de fold; `ajuste(tr)` devuelve kwargs extra de `fit` (p. ej. pesos)."""
    pred = pd.Series(np.nan, index=X.index)
    for tr, va in folds:
        kw = ajuste(tr) if ajuste else {}
        pred.iloc[va] = fabrica().fit(X.iloc[tr], y.iloc[tr], **kw).predict(X.iloc[va])
    return pred


def puertas(cv, val, cand, ref, semillas, fold="fold"):
    """Las cuatro puertas de `cand` frente a `ref`: columnas de `cv` (con `fold`) y de `val`.

    P1 IC 95 % pareado de la mejora en CV > 0 · P2 gana en ≥ 4/5 folds · P3 val no empeora ·
    P4 mejora en val positiva en ≥ 8 de 10 semillas (`semillas`: mejoras % por semilla).
    """
    m_ref, m_cand = mae(cv, ref), mae(cv, cand)
    lo, hi = bootstrap_atraques(cv, cand, ref)
    por_fold = cv.groupby(fold)[[OBJETIVO, ref, cand]].apply(lambda g: mae(g, ref) - mae(g, cand))
    v_ref, v_cand = mae(val, ref), mae(val, cand)
    vlo, vhi = bootstrap_atraques(val, cand, ref)
    semillas = np.asarray(semillas)
    fila = {"MAE_cv_ref": m_ref, "MAE_cv": m_cand, "mejora_cv_%": (m_ref - m_cand) / m_ref * 100,
            "mejora_cv_ic": f"[{lo:.1f}, {hi:.1f}]", "folds_ganados": f"{int((por_fold > 0).sum())}/{len(por_fold)}",
            "MAE_val_ref": v_ref, "MAE_val": v_cand, "mejora_val_%": (v_ref - v_cand) / v_ref * 100,
            "mejora_val_ic": f"[{vlo:.1f}, {vhi:.1f}]", "semillas_positivas": f"{int((semillas > 0).sum())}/{len(semillas)}",
            "P1": bool(lo > 0), "P2": bool((por_fold > 0).sum() >= 0.8 * len(por_fold)), "P3": bool(v_cand <= v_ref),
            "P4": bool((semillas > 0).mean() >= 0.8)}
    fila["adopta"] = all(fila[p] for p in ["P1", "P2", "P3", "P4"])
    return fila
