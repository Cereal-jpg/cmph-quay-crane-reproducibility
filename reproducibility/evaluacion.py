"""Pipeline del modelo final (notebooks 13 y 14) sobre cualquier dataset con el esquema de `dataset_turnos`.

Lo usan la calibración (01), la robustez por semilla (02) y la comparación final (04).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experimentos import (CATEGORICAS, ESCALA, NUM_FINAL_LISTA, OBJETIVO, agregar_e5, bootstrap_atraques, cargar,
                              folds_temporales, mae, pipeline_afinado, predecir_cv, preparar_X, r2)

# Resultados con los datos reales del terminal (artículo, Tablas V y VI)
REAL = {
    "mae_cv": 2.067, "mae_val": 1.986, "mae_test": 1.963, "r2_test": 0.183,
    "mae_test_b_reciente30": 2.264, "mejora_b_reciente30_%": 13.3,
    "mae_test_b_servicio_tv": 2.171, "mejora_b_servicio_tv_%": 9.6,
    "mae_test_b_anterior": 2.280, "mejora_b_anterior_%": 13.9,
    "mae_test_plan": 3.772, "mejora_plan_%": 48.0,
    "mae_escala_test": 1.536, "mae_escala_b_reciente30": 1.932, "mae_escala_plan": 3.847,
}
TOLERANCIA = 0.05          # ±5 % en el MAE del Ridge (CV, val, test), fijada antes de generar


def rmse(df, pred):
    return float(np.sqrt(((df[OBJETIVO] - df[pred]) ** 2).mean()))


def preparar(datos):
    """Partición, líneas base y variables de E5, igual que en el notebook 14."""
    return agregar_e5(cargar(datos))


def solo_cv(turnos):
    """MAE en los folds de CV de train del Ridge final y de las líneas base (para calibrar sin tocar val ni test)."""
    train = turnos[turnos["particion"] == "train"].reset_index(drop=True)
    folds = folds_temporales(train)
    pred = predecir_cv(lambda: pipeline_afinado("ridge", NUM_FINAL_LISTA), preparar_X(train, NUM_FINAL_LISTA),
                       train[OBJETIVO], folds)
    cv = train.assign(pred=pred).dropna(subset=["pred"])
    return {"mae_cv": mae(cv, "pred"), "mae_cv_b_reciente30": mae(cv, "b_reciente30"), "mae_cv_b_anterior": mae(cv, "b_anterior"),
            "mae_cv_b_servicio": mae(cv, "b_servicio")}


def completo(turnos):
    """CV, val y test del modelo final con sus líneas base; devuelve (resumen, test por turno, test por escala)."""
    train = turnos[turnos["particion"] == "train"].reset_index(drop=True)
    val = turnos[turnos["particion"] == "val"].copy()
    test = turnos[turnos["particion"] == "test"].copy()
    X_tr = preparar_X(train, NUM_FINAL_LISTA)
    cv = train.assign(pred=predecir_cv(lambda: pipeline_afinado("ridge", NUM_FINAL_LISTA), X_tr, train[OBJETIVO],
                                       folds_temporales(train))).dropna(subset=["pred"])
    val["pred"] = pipeline_afinado("ridge", NUM_FINAL_LISTA).fit(X_tr, train[OBJETIVO]).predict(preparar_X(val, NUM_FINAL_LISTA))

    tv = pd.concat([train, val], ignore_index=True)
    test["modelo"] = pipeline_afinado("ridge", NUM_FINAL_LISTA).fit(preparar_X(tv, NUM_FINAL_LISTA), tv[OBJETIVO]).predict(
        preparar_X(test, NUM_FINAL_LISTA))
    test["b_media_tv"] = tv[OBJETIVO].mean()
    test["b_servicio_tv"] = test["servicio"].map(tv.groupby("servicio")[OBJETIVO].mean()).fillna(tv[OBJETIVO].mean())
    refs = ["b_reciente30", "b_media_tv", "b_servicio_tv", "b_anterior"] + ["plan_terminal"] * ("plan_terminal" in test)

    lo, hi = bootstrap_atraques(test, "modelo")
    filas = [{"predictor": "modelo", "MAE": mae(test, "modelo"), "IC_MAE": f"[{lo:.3f}, {hi:.3f}]", "RMSE": rmse(test, "modelo"),
              "R2": r2(test[OBJETIVO], test["modelo"]), "mejora_modelo_%": np.nan, "IC_mejora": ""}]
    for b in refs:
        d = test.dropna(subset=[b])
        blo, bhi = bootstrap_atraques(d, "modelo", ref=b)
        filas.append({"predictor": b, "MAE": mae(d, b), "IC_MAE": "", "RMSE": rmse(d, b), "R2": r2(d[OBJETIVO], d[b]),
                      "mejora_modelo_%": (mae(d, b) - mae(d, "modelo")) / mae(d, b) * 100, "IC_mejora": f"[{blo:.1f}, {bhi:.1f}]",
                      "IC_mejora_bajo": blo})
    por_turno = pd.DataFrame(filas).set_index("predictor")

    w = test.assign(**{f"{c}_w": test[c] * test["horas_grua"] for c in ["modelo"] + refs})
    e = w.groupby(ESCALA).agg(movs=("movs", "sum"), horas=("horas_grua", "sum"),
                              **{c: (f"{c}_w", lambda s: s.sum(min_count=1)) for c in ["modelo"] + refs})
    e[OBJETIVO] = e["movs"] / e["horas"]
    for c in ["modelo"] + refs:
        e[c] = e[c] / e["horas"]
    por_escala = pd.DataFrame([{"predictor": c, "escalas": len(d := e.dropna(subset=[c])), "MAE": mae(d, c),
                                "MAPE_%": ((d[OBJETIVO] - d[c]).abs() / d[OBJETIVO]).mean() * 100, "R2": r2(d[OBJETIVO], d[c])}
                               for c in ["modelo"] + refs]).set_index("predictor")

    resumen = {"mae_cv": mae(cv, "pred"), "mae_val": mae(val, "pred"), "mae_test": mae(test, "modelo"),
               "r2_test": r2(test[OBJETIVO], test["modelo"]), "sesgo_test": float((test[OBJETIVO] - test["modelo"]).mean())}
    for b, nombre in [("b_reciente30", "b_reciente30"), ("b_servicio_tv", "b_servicio_tv"), ("b_anterior", "b_anterior"),
                      ("plan_terminal", "plan")]:
        if b in por_turno.index:
            resumen[f"mae_test_{nombre}"] = por_turno.at[b, "MAE"]
            resumen[f"mejora_{nombre}_%"] = por_turno.at[b, "mejora_modelo_%"]
    resumen["mae_escala_test"] = por_escala.at["modelo", "MAE"]
    resumen["mae_escala_b_reciente30"] = por_escala.at["b_reciente30", "MAE"]
    if "plan_terminal" in por_escala.index:
        resumen["mae_escala_plan"] = por_escala.at["plan_terminal", "MAE"]
    return resumen, por_turno, por_escala


def comparar(resumen, por_turno):
    """Tabla real vs sintético y los criterios de la tolerancia."""
    tabla = pd.DataFrame({"real": pd.Series(REAL), "sintetico": pd.Series(resumen)}).dropna()
    tabla["dif_%"] = (tabla["sintetico"] - tabla["real"]) / tabla["real"].abs() * 100
    criterios = {f"MAE {k[4:]} ±{TOLERANCIA:.0%}": abs(resumen[k] - REAL[k]) / REAL[k] <= TOLERANCIA
                 for k in ["mae_cv", "mae_val", "mae_test"]}
    orden_real = ["modelo", "b_servicio_tv", "b_reciente30", "b_anterior", "plan_terminal"]
    presentes = [p for p in orden_real if p in por_turno.index]
    criterios["mismo orden de predictores (MAE test)"] = list(por_turno.loc[presentes, "MAE"].sort_values().index) == presentes
    for b in ["b_reciente30", "plan_terminal"]:
        if b in por_turno.index:
            criterios[f"Ridge gana a {b} con IC 95 % > 0"] = bool(por_turno.at[b, "IC_mejora_bajo"] > 0)
    return tabla, pd.Series(criterios, name="cumple")
