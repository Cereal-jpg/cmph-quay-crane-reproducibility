"""Generador del dataset sintético atraque × turno (mismo esquema que `data/processed/dataset_turnos.parquet`).

`ajustar()` estima los parámetros con los datos reales (se guardan en `models/`, fuera de git) y `generar()`
produce un dataset nuevo con esos parámetros. El código se publica; los parámetros no.

Estructura del generador (ver `reproducibilidad/README.md`):
1. Escalas en secuencia, con los huecos entre escalas del dato real y fechas desplazadas a un origen ficticio.
2. Escala: categorías anonimizadas (servicio y, condicionadas a él, operador, clase y país) y numéricas de la
   escala por cópula gaussiana, con la media latente de cada servicio.
3. Turno: cópula gaussiana condicionada a la escala, a la posición del turno y a T1/T2/T3, con persistencia
   AR(1) dentro de la escala.
4. CMPH con la definición real: productividad latente μ = tendencia + servicio + k·g(X) + ruido de escala y
   de turno (con su varianza móvil en el tiempo); movs ~ Poisson(μ · horas_grua) y cmph = movs / horas_grua.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

ESCALA = ["visita_id", "seq"]
QC = ["qc_1", "qc_2", "qc_3", "qc_4", "qc_5"]
ESC_CAT = ["servicio", "operador_linea", "clase_buque", "pais"]
ESC_NUM = ["esc_total_ctn", "esc_transbordo", "esc_reestiba", "esc_tapas", "esc_pct_40", "esc_pct_vacio",
           "esc_pct_descarga", "atraso_itinerario_h", "horas_cutoff_a_inicio", "n_turnos"]
TUR_NUM = QC + ["n_tractos", "n_rtg", "n_reach_stacker", "n_empty_handler", "pct_reefer", "pct_hazmat", "pct_oog",
                "pct_twin", "pct_sobre_cubierta", "n_bahias", "pts_descarga", "pts_embarque", "pts_max_grua",
                "cit_ofertadas_imp", "cit_ofertadas_exp", "adjacent_berth_active", "h_por_grua"]
POSICION = ["es_primero", "es_ultimo", "pos_rel", "t2", "t3"]
# Variables que ve g(X): las numéricas del modelo final sin las de nivel (salen del propio CMPH) más el turno
G_X = ["n_gruas", "n_tractos", "n_rtg", "n_reach_stacker", "n_empty_handler", "n_tractos_por_grua",
       "pct_reefer", "pct_hazmat", "pct_oog", "pct_twin", "pct_sobre_cubierta", "n_bahias", *QC,
       "esc_total_ctn", "esc_transbordo", "esc_reestiba", "esc_tapas", "esc_pct_40", "esc_pct_vacio",
       "pct_carga_pendiente", "atraso_itinerario_h", "horas_cutoff_a_inicio", "pts_descarga", "pts_embarque",
       "pts_max_grua", "cit_ofertadas_imp", "cit_ofertadas_exp", "adjacent_berth_active", "esc_pct_descarga", "t2", "t3"]
ENTERAS = {"esc_total_ctn", "esc_transbordo", "esc_reestiba", "esc_tapas", "n_turnos", "n_tractos", "n_rtg",
           "n_reach_stacker", "n_empty_handler", "n_bahias", "pts_descarga", "pts_embarque", "pts_max_grua",
           "cit_ofertadas_imp", "cit_ofertadas_exp", *QC, "adjacent_berth_active"}
HORAS = {7: "T1", 15: "T2", 23: "T3"}
CMPH_MAX, H_MIN, MU_MIN = 40, 2.0, 3.0
ORIGEN = pd.Timestamp("2030-01-01 07:00")


# --- Cópula: marginal empírica <-> normal estándar ------------------------------------------------

def marginal(x):
    x = pd.Series(x, dtype=float)
    return {"valores": np.sort(x.dropna().to_numpy()), "p_nulo": float(x.isna().mean()), "entera": False}


def a_normal(x, m):
    """Rango medio -> z; los nulos quedan en 0 (la mediana)."""
    x = pd.Series(x, dtype=float)
    u = x.rank(method="average") / (x.notna().sum() + 1)
    return norm.ppf(u.fillna(0.5)).astype(float)


def desde_normal(z, m, rng):
    u = norm.cdf(z)
    x = np.quantile(m["valores"], u, method="inverted_cdf" if m["entera"] else "linear")
    x = x.astype(float)
    x[rng.random(len(x)) < m["p_nulo"]] = np.nan
    return x


def condicional(cov, i_obj, i_cond):
    """Normal multivariada condicional: z_obj | z_cond ~ N(B z_cond, S)."""
    s_oc = cov[np.ix_(i_obj, i_cond)]
    s_cc = cov[np.ix_(i_cond, i_cond)]
    B = s_oc @ np.linalg.pinv(s_cc)
    S = cov[np.ix_(i_obj, i_obj)] - B @ s_oc.T
    w, v = np.linalg.eigh((S + S.T) / 2)
    return B, v * np.sqrt(np.clip(w, 0, None))           # S = L Lᵀ (raíz por autovalores, robusta a S casi singular)


def clase_posicion(orden, n):
    return np.where(n == 1, "unico", np.where(orden == 0, "primero", np.where(orden == n - 1, "ultimo", "medio")))


# --- Ajuste ------------------------------------------------------------------------------------------

def secuencias(turnos, excluidos):
    """Todos los turnos de cada escala (válidos y excluidos) en orden, con su posición."""
    cols = ESCALA + ["inicio_turno", "movs"]
    todo = pd.concat([turnos[cols].assign(valido=True), excluidos[cols].assign(valido=False)], ignore_index=True)
    todo = todo[todo.set_index(ESCALA).index.isin(turnos.set_index(ESCALA).index)]
    todo = todo.sort_values(["inicio_turno", "visita_id", "seq"], kind="stable").reset_index(drop=True)
    todo["orden"] = todo.groupby(ESCALA).cumcount()
    todo["n_turnos"] = todo.groupby(ESCALA)["orden"].transform("size")
    todo["pos"] = clase_posicion(todo["orden"].to_numpy(), todo["n_turnos"].to_numpy())
    return todo


def ajustar(turnos, excluidos, plan=None, semilla=0):
    """Parámetros del generador.

    turnos: `dataset_turnos.parquet` con la partición de `cargar()` · excluidos: `turnos_excluidos.parquet` ·
    plan: DataFrame (visita_id, seq, plan_terminal) del plan del terminal por escala, o None.
    """
    t = turnos.copy()
    t["h_por_grua"] = t["horas_grua"] / t["n_gruas"]
    t[QC + ["adjacent_berth_active"]] = t[QC + ["adjacent_berth_active"]].astype(float)
    seq = secuencias(t, excluidos)
    t = t.merge(seq.loc[seq["valido"], ESCALA + ["inicio_turno", "orden", "n_turnos", "pos"]],
                on=ESCALA + ["inicio_turno"], how="left", validate="1:1")
    p = {}

    # Categorías anonimizadas por frecuencia
    esc = t.sort_values("inicio_turno").groupby(ESCALA, as_index=False, sort=False).first()
    anonimo = {}
    for c in ESC_CAT:
        orden = esc[c].astype(object).value_counts().index
        anonimo[c] = {v: f"{c}_{i + 1:02d}" for i, v in enumerate(orden)}
        esc[c] = esc[c].astype(object).map(anonimo[c])            # nulos siguen nulos
    p["servicios"] = esc["servicio"].value_counts(normalize=True)
    p["cat_por_servicio"] = {c: esc.groupby("servicio")[c].apply(lambda s: s.fillna("<nulo>").value_counts(normalize=True))
                             for c in ESC_CAT[1:]}

    # Escala: cópula con media latente por servicio (encogida)
    p["m_esc"] = {c: marginal(esc[c]) | {"entera": c in ENTERAS} for c in ESC_NUM}
    z_esc = np.column_stack([a_normal(esc[c], p["m_esc"][c]) for c in ESC_NUM])
    p["sd_z_esc"] = z_esc.std(axis=0)                         # < 1 en las discretas (empates): se deshace al generar
    zs = pd.DataFrame(z_esc).groupby(esc["servicio"].to_numpy())
    n_s = zs.size().to_numpy()[:, None]
    p["media_servicio"] = pd.DataFrame(zs.mean().to_numpy() * n_s / (n_s + 10), index=zs.mean().index)
    resid = z_esc - p["media_servicio"].loc[esc["servicio"]].to_numpy()
    w, v = np.linalg.eigh(np.cov(resid.T))
    p["L_esc"] = v * np.sqrt(np.clip(w, 0, None))

    # Huecos entre escalas, en turnos (del fin de una al inicio de la siguiente)
    ini = seq.groupby(ESCALA, sort=False)["inicio_turno"].agg(["min", "max"]).sort_values("min")
    hueco = (ini["min"].iloc[1:].to_numpy() - ini["max"].iloc[:-1].to_numpy()) / pd.Timedelta(hours=8) - 1
    p["huecos"] = np.clip(np.round(hueco), -2, None).astype(int)
    p["n_escalas"] = len(ini)
    p["dias_real"] = (seq["inicio_turno"].max() - seq["inicio_turno"].min()).days

    # Turnos excluidos: probabilidad y movs según la posición
    p["p_excluido"] = 1 - seq.groupby("pos")["valido"].mean()
    p["movs_excluido"] = {k: d.loc[~d["valido"], "movs"].to_numpy() for k, d in seq.groupby("pos")}

    # Turno: cópula condicionada a escala + posición, con AR(1) dentro de la escala
    p["m_tur"] = {c: marginal(t[c]) | {"entera": c in ENTERAS} for c in TUR_NUM}
    t = t.merge(esc[ESCALA + ESC_CAT], on=ESCALA, how="left", suffixes=("_real", ""))
    t["pos_rel"] = t["orden"] / np.maximum(t["n_turnos"] - 1, 1)
    t["es_primero"] = (t["orden"] == 0).astype(float)
    t["es_ultimo"] = (t["orden"] == t["n_turnos"] - 1).astype(float)
    t["t2"] = (t["turno"] == "T2").astype(float)
    t["t3"] = (t["turno"] == "T3").astype(float)
    z_e = pd.DataFrame(z_esc, columns=ESC_NUM).assign(**{c: esc[c] for c in ESCALA})
    z_e = t[ESCALA].merge(z_e, on=ESCALA, how="left")[ESC_NUM].to_numpy()
    pos = ((t[POSICION] - t[POSICION].mean()) / t[POSICION].std()).to_numpy()
    p["pos_media"], p["pos_sd"] = t[POSICION].mean(), t[POSICION].std()
    z_t = np.column_stack([a_normal(t[c], p["m_tur"][c]) for c in TUR_NUM])
    p["sd_z_tur"] = z_t.std(axis=0)
    todo = np.column_stack([z_e, pos, z_t])
    k_c = z_e.shape[1] + pos.shape[1]
    p["B_tur"], p["L_tur"] = condicional(np.cov(todo.T), list(range(k_c, todo.shape[1])), list(range(k_c)))
    r = pd.DataFrame(z_t - np.column_stack([z_e, pos]) @ p["B_tur"].T)
    lag = r.groupby([t["visita_id"], t["seq"]]).shift(1)
    p["rho_tur"] = np.array([np.corrcoef(r[j][lag[j].notna()], lag[j].dropna())[0, 1] for j in r.columns]).clip(0, 0.95)

    # CMPH: tendencia, servicio, g(X) y descomposición del ruido
    t = t.sort_values("inicio_turno").reset_index(drop=True)
    dia = ((t["inicio_turno"] - t["inicio_turno"].min()) / pd.Timedelta(days=1)).to_numpy()
    serie = pd.Series(t["cmph"].to_numpy(), index=pd.to_datetime(dia, unit="D"))
    diaria = serie.resample("1D").mean().rolling(91, center=True, min_periods=20).mean()
    p["tendencia"] = diaria.interpolate(limit_direction="both").to_numpy()
    tend = p["tendencia"][np.minimum(dia.astype(int), len(p["tendencia"]) - 1)]
    resto = t["cmph"].to_numpy() - tend
    rs = pd.Series(resto).groupby(t["servicio"].to_numpy())
    p["efecto_servicio"] = rs.mean() * rs.size() / (rs.size() + 5)
    resto = resto - p["efecto_servicio"].reindex(t["servicio"]).fillna(0).to_numpy()

    X = t[G_X].astype(float)
    g = lambda: LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, random_state=semilla, verbose=-1)
    oof = np.zeros(len(t))
    grupos = t["visita_id"].astype(str) + "-" + t["seq"].astype(str)
    for tr, va in GroupKFold(5).split(X, resto, grupos):
        oof[va] = g().fit(X.iloc[tr], resto[tr]).predict(X.iloc[va])
    p["g"] = g().fit(X, resto)
    e = pd.Series(resto - oof)
    pares = e.groupby(grupos.to_numpy()).apply(lambda s: (s.sum() ** 2 - (s ** 2).sum()) / 2)
    n_pares = grupos.value_counts().pipe(lambda n: n * (n - 1) / 2).sum()
    p["var_escala"] = max(float(pares.sum() / n_pares), 0.0)
    p["var_poisson"] = float((t["cmph"] / t["horas_grua"]).mean())
    p["var_turno"] = max(float(e.var()) - p["var_escala"] - p["var_poisson"], 0.1)
    p["var_g_oof"] = float(np.var(oof))
    # Ruido que cambia en el tiempo: varianza móvil del residuo relativa a la total (media 1)
    e2 = pd.Series((e ** 2).to_numpy(), index=serie.index).resample("1D").mean()
    f = e2.rolling(181, center=True, min_periods=30).mean().interpolate(limit_direction="both").to_numpy()
    p["factor_ruido"] = f / np.mean(f[np.minimum(dia.astype(int), len(f) - 1)])
    p["k"], p["s"], p["s_esc"] = 1.0, 1.0, 1.0                 # perillas de calibración: señal, ruido de turno y de escala

    # Plan del terminal: plan = a + b · cmph_escala + ruido (escalas de val con plan)
    p["plan"] = None
    if plan is not None:
        e_val = t[t["particion"] == "val"].groupby(ESCALA).agg(movs=("movs", "sum"), horas=("horas_grua", "sum"))
        e_val = e_val.join(plan.set_index(ESCALA)["plan_terminal"], how="inner").dropna()
        x = (e_val["movs"] / e_val["horas"]).to_numpy()
        b, a = np.polyfit(x, e_val["plan_terminal"].to_numpy(), 1)
        p["plan"] = {"a": a, "b": b, "sd": float(np.std(e_val["plan_terminal"] - (a + b * x))), "n": len(e_val),
                     "factor_sd": 1.0}                    # se calibra con el MAE del plan en val (01)
    return p


# --- Generación --------------------------------------------------------------------------------------

def generar(p, semilla=0, k=None, s=None, s_esc=None):
    """Dataset sintético con el esquema de `dataset_turnos.parquet` (más `plan_terminal` por escala)."""
    rng = np.random.default_rng(semilla)
    k = p["k"] if k is None else k
    s = p["s"] if s is None else s
    s_esc = p["s_esc"] if s_esc is None else s_esc
    n_esc = p["n_escalas"]

    # Escalas
    serv = rng.choice(p["servicios"].index.to_numpy(), n_esc, p=p["servicios"].to_numpy())
    esc = pd.DataFrame({"visita_id": np.arange(1, n_esc + 1), "seq": 1, "servicio": serv})
    for c in ESC_CAT[1:]:
        dist = p["cat_por_servicio"][c]
        esc[c] = [rng.choice(dist[sv].index.to_numpy(), p=dist[sv].to_numpy()) for sv in serv]
        esc[c] = esc[c].replace("<nulo>", np.nan)
    z_esc = p["media_servicio"].loc[serv].to_numpy() + rng.standard_normal((n_esc, len(ESC_NUM))) @ p["L_esc"].T
    for j, c in enumerate(ESC_NUM):
        esc[c] = desde_normal(z_esc[:, j] / p["sd_z_esc"][j], p["m_esc"][c], rng)
    esc["n_turnos"] = esc["n_turnos"].fillna(1).clip(lower=1).astype(int)

    # Calendario (en turnos de 8 h desde el origen ficticio)
    huecos = rng.choice(p["huecos"], n_esc)
    inicio = np.zeros(n_esc, dtype=int)
    for i in range(1, n_esc):
        inicio[i] = max(inicio[i - 1] + esc["n_turnos"].iloc[i - 1] + huecos[i], inicio[i - 1] + 1)
    esc["inicio"] = inicio

    # Turnos (todos, válidos y excluidos)
    t = esc.loc[esc.index.repeat(esc["n_turnos"])].reset_index(drop=True)
    t["orden"] = t.groupby("visita_id").cumcount()
    t["inicio_turno"] = ORIGEN + pd.to_timedelta((t["inicio"] + t["orden"]) * 8, unit="h")
    t["turno"] = t["inicio_turno"].dt.hour.map(HORAS)
    t["pos"] = clase_posicion(t["orden"].to_numpy(), t["n_turnos"].to_numpy())
    t["excluido"] = rng.random(len(t)) < t["pos"].map(p["p_excluido"]).fillna(0).to_numpy()
    t["pos_rel"] = t["orden"] / np.maximum(t["n_turnos"] - 1, 1)
    t["es_primero"] = (t["orden"] == 0).astype(float)
    t["es_ultimo"] = (t["orden"] == t["n_turnos"] - 1).astype(float)
    t["t2"] = (t["turno"] == "T2").astype(float)
    t["t3"] = (t["turno"] == "T3").astype(float)

    cond = np.column_stack([z_esc[t["visita_id"].to_numpy() - 1],
                            ((t[POSICION] - p["pos_media"]) / p["pos_sd"]).to_numpy()])
    media = cond @ p["B_tur"].T
    eta = rng.standard_normal((len(t), len(TUR_NUM))) @ p["L_tur"].T
    r = np.zeros_like(eta)
    rho = p["rho_tur"]
    for i in range(len(t)):
        r[i] = rho * r[i - 1] + np.sqrt(1 - rho ** 2) * eta[i] if t.at[i, "orden"] > 0 else eta[i]
    z_t = media + r
    for j, c in enumerate(TUR_NUM):
        t[c] = desde_normal(z_t[:, j] / p["sd_z_tur"][j], p["m_tur"][c], rng)
    for c in QC + ["adjacent_berth_active"]:
        t[c] = t[c].fillna(0)
    sin_grua = t[QC].sum(axis=1) == 0
    mejor = np.array(QC)[np.argmax(z_t[:, :len(QC)], axis=1)]
    for c in QC:
        t.loc[sin_grua & (mejor == c), c] = 1
    t["n_gruas"] = t[QC].sum(axis=1).astype(int)
    t["h_por_grua"] = t["h_por_grua"].clip(lower=H_MIN)
    t["horas_grua"] = t["n_gruas"] * t["h_por_grua"]
    t["n_tractos_por_grua"] = t["n_tractos"] / t["n_gruas"]

    # CMPH turno a turno (pct_carga_pendiente depende de los movs previos de la escala)
    dias = (t["inicio_turno"] - ORIGEN) / pd.Timedelta(days=1)
    dia_real = np.minimum((dias * p["dias_real"] / max(dias.max(), 1)).astype(int), len(p["tendencia"]) - 1)
    f = np.sqrt(p["factor_ruido"][np.minimum(dia_real, len(p["factor_ruido"]) - 1)])
    u = rng.normal(0, np.sqrt(s_esc * p["var_escala"]), n_esc)[t["visita_id"] - 1]
    base = (p["tendencia"][dia_real] + p["efecto_servicio"].reindex(t["servicio"]).fillna(0).to_numpy()
            + f * (u + rng.normal(0, np.sqrt(s * p["var_turno"]), len(t))))
    t["movs"], t["pct_carga_pendiente"] = 0, np.nan
    acumulado = np.zeros(n_esc)
    for o in range(t["orden"].max() + 1):
        i = np.flatnonzero(t["orden"].to_numpy() == o)
        v = t["visita_id"].to_numpy()[i] - 1
        t.loc[i, "pct_carga_pendiente"] = np.clip(1 - acumulado[v] / t["esc_total_ctn"].to_numpy()[i], 0, 1)  # como en el real: no baja de 0
        ex = t["excluido"].to_numpy()[i]
        g = p["g"].predict(t.loc[i, G_X].astype(float))
        mu = np.maximum(base[i] + k * g, MU_MIN)
        movs = rng.poisson(mu * t["horas_grua"].to_numpy()[i])
        pos = t["pos"].to_numpy()[i]
        movs_ex = np.array([rng.choice(p["movs_excluido"][q]) if len(p["movs_excluido"].get(q, [])) else 0 for q in pos])
        movs = np.where(ex, movs_ex, movs)
        t.loc[i, "movs"] = movs
        acumulado[v] += movs
    t["cmph"] = t["movs"] / t["horas_grua"]
    t = t[~t["excluido"] & (t["cmph"] <= CMPH_MAX)].copy()

    # Plan del terminal por escala
    if p["plan"] is not None:
        e = t.groupby("visita_id").agg(movs=("movs", "sum"), horas=("horas_grua", "sum"))
        pl = p["plan"]["a"] + p["plan"]["b"] * e["movs"] / e["horas"] + rng.normal(0, p["plan"]["sd"] * p["plan"].get("factor_sd", 1.0), len(e))
        t["plan_terminal"] = t["visita_id"].map(pl.round(1))

    # Esquema y tipos del dataset real
    t["fecha_turno"] = t["inicio_turno"].dt.normalize()
    for c in ENTERAS - set(QC) - {"adjacent_berth_active", "n_turnos", "esc_tapas"}:
        t[c] = t[c].round().astype("Int64" if t[c].isna().any() else int)
    t["esc_tapas"] = t["esc_tapas"].round().astype("Int64")
    t[QC + ["adjacent_berth_active"]] = t[QC + ["adjacent_berth_active"]].astype(bool)
    t["clase_buque"] = t["clase_buque"].astype("category")
    cols = (ESCALA + ["fecha_turno", "turno", "inicio_turno", "cmph", "movs", "horas_grua", "n_gruas"]
            + [c for c in TUR_NUM if c != "h_por_grua"] + ["n_tractos_por_grua", "pct_carga_pendiente"]
            + [c for c in ESC_NUM if c != "n_turnos"] + ESC_CAT + ["plan_terminal"] * (p["plan"] is not None))
    return t[cols].sort_values(["inicio_turno", "visita_id"], kind="stable").reset_index(drop=True)
