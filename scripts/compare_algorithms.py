"""Compare the algorithms of the paper (Table III) on the synthetic dataset, under the same protocol.
Usage (from the repository root): python scripts/compare_algorithms.py . sintetico"""
import sys, json, time
import numpy as np, pandas as pd
REPO, DATOS = sys.argv[1], sys.argv[2]; sys.path.insert(0, REPO)
from src.experimentos import *
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
import warnings; warnings.filterwarnings("ignore")
assert DATOS == "sintetico", "only the synthetic dataset is public"
turnos = agregar_e5(cargar(pd.read_csv(RAIZ / "data" / "synthetic" / "synthetic_shifts.csv", parse_dates=["fecha_turno", "inicio_turno"])))
NUM = NUM_FINAL_LISTA
train = turnos[turnos.particion == "train"].reset_index(drop=True); val = turnos[turnos.particion == "val"].copy()
FOLDS = folds_temporales(train); X, y, Xv = preparar_X(train, NUM), train[OBJETIVO], preparar_X(val, NUM)
cv = train[ESCALA + [OBJETIVO, "b_reciente30"]].copy(); cv["fold"] = 0
for k, (tr, va) in enumerate(FOLDS, 1): cv.loc[va, "fold"] = k
def pipe(est, esc): return lambda: Pipeline([("prep", preprocesador(esc, NUM)), ("modelo", clone(est))])
def mlp(h=(64, 32), alpha=1e-2, lr=1e-3, bs=64, seed=SEMILLA):
    return MLPRegressor(hidden_layer_sizes=h, alpha=alpha, learning_rate_init=lr, batch_size=bs, early_stopping=True,
                        validation_fraction=0.15, n_iter_no_change=30, max_iter=2000, random_state=seed)
F = {"ridge_af": lambda: pipeline_afinado("ridge", NUM)}
if DATOS == "sintetico":
    F.update({"knn": pipe(KNeighborsRegressor(n_neighbors=25, weights="distance"), True), "svr": pipe(SVR(C=3.0, epsilon=0.5), True),
              "random_forest": pipe(RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.33, n_jobs=-1, random_state=SEMILLA), False),
              "hist_gb": pipe(HistGradientBoostingRegressor(learning_rate=0.05, max_iter=300, min_samples_leaf=20, random_state=SEMILLA), False),
              "xgboost": lambda: pipeline_afinado("xgboost", NUM), "lightgbm": lambda: pipeline_afinado("lightgbm", NUM)})
F["mlp"] = pipe(mlp(), True)
for n, f in F.items(): cv[n] = predecir_cv(f, X, y, FOLDS)
cv = cv[cv.fold > 0].copy()
out = {}
if DATOS == "real":   # búsqueda aleatoria de hiperparámetros del MLP en la CV (como XGBoost/LightGBM), 30 configuraciones
    rng = np.random.default_rng(0); mejor = (9, None)
    for i in range(30):
        c = dict(h=[(32,), (64,), (128,), (32, 16), (64, 32), (128, 64)][rng.integers(6)], alpha=float(10 ** rng.uniform(-4, 1.5)),
                 lr=float(10 ** rng.uniform(-4, -2)), bs=int(rng.choice([32, 64, 128])))
        p = predecir_cv(pipe(mlp(**c), True), X, y, FOLDS).loc[cv.index]; m = np.mean(np.abs(cv[OBJETIVO] - p))
        if m < mejor[0]: mejor = (m, c)
    out["mlp_mejor"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in mejor[1].items()}
    cv["mlp_af"] = predecir_cv(pipe(mlp(**mejor[1]), True), X, y, FOLDS).loc[cv.index]
    sem = []
    for s in range(10):
        p = predecir_cv(pipe(mlp(**mejor[1], seed=s), True), X, y, FOLDS).loc[cv.index]; sem.append(np.mean(np.abs(cv[OBJETIVO] - p)))
    out["mlp_af_semillas"] = [round(v, 4) for v in sem]
    m = pipe(mlp(**mejor[1]), True)().fit(X, y); out["mlp_af_r2_train"] = r2(y, m.predict(X)); val["mlp_af"] = m.predict(Xv)
m0 = pipe(mlp(), True)().fit(X, y); out["mlp_r2_train"] = r2(y, m0.predict(X)); val["mlp"] = m0.predict(Xv)
val["ridge_af"] = pipeline_afinado("ridge", NUM).fit(X, y).predict(Xv)
filas = []
for n in [c for c in cv.columns if c not in ESCALA + [OBJETIVO, "fold", "b_reciente30"]] + ["b_reciente30"]:
    f = {"modelo": n, "MAE_cv": mae(cv, n), "R2_cv": r2(cv[OBJETIVO], cv[n])}
    if n != "b_reciente30":
        lo, hi = bootstrap_atraques(cv, n, "b_reciente30"); f["vs_b30"] = f"{(mae(cv,'b_reciente30')-mae(cv,n))/mae(cv,'b_reciente30')*100:.1f} [{lo:.1f}, {hi:.1f}]"
        pf = cv.groupby("fold")[[OBJETIVO, "b_reciente30", n]].apply(lambda g: mae(g, "b_reciente30") - mae(g, n)); f["folds"] = f"{(pf>0).sum()}/5"
    if n not in ("ridge_af", "b_reciente30"):
        lo, hi = bootstrap_atraques(cv, n, "ridge_af"); f["vs_ridge"] = f"{(mae(cv,'ridge_af')-mae(cv,n))/mae(cv,'ridge_af')*100:.1f} [{lo:.1f}, {hi:.1f}]"
    if n in val: f["MAE_val"] = mae(val, n)
    filas.append(f)
t = pd.DataFrame(filas).sort_values("MAE_cv"); print(t.round(3).to_string(index=False)); print(json.dumps(out, default=float))
t.to_csv(RAIZ / "reports" / "algorithms_synthetic.csv", index=False)
