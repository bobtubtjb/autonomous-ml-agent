"""Универсальное ML-ядро: подготовка данных, CV, модели, настройка порогов."""
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")
N_JOBS = max(1, os.cpu_count() or 1)

try:
    import lightgbm as lgb
    HAVE_LGB = True
except Exception:  # локальный запуск без lightgbm
    HAVE_LGB = False
try:
    import catboost as cb
    HAVE_CB = True
except Exception:
    HAVE_CB = False

DEFAULT_PARAMS = dict(learning_rate=0.05, num_leaves=63, min_child_samples=20,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=1.0, n_estimators=3000)


# ---------- метрики (всегда "больше = лучше") ----------
LABEL_METRICS = {"macro_f1", "weighted_f1", "accuracy"}
PROB_METRICS = {"roc_auc", "logloss"}


def metric_score(metric, y, pred):
    """Метрики по меткам/значениям."""
    if metric == "rmse":
        return -float(np.sqrt(mean_squared_error(y, pred)))
    if metric == "mae":
        return -float(mean_absolute_error(y, pred))
    if metric == "r2":
        from sklearn.metrics import r2_score
        return float(r2_score(y, pred))
    if metric == "rmsle":
        return -float(np.sqrt(mean_squared_error(np.log1p(np.clip(y, 0, None)), np.log1p(np.clip(pred, 0, None)))))
    if metric == "accuracy":
        return float(accuracy_score(y, pred))
    if metric == "weighted_f1":
        return float(f1_score(y, pred, average="weighted"))
    return float(f1_score(y, pred, average="macro"))


def score_oof(metric, y, oof, task, tune=False):
    """Единая оценка OOF: для классификации oof — матрица вероятностей."""
    if task == "regression":
        return metric_score(metric, y, oof)
    K = oof.shape[1]
    try:
        if metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            if K == 2:
                return float(roc_auc_score(y, oof[:, 1]))
            return float(roc_auc_score(y, oof / oof.sum(1, keepdims=True), multi_class="ovr", labels=list(range(K))))
        if metric == "logloss":
            from sklearn.metrics import log_loss
            return -float(log_loss(y, np.clip(oof, 1e-15, 1), labels=list(range(K))))
    except ValueError:
        return float("nan")
    if tune and metric in ("macro_f1", "weighted_f1"):   # оценка с подобранными весами классов — как в финале
        return tune_class_weights(oof, y, K, metric)[1]
    return metric_score(metric, y, oof.argmax(1))


def detect_metric(task, description=""):
    import re
    d = (description or "").lower()
    if task == "regression":
        if "rmsle" in d:
            return "rmsle"
        if re.search(r"\bmae\b", d) and "rmse" not in d:
            return "mae"
        if re.search(r"\br2\b|r\^2|r-squared", d) and "rmse" not in d:
            return "r2"
        return "rmse"
    if re.search(r"\bauc\b|roc.?auc", d) and "macro" not in d:
        return "roc_auc"
    if re.search(r"log.?loss|cross.?entropy", d) and "macro" not in d:
        return "logloss"
    if "macro" in d:
        return "macro_f1"
    if "accuracy" in d and "f1" not in d:
        return "accuracy"
    if "weighted" in d:
        return "weighted_f1"
    return "macro_f1"


def detect_task(y: pd.Series):
    if y.dtype == bool or not pd.api.types.is_numeric_dtype(y):
        return "classification"
    yv = y.dropna()
    if np.allclose(yv, np.round(yv)) and yv.nunique() <= 20:
        return "classification"
    return "regression"


def make_splits(task, y, n_splits, seed, groups=None, order=None):
    """Схема валидации: блочная по времени / GroupKFold / стратифицированная / обычная."""
    n = len(y)
    if order is not None:
        pos = np.argsort(np.asarray(order), kind="stable")
        return [(pos[tr], pos[va]) for tr, va in KFold(n_splits).split(pos)]
    if groups is not None:
        from sklearn.model_selection import GroupKFold
        return list(GroupKFold(n_splits).split(np.zeros(n), y, groups))
    if task == "classification":
        try:
            return list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n), y))
        except ValueError:
            pass
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n)))


# ---------- подготовка признаков ----------
def add_row_statistics(df):
    numeric = df.select_dtypes("number")
    df["na_cnt"] = df.isna().sum(1)
    df["num_mean"] = numeric.mean(1)
    df["num_std"] = numeric.std(1)
    return df


def add_category_frequencies(df):
    for column in df.select_dtypes(exclude="number").columns:
        frequencies = df[column].value_counts(normalize=True)
        df[column + "_freq"] = df[column].map(frequencies).astype(float)
    return df


def run_fe(df, transformations):
    """Apply the trusted transformations selected within the search loop."""
    for transform in transformations:
        try:
            out = transform(df.copy())
            if isinstance(out, pd.DataFrame) and len(out) == len(df):
                out = out.loc[:, ~out.columns.duplicated()]
                df = out.replace([np.inf, -np.inf], np.nan)
        except Exception:
            continue
    return df


def expand_dates(df):
    """Строковые даты -> числовые компоненты (год, месяц, день недели, час, unix-время)."""
    for c in list(df.columns):
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) and False:
            continue
        if not pd.api.types.is_datetime64_any_dtype(s):
            smp = s.dropna().astype(str).head(200)
            if len(smp) == 0 or not smp.str.contains(r"[-/:.]").mean() > 0.9 or smp.str.len().median() < 8:
                continue
            parsed = pd.to_datetime(smp, errors="coerce")
            if parsed.notna().mean() < 0.9:
                continue
            s = pd.to_datetime(s, errors="coerce")
        df[c + "_year"] = s.dt.year; df[c + "_month"] = s.dt.month; df[c + "_dow"] = s.dt.dayofweek
        df[c + "_hour"] = s.dt.hour; df[c + "_ts"] = (s - pd.Timestamp("1970-01-01")).dt.total_seconds()
        df = df.drop(columns=[c])
    return df


def san(c):
    return (str(c).replace('"', "").replace(",", "_").replace(":", "_")
            .replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")"))


# ---------- OOF target-encoding по ключу сущности ----------
def build_gids(X, Xte, keys):
    """Целочисленные id групп для каждого ключа (по train+test вместе)."""
    out = []
    for ks in keys or []:
        ks = [san(k) for k in ks if san(k) in X.columns]
        if not ks:
            continue
        full = pd.concat([X[ks], Xte[ks]], ignore_index=True)
        g = full.groupby(ks, dropna=False, observed=True, sort=False).ngroup().values
        out.append((g[:len(X)], g[len(X):]))
    return out


def _enc(gid, s, c, scale=1.0):
    cnt = c[gid]
    return np.where(cnt > 0, s[gid] / np.maximum(cnt, 1), np.nan), cnt * scale


def te_features(gid_tr, y_tr, gid_list, n_inner=5, seed=0):
    """Признаки для train-части (внутренний OOF, без утечки) и для каждого gid из gid_list (по всему train)."""
    G = int(max(gid_tr.max(), *(g.max() for g in gid_list))) + 1
    n = len(gid_tr)
    mean_tr, cnt_tr = np.full(n, np.nan), np.zeros(n)
    for a, b in KFold(n_inner, shuffle=True, random_state=seed).split(np.arange(n)):
        s = np.bincount(gid_tr[a], weights=y_tr[a], minlength=G); c = np.bincount(gid_tr[a], minlength=G)
        mean_tr[b], cnt_tr[b] = _enc(gid_tr[b], s, c, 1.0 / (1 - 1.0 / n_inner))
    s = np.bincount(gid_tr, weights=y_tr, minlength=G); c = np.bincount(gid_tr, minlength=G)
    return (mean_tr, cnt_tr), [_enc(g, s, c) for g in gid_list]


def auto_entity_key(df, y, max_cols=8):
    """Ищет набор колонок, задающий «сущность» (здание, клиент...), повторяющуюся в разных строках,
    с максимальным coverage * R2 (leave-one-out среднее таргета по группе). Возвращает (cols, score)."""
    y = np.asarray(y, float)
    if len(df) > 200000:
        idx = np.random.default_rng(0).choice(len(df), 200000, replace=False)
        df, y = df.iloc[idx].reset_index(drop=True), y[idx]
    n = len(df)
    cols = [c for c in df.columns if 1 < df[c].nunique(dropna=False) < 0.5 * n][:150]
    codes = {c: pd.factorize(df[c], use_na_sentinel=False)[0].astype(np.int64) for c in cols}
    gm, sst_all = y.mean(), None

    def combine(key):
        g = codes[key[0]]
        for k in key[1:]:
            g = pd.factorize(g * (codes[k].max() + 1) + codes[k])[0].astype(np.int64)
        return g

    def score(key):
        g = combine(key)
        cnt = np.bincount(g); s = np.bincount(g, weights=y)
        ng = cnt[g]; mask = ng >= 2
        if mask.mean() < 0.05:
            return 0.0
        loo = (s[g] - y) / (ng - 1)
        sst = ((y[mask] - gm) ** 2).sum()
        r2 = 1 - ((y[mask] - loo[mask]) ** 2).sum() / max(sst, 1e-12)
        return float(mask.mean() * max(r2, 0))

    key, best = [], 0.0
    for _ in range(max_cols):
        cand = [(score(key + [c]), c) for c in cols if c not in key]
        if not cand:
            break
        sc, c = max(cand)
        if sc <= best * 1.01:
            break
        key, best = key + [c], sc
    return key, best


def text_feats(strs, name, n_comp=16):
    """Длина/слова + TF-IDF->SVD для текстовой колонки."""
    out = {name + "_len": strs.str.len().values, name + "_words": strs.str.count(" ").values + 1}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        tv = TfidfVectorizer(max_features=5000, min_df=3, sublinear_tf=True)
        tv.fit(strs.sample(min(len(strs), 100000), random_state=0))
        Xs = tv.transform(strs)
        k = min(n_comp, Xs.shape[1] - 1)
        if k >= 2:
            Z = TruncatedSVD(k, random_state=0).fit_transform(Xs)
            for i in range(k):
                out[f"{name}_svd{i}"] = Z[:, i]
    except Exception:
        pass
    return out


def prepare(train_x, test_x, state):
    n = len(train_x)
    full = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    full = run_fe(full, state.get("fe_steps", []))
    full = full.drop(columns=[c for c in state.get("drop", []) if c in full.columns])
    full = expand_dates(full)
    keep = [c for c in full.columns if full[c].nunique(dropna=False) > 1]
    full = full[keep]
    protect = {c for k in state.get("te_keys", []) for c in k}   # колонки ключей target-encoding не трогаем
    new_cols, drop_cols = {}, []
    for c in full.columns:
        s = full[c]
        if s.dtype == bool:
            full[c] = s.astype(int)
        elif pd.api.types.is_datetime64_any_dtype(s):
            full[c] = (s - pd.Timestamp("1970-01-01")).dt.total_seconds()
        elif not pd.api.types.is_numeric_dtype(s):
            strs = s.astype(str).where(s.notna(), "nan")
            nun = strs.nunique()
            if c not in protect and nun > 256:
                smp = strs.head(2000)
                if nun > 0.3 * len(strs) and smp.str.len().mean() > 15 and smp.str.count(" ").mean() >= 2:
                    new_cols.update(text_feats(strs, c))                      # текст
                else:
                    new_cols[c + "_freq"] = strs.map(strs.value_counts(normalize=True)).values   # высокая кардинальность
                    new_cols[c + "_code"] = pd.factorize(strs)[0]
                drop_cols.append(c)
            else:
                full[c] = s.astype(str).where(s.notna(), None).astype("category")
        else:
            full[c] = pd.to_numeric(s, errors="coerce")
    if drop_cols:
        full = full.drop(columns=drop_cols)
    if new_cols:
        full = pd.concat([full, pd.DataFrame(new_cols, index=full.index)], axis=1)
    full.columns = [san(c) for c in full.columns]
    full = full.loc[:, ~full.columns.duplicated()]
    return full.iloc[:n].reset_index(drop=True), full.iloc[n:].reset_index(drop=True)


# ---------- модели ----------
def _make_model(kind, task, params, K, seed):
    if kind == "cat" and HAVE_CB:
        common = dict(iterations=params.get("cat_iters", 1500), learning_rate=params.get("cat_lr", 0.06),
                      depth=params.get("cat_depth", 6), random_seed=seed, thread_count=N_JOBS, verbose=0,
                      od_type="Iter", od_wait=100)
        if task == "classification":
            return cb.CatBoostClassifier(loss_function="MultiClass" if K > 2 else "Logloss", **common)
        return cb.CatBoostRegressor(loss_function="RMSE", **common)
    if HAVE_LGB:
        p = {**DEFAULT_PARAMS, **{k: v for k, v in params.items() if not k.startswith("cat_")}}
        p.update(random_state=seed, n_jobs=N_JOBS, verbose=-1)
        return lgb.LGBMClassifier(**p) if task == "classification" else lgb.LGBMRegressor(**p)
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    kw = dict(max_iter=300, learning_rate=0.1, categorical_features="from_dtype", random_state=seed)
    return HistGradientBoostingClassifier(**kw) if task == "classification" else HistGradientBoostingRegressor(**kw)


def _fit(model, kind, Xtr, ytr, Xva, yva):
    if kind == "cat" and HAVE_CB:
        cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == "category"]
        Xtr, Xva = Xtr.copy(), Xva.copy()
        for c in cats:
            Xtr[c] = Xtr[c].astype(str); Xva[c] = Xva[c].astype(str)
        model.set_params(cat_features=cats)
        model.fit(Xtr, ytr, eval_set=(Xva, yva))
        return model, Xtr.columns, cats
    if HAVE_LGB:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        model.fit(Xtr, ytr)
    return model, Xtr.columns, []


def _cat_view(X, cats):
    if not cats:
        return X
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(str)
    return X


def cv_run(X, y, Xte, task, params, kind="lgb", splits=None, seed=42, log_target=False, K=None, deadline=None, gids=None):
    """Возвращает (oof, test_pred, mask). Для классификации — матрицы вероятностей."""
    n = len(X)
    n_splits = len(splits)
    oof = np.zeros((n, K)) if task == "classification" else np.zeros(n)
    te = np.zeros((len(Xte), K)) if task == "classification" else np.zeros(len(Xte))
    yt = np.log1p(y) if (task == "regression" and log_target) else y
    done, mask = 0, np.zeros(n, bool)
    for tr, va in splits:
        if deadline and time.time() > deadline and done > 0:
            break
        m = _make_model(kind, task, params, K, seed)
        Xtr, Xva, Xt = X.iloc[tr], X.iloc[va], Xte
        if gids and (task == "regression" or K == 2):
            Xtr, Xva, Xt = Xtr.copy(), Xva.copy(), Xte.copy()
            for j, (g_all, g_te) in enumerate(gids):
                (m_tr, c_tr), ((m_va, c_va), (m_te, c_te)) = te_features(g_all[tr], yt[tr], [g_all[va], g_te], 5, seed + j)
                for D, mm, cc in ((Xtr, m_tr, c_tr), (Xva, m_va, c_va), (Xt, m_te, c_te)):
                    D[f"te{j}_mean"] = mm; D[f"te{j}_cnt"] = cc
        m, _, cats = _fit(m, kind, Xtr, yt[tr], Xva, yt[va])
        if task == "classification":
            cl = np.asarray(m.classes_, dtype=int)
            po = np.zeros((len(va), K)); po[:, cl] = m.predict_proba(_cat_view(Xva, cats))
            pt = np.zeros((len(Xte), K)); pt[:, cl] = m.predict_proba(_cat_view(Xt, cats))
        else:
            po = m.predict(_cat_view(Xva, cats)); pt = m.predict(_cat_view(Xt, cats))
            if log_target:
                po, pt = np.expm1(po), np.expm1(pt)
        oof[va] = po; te += pt; mask[va] = True; done += 1
    te /= max(done, 1)
    return oof, te, mask


def _f1_fast(y, p, K, weighted=False):
    cm = np.bincount(y * K + p, minlength=K * K).reshape(K, K)
    tp = np.diag(cm).astype(float); sup = cm.sum(1); pred = cm.sum(0)
    den = sup + pred
    f1 = np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)
    if weighted:
        return float((f1 * sup).sum() / max(sup.sum(), 1))
    return float(f1[den > 0].mean())


def tune_class_weights(P, y, K, metric="macro_f1", iters=3):
    """Подбор множителей вероятностей классов под F1 (coordinate ascent, быстрый F1 через confusion matrix)."""
    y = np.asarray(y, dtype=np.int64)
    w = np.ones(K)
    f = lambda ww: _f1_fast(y, (P * ww).argmax(1), K, metric == "weighted_f1")
    best = f(w)
    grid = np.exp(np.linspace(-1.5, 1.5, 13))
    for _ in range(iters):
        improved = False
        for k in range(K):
            for g in grid:
                w2 = w.copy(); w2[k] = g
                s = f(w2)
                if s > best + 1e-9:
                    best, w, improved = s, w2, True
        if not improved:
            break
    return w, best
