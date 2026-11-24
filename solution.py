"""Точка входа: python solution.py task.md (как baseline.py).
Env: OPENAI_API_KEY / OPENAI_MODEL / OPENAI_BASE_URL (+ TIME_BUDGET_SEC, по умолчанию 3 часа)."""
import json, os, re, sys, time, glob, threading
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import ml_core as M
import research as R

T0 = time.time()
TOTAL = float(os.getenv("TIME_BUDGET_SEC", 3 * 3600))
RESEARCH_END = T0 + TOTAL * 0.55
HARD_END = T0 + TOTAL * 0.85


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def start_watchdog():
    """Жёсткий стоп до принудительного убийства платформой: файл предсказаний к этому моменту уже сохранён."""
    def kill():
        log("WATCHDOG: время вышло, завершаем с последним сохранённым файлом")
        os._exit(0)
    t = threading.Timer(TOTAL * 0.94, kill); t.daemon = True; t.start()


def find_data_dir(task_dir, desc=""):
    for tok in re.findall(r"[\w./\\-]*train\.csv", desc):   # пути из task.md (напр. regression/participants/train.csv)
        p = os.path.normpath(os.path.join(task_dir, tok))
        if os.path.exists(p):
            return os.path.dirname(p)
    for d in (task_dir, os.path.join(task_dir, "data"), os.getenv("DATA_DIR") or task_dir):
        if os.path.exists(os.path.join(d, "train.csv")):
            return d
    for p in glob.glob(os.path.join(task_dir, "**", "train.csv"), recursive=True)[:1]:
        return os.path.dirname(p)
    raise FileNotFoundError(f"train.csv не найден в {task_dir}")


def llm_json(client, prompt, max_tokens=400):
    if not client:
        return {}
    try:
        t = R.chat(client, [{"role": "user", "content": prompt}], max(max_tokens, 3000), 0)
        return json.loads(re.search(r"\{.*\}", t, flags=re.S).group(0))
    except Exception:
        return {}


def get_spec(client, desc, train_cols, sample_cols):
    """Единый вызов LLM: путь вывода, тип задачи, метрика, время/группа для валидации."""
    p = ("Разбери условие ML-задачи. Ответ — строго JSON с ключами: "
         '"output" (путь файла для сохранения предсказаний или null), '
         '"task" ("regression"|"classification"|null), '
         '"metric" (rmse|mae|r2|rmsle|macro_f1|weighted_f1|accuracy|roc_auc|logloss|null), '
         '"time_col" (колонка времени, по которой тест лежит в будущем, или null), '
         '"group_col" (колонка-группа, чьи значения в тесте не пересекаются с train, или null).\n\n'
         f"Колонки train: {list(train_cols)[:120]}\nКолонки sample_submission: {list(sample_cols)}\n\nУсловие:\n{desc[:6000]}")
    return llm_json(client, p)


def resolve_output(spec, desc, task_dir):
    """Платформа ждёт submission.csv рядом с task.md — путь фиксирован."""
    return os.path.join(task_dir, "submission.csv")


def pick_cv(train_x, test_x, y, task, spec, n_splits, seed):
    """Выбор схемы валидации по реальному сдвигу train->test (а не по догадке)."""
    cols = list(train_x.columns)
    # время
    cands = [spec.get("time_col")] + [c for c in cols if re.search(r"date|time|timestamp", c.lower())]
    for c in cands:
        if c in cols:
            tr_t, te_t = _as_time(train_x[c]), _as_time(test_x[c])
            if tr_t.notna().mean() > 0.9 and te_t.notna().mean() > 0.9 and (te_t > tr_t.max()).mean() > 0.5:
                log(f"CV: блочная по времени ({c})")
                return M.make_splits(task, y, n_splits, seed, order=tr_t.fillna(tr_t.median()).values), [], c
    # группы
    cands = [spec.get("group_col")] + [c for c in cols if re.search(r"(^|_)(id|name|user|host|client|building|facility|device|ip|group)($|_)", c.lower())]
    for c in cands:
        if c in cols and 5 < train_x[c].nunique() < len(train_x) * 0.5:
            overlap = test_x[c].isin(set(train_x[c].dropna())).mean()
            if overlap < 0.2:
                log(f"CV: GroupKFold по {c} (пересечение с test {overlap:.0%})")
                return M.make_splits(task, y, n_splits, seed, groups=train_x[c].astype(str).values), [c], None
    return M.make_splits(task, y, n_splits, seed), [], None


def _as_time(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = pd.to_datetime(s, errors="coerce")
    return (t - pd.Timestamp("1970-01-01")).dt.total_seconds()


PROTECTED = {"train.csv", "test.csv", "data_dictionary.csv", "sample_submission.csv"}


def fix_sub(sub):
    """Конечные числа вместо NaN/inf в числовых колонках (платформа отвергает такие файлы)."""
    sub = sub.copy()
    for c in sub.columns[1:]:
        if pd.api.types.is_numeric_dtype(sub[c]):
            v = sub[c].replace([np.inf, -np.inf], np.nan)
            sub[c] = v.fillna(v.median() if v.notna().any() else 0.0)
    return sub


def save(sub, path):
    if os.path.basename(path) in PROTECTED:   # защита входных файлов от перезаписи
        path = os.path.join(os.path.dirname(os.path.abspath(path)), "submission.csv")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    fix_sub(sub).to_csv(tmp, index=False)
    os.replace(tmp, path)  # атомарно


def main():
    start_watchdog()
    task_file = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else "task.md")
    task_dir = os.path.dirname(task_file)
    desc = open(task_file, encoding="utf-8", errors="ignore").read() if os.path.exists(task_file) else ""
    d = find_data_dir(task_dir, desc)
    train = pd.read_csv(os.path.join(d, "train.csv")); test = pd.read_csv(os.path.join(d, "test.csv"))
    sample = pd.read_csv(os.path.join(d, "sample_submission.csv"))
    dictionary, id_like = {}, set()
    dp = os.path.join(d, "data_dictionary.csv")
    if os.path.exists(dp):
        dd = pd.read_csv(dp)
        if "feature" in dd and "description_ru" in dd:
            dictionary = dict(zip(dd["feature"].astype(str), dd["description_ru"].astype(str)))
        if "feature" in dd and "role" in dd:
            id_like = set(dd.loc[dd["role"].astype(str).str.lower().str.contains(r"^id$|identifier|index"), "feature"].astype(str))

    id_col = sample.columns[0]
    extra = [c for c in train.columns if c not in test.columns]
    target = extra[0] if len(extra) == 1 else sample.columns[1]
    client = R._client()
    spec = get_spec(client, desc, train.columns, sample.columns) if os.getenv("LLM_SPEC") == "1" else {}
    out_path = os.getenv("OUTPUT_PATH") or resolve_output(spec, desc, task_dir)
    log("data:", d, "| output:", out_path, "| spec:", spec)

    ids = test[id_col].values
    train = train.dropna(subset=[target]).reset_index(drop=True)
    task = M.detect_task(train[target])
    if spec.get("task") in ("regression", "classification"):
        ok = spec["task"] == "classification" or pd.api.types.is_numeric_dtype(train[target])
        ok &= not (spec["task"] == "classification" and pd.api.types.is_float_dtype(train[target])
                   and train[target].nunique() > 50)
        if ok:
            task = spec["task"]
    metric = spec.get("metric") if spec.get("metric") in (M.LABEL_METRICS | M.PROB_METRICS | {"rmse", "mae", "r2", "rmsle"}) \
        else M.detect_metric(task, desc)
    if (task == "regression") == (metric in M.LABEL_METRICS | M.PROB_METRICS):
        metric = M.detect_metric(task, desc)
    feats = [c for c in train.columns if c not in (target, id_col) and c in test.columns]
    train_x, test_x = train[feats], test[feats]
    log(f"task={task} metric={metric} train={train_x.shape} test={test_x.shape}")

    if task == "classification":
        try:
            classes = np.array(sorted(train[target].unique()))
        except TypeError:
            classes = np.array(sorted(train[target].unique(), key=str))
        cmap = {c: i for i, c in enumerate(classes)}
        y = train[target].map(cmap).values.astype(int); K = len(classes)
    else:
        classes, y, K = None, train[target].values.astype(float), None

    scols = list(sample.columns[1:])

    def to_sub(P, w=None):
        if task == "regression":
            return pd.DataFrame({id_col: ids, scols[0]: P})
        if len(scols) == K and K > 2 or (len(scols) > 1 and metric in M.PROB_METRICS):
            Pn = np.clip(P / P.sum(1, keepdims=True), 1e-5, 1); Pn = Pn / Pn.sum(1, keepdims=True)
            order = [list(map(str, classes)).index(c) if c in list(map(str, classes)) else i for i, c in enumerate(scols)]
            df = pd.DataFrame(Pn[:, order], columns=scols[:len(order)]); df.insert(0, id_col, ids); return df
        if metric in M.PROB_METRICS:   # бинарная вероятность положительного класса
            return pd.DataFrame({id_col: ids, scols[0]: np.clip(P[:, 1] / P.sum(1), 1e-5, 1 - 1e-5)})
        return pd.DataFrame({id_col: ids, scols[0]: classes[(P * (w if w is not None else 1)).argmax(1)]})

    # --- 0. заведомо валидный файл сразу (константа) ---
    prior = np.bincount(y, minlength=K) / len(y) if task == "classification" else None
    try:
        save(to_sub(np.tile(prior, (len(ids), 1)) if task == "classification" else np.full(len(ids), y.mean())), out_path)
        log("constant submission saved")
    except Exception as e:
        log("constant fallback failed:", repr(e))

    state = dict(task=task, params={}, log_target=(metric == "rmsle"), force_log=(metric == "rmsle"),
                 fe_steps=[], drop=[c for c in id_like if c in feats])
    splits5, cv_drop, tcol = pick_cv(train_x, test_x, y, task, spec, 5, 42)
    state["drop"] = sorted(set(state["drop"]) | set(cv_drop))
    fixed_cv = bool(cv_drop or tcol)
    splits3 = splits5[:3] if fixed_cv else M.make_splits(task, y, 3, 42)

    # --- 1. быстрый надёжный результат ---
    try:
        X, Xte = M.prepare(train_x, test_x, state)
        quick = dict(learning_rate=0.1, n_estimators=500)
        oof, te, _ = M.cv_run(X, y, Xte, task, quick, "lgb", splits3, 42, state["log_target"], K)
        save(to_sub(te), out_path); log("baseline submission saved")
    except Exception as e:
        log("baseline failed:", repr(e))

    # --- 2. AutoResearch (на больших данных — по подвыборке) ---
    rtx, rty, rsp = train_x, y, splits5
    SUB = int(os.getenv("RESEARCH_SUBSAMPLE", 300000))
    if len(train_x) > SUB and not fixed_cv:
        idx = np.sort(np.random.default_rng(0).choice(len(train_x), SUB, replace=False))
        rtx, rty = train_x.iloc[idx].reset_index(drop=True), y[idx]
        rsp = M.make_splits(task, rty, 5, 42)
        log(f"research на подвыборке {SUB} строк")
    try:
        state, best = R.research_loop(state, rtx, rty, test_x, K, metric, dictionary, desc, RESEARCH_END, rsp, log)
        log(f"research done, best CV={best:.5f}, params={state['params']}, te_keys={state.get('te_keys')}, "
            f"log_target={state['log_target']}, fe_steps={len(state['fe_steps'])}")
    except Exception as e:
        log("research failed:", repr(e))

    # --- 3. финал: LGBM и CatBoost по нескольким сидам, бленд ---
    try:
        X, Xte = M.prepare(train_x, test_x, state)
        gids = M.build_gids(X, Xte, state.get("te_keys"))

        def run_seeds(kind, seeds, until, first_ok=True):
            oofs, tes, masks, dur = [], [], [], 0.0
            for i, sd in enumerate(seeds):
                if i > 0 and time.time() + 1.3 * dur > until:
                    break
                if i == 0 and not first_ok:
                    break
                t = time.time()
                spl = splits5 if fixed_cv else M.make_splits(task, y, 5, sd)
                o, tt, mk = M.cv_run(X, y, Xte, task, state["params"], kind, spl, sd, state["log_target"], K, HARD_END, gids)
                dur = time.time() - t
                oofs.append(o); tes.append(tt); masks.append(mk)
                log(f"{kind} seed {sd}: {dur:.0f}s")
            if not oofs:
                return None
            mk = np.logical_and.reduce(masks)
            return np.mean(oofs, 0), np.mean(tes, 0), mk

        def finish(oof, te, mk, tag):
            sc = M.score_oof(metric, y[mk], oof[mk], task)
            w = None
            if task == "classification" and metric in ("macro_f1", "weighted_f1"):
                w, s2 = M.tune_class_weights(oof[mk], y[mk], K, metric); log(f"{tag} CV {sc:.5f} -> tuned {s2:.5f}")
            else:
                log(f"{tag} CV {sc:.5f}")
            save(to_sub(te, w), out_path); log(f"{tag} submission saved")
            return sc

        t_fin = time.time()
        until_l = t_fin + 0.25 * max(HARD_END - t_fin, 0)
        res_l = run_seeds("lgb", [7, 11, 13], until_l)
        oof_l, te_l, m1 = res_l
        sc = finish(oof_l, te_l, m1, "lgb")

        if M.HAVE_CB and time.time() < HARD_END - 0.1 * TOTAL:
            try:
                res_c = run_seeds("cat", [7, 11], HARD_END)
                if res_c is not None:
                    oof_c, te_c, m2 = res_c
                    mm = m1 & m2
                    if mm.sum() > 0.9 * len(y):
                        bestw, bests = 1.0, -1e18
                        for a in np.linspace(0, 1, 11):
                            s_ = M.score_oof(metric, y[mm], (a * oof_l + (1 - a) * oof_c)[mm], task)
                            if s_ > bests: bestw, bests = a, s_
                        log(f"blend a={bestw:.1f} CV={bests:.5f} (lgb {sc:.5f})")
                        finish(bestw * oof_l + (1 - bestw) * oof_c, bestw * te_l + (1 - bestw) * te_c, mm, "blend")
            except Exception as e:
                log("catboost skipped:", type(e).__name__, e)
    except Exception as e:
        log("final stage failed, остаётся предыдущий сохранённый файл:", repr(e))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("FATAL", repr(e)); raise
