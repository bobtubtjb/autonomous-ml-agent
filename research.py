"""AutoResearch: гипотеза -> эксперимент -> оценка -> следующая итерация."""
import json, os, re, time, copy
import numpy as np
import pandas as pd
import ml_core as M

LLM_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
LLM_URL = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "Qwen3.8-27B"
MAX_LLM_CALLS = int(os.getenv("MAX_LLM_CALLS", "25"))

SYSTEM = """Ты — ML-исследователь, улучшающий табличную модель (LightGBM) по метрике кросс-валидации.
Предлагай ОДНУ проверяемую гипотезу за раз. Ответ — строго JSON без пояснений вокруг:
{"hypothesis": "кратко", "changes": {
  "lgbm_params": {"num_leaves": 31, ...},          // опционально
  "log_target": true,                              // опционально, только регрессия, target>-1
  "drop_columns": ["col"]                         // опционально
}}
Не повторяй уже проверенные гипотезы.
Рассуждай кратко (несколько предложений), сразу затем выдай JSON."""


def _client():
    if not (LLM_KEY and LLM_URL):
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=LLM_KEY, base_url=LLM_URL, timeout=120, max_retries=2)
    except Exception:
        return None


def chat(client, messages, max_tokens=4000, temperature=0.4, cap=16000):
    """Вызов LLM, устойчивый к «думающим» моделям: отключаем thinking, если сервер умеет, и берём ответ из любого поля."""
    last = None
    for extra in ({"chat_template_kwargs": {"enable_thinking": False}}, None):
        try:
            r = client.chat.completions.create(model=LLM_MODEL, messages=messages, temperature=temperature,
                                               max_tokens=max_tokens, extra_body=extra)
            break
        except Exception as e:
            last = e; r = None
    if r is None:
        raise last
    msg = r.choices[0].message
    txt = msg.content or ""
    if not txt.strip() and r.choices[0].finish_reason == "length" and max_tokens * 2 <= cap:
        print(f"[llm] ответ обрезан на {max_tokens} токенах (рассуждение), повтор с x2", flush=True)
        return chat(client, messages, max_tokens * 2, temperature, cap)
    if not txt.strip():
        txt = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        print(f"[llm] пустой content, finish_reason={r.choices[0].finish_reason}", flush=True)
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S)


def ask_llm(client, user_msg):
    txt = chat(client, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}], 5000, 0.4, cap=10000)
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if m is None:
        raise ValueError("LLM вернул ответ без JSON")
    return json.loads(m.group(0))


def profile(train_x, y, task, dictionary, metric, description):
    lines = [f"Задача: {task}, метрика (больше=лучше после знака): {metric}. Строк: {len(train_x)}, признаков: {train_x.shape[1]}."]
    if description:
        lines.append("Описание: " + description[:1200])
    if task == "regression":
        lines.append(f"target: mean={y.mean():.3g} std={y.std():.3g} min={y.min():.3g} max={y.max():.3g} skew={pd.Series(y).skew():.2f}")
    else:
        lines.append("классы: " + str(pd.Series(y).value_counts().head(15).to_dict()))
    for c in train_x.columns[:90]:
        s = train_x[c]
        d = dictionary.get(c, "")[:60]
        lines.append(f"- {c} [{s.dtype}] nuniq={s.nunique()} na={s.isna().mean():.0%} {d}")
    return "\n".join(lines)


def fallback_hypotheses(task):
    hs = [
        ("меньше листьев, сильнее регуляризация", {"lgbm_params": {"num_leaves": 31, "min_child_samples": 40, "reg_lambda": 5.0}}),
        ("больше листьев", {"lgbm_params": {"num_leaves": 127, "min_child_samples": 10}}),
        ("сильнее сэмплирование признаков", {"lgbm_params": {"colsample_bytree": 0.5}}),
        ("кол-во пропусков и NaN-статистики как признаки", {"fe_step": M.add_row_statistics}),
        ("частоты категорий", {"fe_step": M.add_category_frequencies}),
    ]
    if task == "classification":
        hs.insert(0, ("class_weight=balanced для редких классов", {"lgbm_params": {"class_weight": "balanced"}}))
    if task == "regression":
        hs.insert(0, ("лог-преобразование таргета", {"log_target": True}))
    return hs


def key_coverage(train_x, test_x, key):
    """(доля train-строк с повтором ключа внутри train, доля test-строк, чей ключ есть в train)."""
    full = pd.concat([train_x[key], test_x[key]], ignore_index=True)
    g = full.groupby(key, dropna=False, observed=True, sort=False).ngroup().values
    gtr, gte = g[:len(train_x)], g[len(train_x):]
    cnt = np.bincount(gtr, minlength=g.max() + 1)
    return float((cnt[gtr] >= 2).mean()), float((cnt[gte] > 0).mean())


def random_hypothesis(rng):
    """Шаг hill-climbing по параметрам, когда LLM-гипотезы и запасной список исчерпаны."""
    grid = {"num_leaves": [15, 31, 63, 127, 255], "min_child_samples": [5, 10, 20, 40, 80],
            "colsample_bytree": [0.4, 0.5, 0.7, 0.9], "subsample": [0.6, 0.8, 1.0],
            "reg_lambda": [0.0, 1.0, 5.0, 20.0], "reg_alpha": [0.0, 0.1, 1.0]}
    ks = rng.choice(list(grid), size=2, replace=False)
    ch = {str(k): grid[str(k)][int(rng.integers(len(grid[str(k)])))] for k in ks}
    return f"случайный поиск параметров {ch}", {"lgbm_params": ch}


ALLOWED_PARAMS = {  # только безопасные параметры LightGBM; objective/num_class/metric и т.п. LLM менять нельзя
    "num_leaves": (2, 1024), "min_child_samples": (1, 1000), "subsample": (0.3, 1.0), "subsample_freq": (0, 10),
    "colsample_bytree": (0.1, 1.0), "reg_lambda": (0.0, 200.0), "reg_alpha": (0.0, 100.0),
    "learning_rate": (0.01, 0.3), "max_depth": (-1, 32), "min_split_gain": (0.0, 10.0),
    "max_bin": (15, 511), "min_child_weight": (0.0, 100.0), "path_smooth": (0.0, 100.0)}


def clean_params(p):
    out = {}
    for k, v in (p or {}).items():
        if k == "class_weight" and v == "balanced":
            out[k] = v
        elif k in ALLOWED_PARAMS and isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
            lo, hi = ALLOWED_PARAMS[k]
            v = min(max(v, lo), hi)
            out[k] = int(v) if k in ("num_leaves", "min_child_samples", "subsample_freq", "max_depth", "max_bin") else float(v)
    return out


def merge_state(state, ch):
    s = copy.deepcopy(state)
    s["params"] = {**s["params"], **clean_params(ch.get("lgbm_params"))}
    if ch.get("log_target") is not None and s["task"] == "regression" and not s.get("force_log"):
        s["log_target"] = bool(ch["log_target"])
    if ch.get("target_encode"):
        ks = [[str(c) for c in k] for k in ch["target_encode"] if isinstance(k, list) and k]
        s["te_keys"] = s.get("te_keys", []) + [k for k in ks if k not in s.get("te_keys", [])]
    if ch.get("drop_columns"):
        s["drop"] = sorted(set(s["drop"]) | set(map(str, ch["drop_columns"])))
    if ch.get("fe_step"):
        s["fe_steps"] = s["fe_steps"] + [ch["fe_step"]]
    return s


def evaluate(state, train_x, y, test_x, K, metric, deadline, splits):
    X, Xte = M.prepare(train_x, test_x, state)
    p = {**state["params"], "learning_rate": max(state["params"].get("learning_rate", 0.05), 0.08)}
    gids = M.build_gids(X, Xte, state.get("te_keys"))
    oof, _, mask = M.cv_run(X, y, Xte, state["task"], p, "lgb", splits, 42,
                            state.get("log_target", False), K, deadline, gids)
    return M.score_oof(metric, y[mask], oof[mask], state["task"], tune=True)


def research_loop(state, train_x, y, test_x, K, metric, dictionary, description, t_end, splits, log=print):
    client = _client()
    t0 = time.time()
    best = evaluate(state, train_x, y, test_x, K, metric, t_end, splits)
    eval_time = time.time() - t0
    log(f"[research] baseline CV={best:.5f} ({eval_time:.0f}s/eval, LLM={'on' if client else 'off'})")
    history, calls, fb = [], 0, fallback_hypotheses(state["task"])
    pre = []
    if state["task"] == "regression" or K == 2:
        try:
            key, ksc = M.auto_entity_key(train_x, y)
            log(f"[research] авто-ключ сущности: {key} (coverage*R2={ksc:.3f})")
            if key and ksc >= 0.05:
                dup, hit = key_coverage(train_x, test_x, key)
                log(f"[research] ключ: повторы в train {dup:.0%}, попадание test в train {hit:.0%}")
                if hit < 0.6 * dup:
                    log("[research] ключ отклонён: test не пересекается с train так, как внутри train (разбиение по сущностям)")
                    key = None
            if key and ksc >= 0.05:
                pre.append((f"OOF target-encoding по ключу сущности {key}", {"target_encode": [key]}))
        except Exception as e:
            log(f"[research] авто-ключ не найден: {type(e).__name__}")
    prof = profile(train_x, y, state["task"], dictionary, metric, description)
    rng, n_rand, tried_rand = np.random.default_rng(0), 0, set()
    while time.time() + eval_time * 1.6 < t_end:
        ch = None
        if pre:
            hyp, ch = pre.pop(0)
        elif client and calls < MAX_LLM_CALLS:
            calls += 1
            try:
                msg = (prof + f"\n\nТекущий лучший CV: {best:.5f}\nТекущее состояние: params={state['params']}, "
                       f"log_target={state.get('log_target')}, drop={state['drop']}, шагов обработки признаков={len(state['fe_steps'])}, te_keys={state.get('te_keys', [])}\n"
                       "История экспериментов:\n" + ("\n".join(history[-12:]) or "пока пусто"))
                out = ask_llm(client, msg)
                hyp, ch = out.get("hypothesis", "?"), out.get("changes") or {}
                if not isinstance(ch, dict):
                    raise ValueError("LLM changes must be a JSON object")
                # Only built-in hypotheses may supply executable feature transformations.
                ch = {key: ch[key] for key in ("lgbm_params", "log_target", "drop_columns") if key in ch}
                if "lgbm_params" in ch and not isinstance(ch["lgbm_params"], dict):
                    raise ValueError("lgbm_params must be a JSON object")
                if "log_target" in ch and not isinstance(ch["log_target"], bool):
                    raise ValueError("log_target must be a boolean")
                if "drop_columns" in ch:
                    cols = ch["drop_columns"]
                    if not isinstance(cols, list) or not all(isinstance(col, str) for col in cols):
                        raise ValueError("drop_columns must be a list of strings")
                if not ch:
                    raise ValueError("LLM proposed no supported changes")
            except Exception as e:
                ch = None
                log(f"[research] LLM error: {type(e).__name__}")
                client = client if calls < MAX_LLM_CALLS else None
        if ch is None:
            if fb:
                hyp, ch = fb.pop(0)
            elif n_rand < 30:
                for _ in range(20):   # не повторять уже проверенные комбинации
                    hyp, ch = random_hypothesis(rng)
                    if hyp not in tried_rand:
                        break
                tried_rand.add(hyp); n_rand += 1
            else:
                break
        cand = merge_state(state, ch)
        t1 = time.time()
        try:
            sc = evaluate(cand, train_x, y, test_x, K, metric, t_end, splits)
        except Exception as e:
            history.append(f"- {hyp}: ОШИБКА {type(e).__name__}")
            log(f"[research] ОШИБКА в гипотезе «{str(hyp)[:70]}»: {type(e).__name__}: {str(e)[:160]}")
            continue
        eval_time = max(eval_time * 0.5 + (time.time() - t1) * 0.5, 1)
        eps = 0.0005 * abs(best)  # защита от переобучения на шум CV
        ok = sc > best + eps
        history.append(f"- {hyp}: CV={sc:.5f} ({'принято' if ok else 'отклонено'})")
        log(f"[research] {history[-1]}")
        if ok:
            best, state = sc, cand
    return state, best

