# PN-TRACE v1 — ячейка B: дамп трассы + сверка с PNDIAG + RUN-FP. Запускать ПОСЛЕ ячейки запуска. Read-only.
# Пишет pntrace_<tag>.npz (колонки как в _PNT_COLS + fp/real_raw/real_clip/pred/births) и pntrace_<tag>_meta.json в DIR.
import json, os, zlib
import numpy as np


def _pnt_arrays(run=None):
    rows = _PNT['rows']
    if not rows:
        return None
    M = np.array(rows, dtype=np.float64)
    keep = np.ones(len(M), dtype=bool)
    if run is not None:
        keep = M[:, _PNT_COLS.index('run')] == run
    A = {c: M[keep, i] for i, c in enumerate(_PNT_COLS)}
    for c in ('run', 't', 'agent', 'state', 'cur_before', 'node', 'created', 'arrival', 'node_nvis', 'node_hits', 'n_places',
              'top1_id', 'top2_id', 'recog_id', 'recog_n', 'attr', 'stance', 'ch_age'):
        A[c] = A[c].astype(np.int64)
    A['fp'] = np.stack(_PNT['fp'])[keep]
    A['real_raw'] = np.stack(_PNT['real_raw'])[keep]
    A['real_clip'] = np.stack(_PNT['real_clip'])[keep]
    A['pred'] = np.stack(_PNT['pred'])[keep]
    B = _PNT['births']
    bk = [b for b in B if run is None or b[0] == run]
    if bk:
        A['birth_scalar'] = np.array([[b[0], b[1], b[2], b[3], b[4], b[5], b[6]] for b in bk], dtype=np.float64)  # run,t,agent,node,nvis,cx,cy
        A['birth_fp_mean'] = np.stack([b[7] for b in bk])
        A['birth_pi_mean'] = np.array([b[8] for b in bk], dtype=np.float64)
    return A


def _pnt_report(tag):
    runs = _PNT['runs']
    print("\n" + "=" * 78)
    print(f"[PN-TRACE] ДАМП И СВЕРКА | tag={tag}")
    print("=" * 78)
    print(f"  on={_PNT['on']} | строк={len(_PNT['rows'])} | отброшено по max_rows={_PNT['dropped']} | ошибок трассера={dict(_PNT['errors']) or 0}")
    if _PNT['first_err']:
        print("  первая ошибка:\n" + _PNT['first_err'])
    if not _PNT['rows']:
        print("  ❌ трасса пуста: ячейка A не была запущена до прогона, либо _PNT['on']=False, либо NAV_PLACE_NODES выключен.")
        _e = _PN_LAST.get('eng') if '_PN_LAST' in globals() else None
        try:
            print(f"  RUN-FP (для сверки ON/OFF) = {_state_fingerprint(_e)}" if _e is not None else "  RUN-FP: движок не найден")
        except Exception as _x:
            print(f"  RUN-FP: не получен ({_x!r})")
        return None
    cnt = {}
    for r in _PNT['rows']:
        cnt[r[0]] = cnt.get(r[0], 0) + 1
    main_run = max(cnt, key=lambda k: cnt[k])        # основной прогон = самый длинный; self-test'ы короче
    print(f"  прогонов в трассе: {len(runs)} (строк по прогонам {cnt}); основной = {main_run}")
    A = _pnt_arrays(main_run)
    n = len(A['t'])
    st = np.bincount(A['state'][A['state'] >= 0], minlength=len(_PNT_STATES))
    print(f"  основной прогон: строк={n} агентов={len(np.unique(A['agent']))} тики {A['t'].min()}..{A['t'].max()}")
    print("  состояния: " + " ".join(f"{s}={int(st[i])}" for i, s in enumerate(_PNT_STATES)))
    print(f"  узлов создано={int(A['created'].sum())} | приходов в известный узел (arrival)={int(A['arrival'].sum())} | "
          f"строк без узла (node=-1)={int((A['node'] < 0).sum())}")
    fresh = int((A['ch_age'] >= 0).sum())
    print(f"  каналы когниции: свежих (ch_age известен)={fresh}/{n}; доля NaN в pred={float(np.isnan(A['pred']).any(axis=1).mean()):.3f}")
    # --- сверка со счётчиками PNDIAG (живые счётчики движка): трассер обязан видеть КАЖДЫЙ вызов основного прогона ---
    ok_all = True
    eng = _PN_LAST.get('eng') if '_PN_LAST' in globals() else None
    if eng is not None:
        c = _fch(eng).st
        # NB: счётчики движка накапливаются за прогон; трассер — только за основной прогон (по числу строк). Если self-test'ы
        # тоже писали в этот движок, расхождение будет объяснимо; для чистого прогона ожидаем точное равенство.
        pairs = [("new", int(A['created'].sum()), int(c['place_new'])),
                 ("recognized", int(st[_PNT_STATE_ID['recognized']]), int(c['place_hit'] + c['place_cont'] + c['diag_rec_stay'])),
                 ("ambiguous", int(st[_PNT_STATE_ID['ambiguous']]), int(c['place_ambig'])),
                 ("uncertain", int(st[_PNT_STATE_ID['uncertain']]), int(c['place_uncertain'])),
                 # place_tentative растёт и в замерах, вернувших 'uncertain' при идущем подтверждении (recog_id>=0): считаем оба вида
                 ("tentative", int(st[_PNT_STATE_ID['tentative']]) + int(((A['state'] == _PNT_STATE_ID['uncertain']) & (A['recog_id'] >= 0)).sum()),
                  int(c['place_tentative'])),
                 # place_pending в движке растёт и на 3-м наблюдении, которое создаёт узел (см. баланс PNDIAG: added = 3*new + ...)
                 ("pending+new", int(st[_PNT_STATE_ID['pending']]) + int(A['created'].sum()), int(c['place_pending']))]
        print("  сверка с PNDIAG (трасса | движок):")
        for name, a, b in pairs:
            flag = "✅" if a == b else "❌"
            ok_all &= (a == b)
            print(f"    {flag} {name:<11} {a:>7} | {b:>7}")
        print("  ИТОГ СВЕРКИ: " + ("✅ трассер видит все вызовы" if ok_all else
              "❌ расхождения — трассер что-то пропускает или в счётчики писал другой прогон; НЕ использовать трассу до выяснения"))
        # RUN-FP: отпечаток итогового состояния по штатному механизму проекта; сравнить ON/OFF и OFF/OFF на одном GIVEN_SEED и STEPS
        try:
            fpr = _state_fingerprint(eng)
            print(f"  RUN-FP (field, rng_pos, n_agents, agents) = {fpr}")
        except Exception as e:
            fpr = None
            print(f"  RUN-FP: не получен ({e!r})")
    else:
        fpr = None
        print("  движок (_PN_LAST['eng']) не найден: сверка и RUN-FP пропущены.")
    # --- запись на диск ---
    path = None
    if 'DIR' in globals():
        path = os.path.join(DIR, f"pntrace_{tag}")
        try:
            np.savez_compressed(path + ".npz", **A)
            keys = ('WORLD_SIZE', 'NAV_RING_R', 'NAV_SENSE_EVERY', 'NAV_LANDMARK_SEED', 'NAV_LANDMARK_SMOOTH', 'GIVEN_SEED',
                    'NAV_PLACE_NODES', 'NAV_PLACE_PI_GATE', 'NAV_PLACE_PI_SCALE', 'NAV_PLACE_PI_DRIFT', 'NAV_PLACE_FP_SIGMA_MIN',
                    'NAV_PLACE_FP_SIGMA_MAX', 'NAV_PLACE_NOVEL_CONFIRM', 'NAV_PLACE_NOVEL_FP_TOL', 'NAV_PLACE_COARSE_FP',
                    'NAV_PLACE_MIN_CONF', 'NAV_PLACE_MAX_ERR', 'NAV_PLACE_MAX_NODES', 'NAV_PLACE_MAX_VISITS',
                    'NAV_PI_NOISE', 'NAV_TURN_NOISE', 'NAV_SPATIAL_TENSION_GAIN', 'PN_V2_APPLY_STANCE', 'PN3_PROPOSALS',
                    'NAV_GATE_FLOOR', 'NEG_SAFE_TRANSFER')
            meta = {"tag": tag, "rows": int(n), "main_run": int(main_run), "runs": runs, "cols": list(_PNT_COLS),
                    "states": list(_PNT_STATES), "attrs": list(_PNT_ATTRS), "stances": list(_PNT_STANCES), "channels": list(_PNT_CHANNELS),
                    "config": {k: getattr(Config, k, None) for k in keys}, "dropped": int(_PNT['dropped']),
                    "errors": dict(_PNT['errors']), "pntrace_on": bool(_PNT['on']), "run_fp": fpr, "sync_check_ok": bool(ok_all),
                    "note": "center_x/y = истинный центр тела (диагностика; агенту недоступен); fp пересчитывается из центра детерминированно; "
                            "pi_x/pi_y = path integration ДО привязки к узлу (в собственной системе агента, старт (0,0))."}
            with open(path + "_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
            print(f"  [PN-TRACE] сохранено: {path}.npz ({os.path.getsize(path + '.npz') / 1e6:.1f} МБ) + _meta.json")
        except Exception as e:
            print(f"  [PN-TRACE] запись не удалась: {e!r}")
    print("=" * 78)
    return A


_pnt_A = _pnt_report(_tag_main) if '_tag_main' in globals() else _pnt_report('untagged')
