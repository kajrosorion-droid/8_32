# PN-TRACE v1 — read-only трассер PlaceNode (этап 1: линейка геометрии узлов; этап 1.5: предсказательная ценность места).
# ГАРАНТИИ (проверяются _pnt_selftest() и сверкой _state_fingerprint ON/OFF):
#   * не пишет в агента и в его _nav: ни одного нового атрибута (иначе он попал бы в _agent_crc/_state_fingerprint);
#   * не вызывает RNG и phi_hash; не трогает поле; не меняет intent/goals/PN-узлы;
#   * все данные — только в модульном словаре _PNT; любая ошибка трассера проглатывается и считается (_PNT['errors']).
# ГДЕ СТОИТ: ПОСЛЕ ячеек NEGATIVE FIELD TRACE / NEG-FIX / NAV-GATE и ПЕРЕД ячейкой запуска. Повторный запуск безопасен.
# ВЫКЛЮЧАТЕЛЬ: _PNT['on'] = False (хуки остаются, но ничего не пишут и сразу делегируют в super()).
import json as _pnt_json
import os as _pnt_os
import zlib as _pnt_zlib
import numpy as _pnt_np
from collections import Counter as _pnt_Counter

# Словари кодов. Порядок состояний = все значения rec['state'], которые возвращает _nav_place_recognition.
_PNT_STATES = ('unknown', 'pending', 'tentative', 'ambiguous', 'uncertain', 'recognized', 'new')
_PNT_STATE_ID = {s: i for i, s in enumerate(_PNT_STATES)}
_PNT_ATTRS = ('none', 'self_consistency', 'self_motion', 'world_change', 'place_unknown', 'place_shifted', 'unexplained')
_PNT_ATTR_ID = {s: i for i, s in enumerate(_PNT_ATTRS)}
_PNT_STANCES = ('none', 'investigating', 'accommodating', 'defensive', 'denying')
_PNT_STANCE_ID = {s: i for i, s in enumerate(_PNT_STANCES)}
# Порядок 8 каналов когниции (update_model_part1: raw_vals_base + unknown, binding, trust); сверен с таблицей [G1] отчёта GAP.
_PNT_CHANNELS = ('energy', 'flux', 'scar', 'noise', 'vorticity', 'unknown', 'binding', 'trust')
# Порядок скалярных колонок строки замера (одна строка = один вызов _nav_place_recognition нормального агента).
_PNT_COLS = ('run', 't', 'agent', 'cx', 'cy', 'state', 'conf', 'err', 'pi_err', 'pi_x', 'pi_y', 'th', 'path',
             'cur_before', 'node', 'created', 'arrival', 'node_nvis', 'node_hits', 'n_places',
             'top1_id', 'top1_err', 'top1_conf', 'top2_id', 'top2_err', 'top2_conf', 'recog_id', 'recog_n',
             'spatial_err_post', 'place_conf_post', 'pred_error', 'cog_tension', 'attr', 'stance', 'ch_age')
_PNT_NAN = float('nan')

if '_PNT' not in globals():
    _PNT = {'on': True, 'max_rows': 600000, 'rows': [], 'fp': [], 'real_raw': [], 'real_clip': [], 'pred': [],
            'births': [], 'stash': {}, 'runs': [], 'run': -1, 'last_world': None, 'last_now': -1,
            'dropped': 0, 'commits': 0, 'errors': _pnt_Counter(), 'first_err': None, 'installed': []}


def _pnt_err(tag, e):
    _PNT['errors'][tag] += 1
    if _PNT['first_err'] is None:
        import traceback
        _PNT['first_err'] = f"{tag}: {e!r}\n{traceback.format_exc()}"
        print(f"[PN-TRACE] ⚠️ ошибка трассера ({tag}): {e!r} — прогон не затронут, трассер продолжает", flush=True)


def _pnt_f32(v, n):
    try:
        a = _pnt_np.asarray(v, dtype=_pnt_np.float32).reshape(-1)
        if a.size == n:
            return a.copy()
    except Exception:
        pass
    return _pnt_np.full(n, _PNT_NAN, dtype=_pnt_np.float32)


def _pnt_new_run_check(agent, now):
    w = getattr(agent, 'world', None)
    wid = id(w)
    if wid != _PNT['last_world'] or now < _PNT['last_now']:
        _PNT['run'] += 1
        _PNT['runs'].append({'run': _PNT['run'], 'start_tick': int(now), 'world_id': int(wid),
                             'seed': getattr(Config, 'GIVEN_SEED', None) if 'Config' in globals() else None})
    _PNT['last_world'] = wid
    _PNT['last_now'] = int(now)


class PnTracePatternMixin:
    """Пассивный наблюдатель над Pattern._nav_place_recognition и Pattern._nav_tick. Всё делегируется в super()."""

    def _nav_place_recognition(self, fp, now, s):
        if not _PNT['on']:
            return super()._nav_place_recognition(fp, now, s)
        pre = None
        try:
            n = getattr(self, '_nav', None)
            if n is not None and getattr(self, 'world', None) is not None:
                cur = n.get('current_place')
                cx, cy = self._nav_body_center()          # чтение self.cells; тот же вызов делает _nav_tick
                pre = {'c': (float(cx), float(cy)), 'pi': (float(n['pi'][0]), float(n['pi'][1])), 'th': float(n['th']),
                       'path': float(n['path']), 'cur': (int(cur['id']) if cur is not None else -1),
                       'fp': (_pnt_f32(fp, 9) if fp is not None else _pnt_f32(None, 9))}
        except Exception as e:
            _pnt_err('pre', e)
        res = super()._nav_place_recognition(fp, now, s)
        try:
            if pre is not None:
                n = self._nav
                cur = n.get('current_place')
                post = {'state': str(res.get('state')), 'conf': float(res.get('conf', 0.0)), 'err': float(res.get('err', 0.0)),
                        'pi_err': float(res.get('pi_err', 0.0)),
                        'node': (int(cur['id']) if cur is not None else -1),
                        'nvis': (len(cur['visits']) if cur is not None else 0),
                        'hits': (int(cur.get('n_hits', 0)) if cur is not None else 0),
                        'np': len(n['places']),
                        'recog': ((int(n['recog']['id']), int(n['recog']['n'])) if n.get('recog') else (-1, 0)),
                        'cand': list(n.get('place_candidates') or [])[:2],
                        'fp_mean': (_pnt_f32(cur['fp_mean'], 9) if (cur is not None and str(res.get('state')) == 'new') else None),
                        'pi_mean': ((float(cur['pi_mean'][0]), float(cur['pi_mean'][1])) if (cur is not None and str(res.get('state')) == 'new') else None)}
                _PNT['stash'][int(self.id)] = (int(now), pre, post)
        except Exception as e:
            _pnt_err('post', e)
        return res

    def _nav_tick(self, field, t):
        if not _PNT['on']:
            return super()._nav_tick(field, t)
        _PNT['stash'].pop(int(self.id), None)
        out = super()._nav_tick(field, t)
        try:
            st = _PNT['stash'].pop(int(self.id), None)
            if st is not None:
                _pnt_commit(self, st)
        except Exception as e:
            _pnt_err('commit', e)
        return out


def _pnt_commit(agent, st):
    now, pre, post = st
    if len(_PNT['rows']) >= int(_PNT['max_rows']):
        _PNT['dropped'] += 1
        return
    _pnt_new_run_check(agent, now)
    n = agent._nav
    p2 = n.get('pn2')
    cand = post['cand']
    c1 = cand[0] if len(cand) > 0 else (-1, _PNT_NAN, _PNT_NAN)
    c2 = cand[1] if len(cand) > 1 else (-1, _PNT_NAN, _PNT_NAN)
    age = getattr(agent, '_err_dim_age', None)
    try:
        ch_age = int(age[0]) if age else -1
    except Exception:
        ch_age = -1
    state = post['state']
    created = 1 if state == 'new' else 0
    arrival = 1 if (state == 'recognized' and post['node'] != pre['cur']) else 0
    row = (int(_PNT['run']), int(now), int(agent.id), pre['c'][0], pre['c'][1], _PNT_STATE_ID.get(state, -1),
           post['conf'], post['err'], post['pi_err'], pre['pi'][0], pre['pi'][1], pre['th'], pre['path'],
           pre['cur'], post['node'], created, arrival, post['nvis'], post['hits'], post['np'],
           int(c1[0]), float(c1[1]), float(c1[2]), int(c2[0]), float(c2[1]), float(c2[2]), post['recog'][0], post['recog'][1],
           float(getattr(agent, 'spatial_pred_error', _PNT_NAN)), float(getattr(agent, 'place_confidence', _PNT_NAN)),
           float(getattr(agent, 'pred_error', _PNT_NAN)), float(getattr(agent, 'cognitive_tension', _PNT_NAN)),
           _PNT_ATTR_ID.get(p2['last_attr'], -1) if p2 else -1, _PNT_STANCE_ID.get(p2['stance'], -1) if p2 else -1, ch_age)
    _PNT['rows'].append(row)
    _PNT['fp'].append(pre['fp'])
    _PNT['real_raw'].append(_pnt_f32(getattr(agent, '_avg_real_raw_dim', None), 8))
    _PNT['real_clip'].append(_pnt_f32(getattr(agent, '_avg_real_dim', None), 8))
    _PNT['pred'].append(_pnt_f32(getattr(agent, '_pred_dim', None), 8))
    if created and post['fp_mean'] is not None:
        _PNT['births'].append((int(_PNT['run']), int(now), int(agent.id), int(post['node']), int(post['nvis']),
                               pre['c'][0], pre['c'][1], post['fp_mean'], post['pi_mean']))
    _PNT['commits'] += 1


def _pnt_reset():
    """Очистить накопленное (между прогонами). Хуки остаются."""
    for k in ('rows', 'fp', 'real_raw', 'real_clip', 'pred', 'births', 'runs'):
        _PNT[k].clear()
    _PNT['stash'].clear()
    _PNT.update(run=-1, last_world=None, last_now=-1, dropped=0, commits=0, first_err=None)
    _PNT['errors'].clear()


# --- установка (идемпотентно, тем же приёмом extend_class, что у NEG-TRACE) ---
if not callable(globals().get('extend_class')):
    raise RuntimeError("PN-TRACE: extend_class не найден — выполни ноутбук сверху вниз.")
for _need in ('_nav_tick', '_nav_place_recognition', '_nav_body_center'):
    if not hasattr(Pattern, _need):
        raise RuntimeError(f"PN-TRACE: у Pattern нет {_need}: слой NAV (ячейка 30) ещё не подключён — запусти его раньше.")
if any(k.__name__ == 'PnTracePatternMixin' for k in Pattern.__mro__):
    print("[PN-TRACE] уже установлен (идемпотентно); накопленные данные сохранены.")
else:
    Pattern = extend_class(Pattern, PnTracePatternMixin)
    _PNT['installed'].append('Pattern._nav_place_recognition/_nav_tick via extend_class')


def _pnt_selftest():
    """Быстрая самопроверка без запуска симуляции: хуки на верху MRO, делегируют, ничего не пишут в агента/_nav."""
    import copy as _copy
    ok, why = True, []
    owner = lambda name: next((k.__name__ for k in Pattern.__mro__ if name in k.__dict__), None)
    for m in ('_nav_place_recognition', '_nav_tick'):
        if owner(m) != 'PnTracePatternMixin':
            ok = False; why.append(f"{m}: верхний слой {owner(m)}, а не PnTracePatternMixin")
    saved = {k: (list(v) if isinstance(v, list) else v) for k, v in _PNT.items() if k in ('rows', 'fp', 'real_raw', 'real_clip', 'pred', 'births', 'runs')}
    saved_scalar = {k: _PNT[k] for k in ('run', 'last_world', 'last_now', 'dropped', 'commits', 'first_err', 'on')}
    saved_err = dict(_PNT['errors'])
    try:
        class _W: age = 7
        _w_shared = _W()          # один объект мира на обе заглушки: иначе repr различается адресом, а не состоянием

        class _Base:
            def __init__(self):
                self.id = 5; self.world = _w_shared; self.cells = {(1, 1)}; self.spatial_pred_error = 0.1; self.place_confidence = 0.5
                self.pred_error = 0.2; self.cognitive_tension = 0.3; self._err_dim_age = (7,)
                self._avg_real_raw_dim = _pnt_np.arange(8) * 0.1; self._avg_real_dim = _pnt_np.arange(8) * 0.05; self._pred_dim = _pnt_np.zeros(8)
                self._nav = {'pi': [1.0, 2.0], 'th': 0.5, 'path': 3.0, 'current_place': None, 'places': [], 'place_candidates': [],
                             'pn2': {'last_attr': 'self_consistency', 'stance': 'none'}}
            def _nav_body_center(self): return (10.5, 20.5)
            def _nav_place_recognition(self, fp, now, s):
                n = self._nav
                n['pi'][0] += 0.25                       # как настоящая: меняет pi
                n['current_place'] = {'id': 0, 'visits': [1, 2, 3], 'n_hits': 0, 'fp_mean': _pnt_np.zeros(9), 'pi_mean': [1.0, 2.0]}
                n['places'] = [n['current_place']]
                n['place_candidates'] = [(0, 0.1, 0.9)]
                return {'state': 'new', 'pi_err': 0.0, 'err': 0.0, 'conf': 0.4, 'conf_before': 1.0}
            def _nav_tick(self, field, t):
                self._nav_place_recognition(_pnt_np.zeros(9), int(self.world.age), None)
                self._nav['th'] += 0.1

        T = type('_T', (PnTracePatternMixin, _Base), {})
        a, b = _Base(), T()
        _PNT['on'] = True
        a._nav_tick(None, 7); b._nav_tick(None, 7)
        same = repr(sorted((k, repr(v)) for k, v in vars(a).items() if k != '_nav')) == repr(sorted((k, repr(v)) for k, v in vars(b).items() if k != '_nav'))
        same_nav = repr(a._nav) == repr(b._nav)
        if not (same and same_nav):
            ok = False; why.append("трассер изменил состояние агента/_nav относительно прогона без него")
        if len(_PNT['rows']) != len(saved['rows']) + 1:
            ok = False; why.append("после тика не появилась ровно одна строка")
        else:
            r = dict(zip(_PNT_COLS, _PNT['rows'][-1]))
            if not (r['created'] == 1 and r['node'] == 0 and abs(r['pi_x'] - 1.0) < 1e-9 and r['cx'] == 10.5 and r['state'] == _PNT_STATE_ID['new']):
                ok = False; why.append(f"строка записана неверно: {r}")
        _PNT['on'] = False
        n0 = len(_PNT['rows']); b2 = T(); b2._nav_tick(None, 7)
        if len(_PNT['rows']) != n0:
            ok = False; why.append("при _PNT['on']=False трассер всё равно пишет")
        _PNT['on'] = True

        class _Bad(_Base):
            def _nav_body_center(self): raise RuntimeError('boom')
        TB = type('_TB', (PnTracePatternMixin, _Bad), {})
        e0 = sum(_PNT['errors'].values())
        try:
            TB()._nav_tick(None, 7)
            if sum(_PNT['errors'].values()) == e0:
                ok = False; why.append("ошибка внутри трассера не посчитана")
        except Exception as ex:
            ok = False; why.append(f"ошибка трассера просочилась в прогон: {ex!r}")
    finally:
        for k, v in saved.items():
            _PNT[k][:] = v
        _PNT.update(saved_scalar)
        _PNT['errors'].clear(); _PNT['errors'].update(saved_err)
        _PNT['stash'].clear()
    print(f"[PN-TRACE selftest] {'✅ OK' if ok else '❌ FAIL: ' + '; '.join(why)}")
    return ok


_pnt_selftest()
print(f"[PN-TRACE] установлен | on={_PNT['on']} | max_rows={_PNT['max_rows']} | колонок={len(_PNT_COLS)} + fp(9) + real_raw/real_clip/pred(8)")
print("[PN-TRACE] после прогона запусти ячейку B (дамп + сверка со счётчиками PNDIAG + RUN-FP).")
