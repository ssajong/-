# -*- coding: utf-8 -*-
"""
🧪 건식전극 PTFE 결정화도 자동 분석 프로그램 (Streamlit)

실행 방법:
    pip install -r requirements.txt
    streamlit run app.py
(Windows는 run_windows.bat 더블클릭)

업로드한 Raw 데이터와 모든 설정은 app.py 옆의 dsc_ptfe_data.db(SQLite)에 자동 저장됩니다.
"""
import io
import os
import re
import json
import base64
import sqlite3
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch
import streamlit as st
import streamlit.components.v1 as components
from scipy.integrate import trapezoid
from scipy.signal import savgol_filter

# =====================================================================
# 전역 스타일
# =====================================================================
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['axes.unicode_minus'] = False


def _set_korean_font():
    """로트명에 한글이 들어가도 그래프에서 깨지지 않도록 설치된 한글 폰트를 찾아 지정."""
    candidates = ['Malgun Gothic', 'AppleGothic', 'Apple SD Gothic Neo', 'NanumGothic',
                  'Noto Sans CJK KR', 'Noto Sans KR', 'WenQuanYi Zen Hei']
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams['font.family'] = [name, 'DejaVu Sans']
            return name
    return None


KOREAN_FONT = _set_korean_font()

APP_VERSION = '8.0'
PROCESSES = ['Mixing', 'Kneading']
PROC_COLOR = {'Mixing': '#2a78d6', 'Kneading': '#eb6834'}
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER_COLOR = '#9a9993'
TEMP_KW = ('temp', '온도')
HF_KW = ('heat', 'flow', 'hf', '열')
BLANK_RE = re.compile(r'BLAN[CK]|빈\s*팬|empty\s*pan', re.I)
DB_PATH = os.environ.get('DSC_DB_PATH',
                         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dsc_ptfe_data.db'))

DEFAULT_SETTINGS = {
    'dh0': 85.4,            # 100% 결정질 PTFE 표준 엔탈피 (J/g)
    'rate': 10.0,           # 승온 속도 (°C/min)
    'proc': {
        'Mixing':   {'t_start': 312.0, 't_end': 354.0, 'view_min': 250.0, 'view_max': 400.0, 'ptfe': 1.0},
        'Kneading': {'t_start': 312.0, 't_end': 354.0, 'view_min': 250.0, 'view_max': 400.0, 'ptfe': 1.0},
    },
    'sg_window': 15, 'sg_poly': 3,
    'peak_min': 250.0, 'peak_max': 400.0,
    'onset_width': 5.0,
    'endo': '아래 (흡열 = 최솟값)',
    'deriv_ylim_mode': '자동',
    'deriv_ylim': 1.0,
    'heating_only': True,
    'header_start_row': 116,
    'min_points': 15,
    'use_blank': True, 'active_blank': None,
    'cv_warn': 5.0,
    'name_batch_mul': 100, 'name_feed_mul': 100, 'name_temp_mul': 10,
    'lot_memo': {},
    'dash_lot_sel': {},
    'dash_sample_off': [],
}


def deep_merge(base, over):
    for k, v in (over or {}).items():
        if isinstance(base.get(k), dict) and isinstance(v, dict) and k not in ('lot_memo', 'dash_lot_sel'):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


# =====================================================================
# 저장소 (SQLite) — 업로드 즉시 자동 저장
# =====================================================================
DB_LOCK = threading.Lock()
SAMPLE_COLS = ['id', 'filename', 'lot', 'rep', 'proc', 'ptfe', 't_start', 't_end', 'include', 'temp_col', 'hf_col',
               'hf_unit', 'mass', 'header_row', 'unit_row', 'is_blank', 'uploaded_at', 'version', 'meta']


@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("""CREATE TABLE IF NOT EXISTS samples (
        id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT UNIQUE, lot TEXT, rep INTEGER, proc TEXT,
        ptfe REAL, t_start REAL, t_end REAL, include INTEGER DEFAULT 1, temp_col TEXT, hf_col TEXT,
        hf_unit TEXT, mass REAL, header_row INTEGER, unit_row INTEGER, is_blank INTEGER DEFAULT 0,
        uploaded_at TEXT, version INTEGER DEFAULT 1, meta TEXT, data BLOB)""")
    conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    conn.commit()
    return conn


def db_exec(sql, params=()):
    with DB_LOCK:
        conn = get_conn()
        cur = conn.execute(sql, params)
        conn.commit()
        return cur


def db_samples():
    conn = get_conn()
    with DB_LOCK:
        df = pd.read_sql_query(f"SELECT {', '.join(SAMPLE_COLS)} FROM samples", conn)
    if not df.empty:
        df = df.set_index('id', drop=False)
    return df


def db_update(sid, **fields):
    if not fields:
        return
    keys = ', '.join(f'{k}=?' for k in fields)
    db_exec(f'UPDATE samples SET {keys} WHERE id=?', (*fields.values(), int(sid)))


def db_fetch_full(ids):
    conn = get_conn()
    with DB_LOCK:
        cur = conn.execute(f"SELECT * FROM samples WHERE id IN ({','.join('?' * len(ids))})", [int(i) for i in ids])
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def db_delete(ids):
    db_exec(f"DELETE FROM samples WHERE id IN ({','.join('?' * len(ids))})", [int(i) for i in ids])


def db_restore(rows):
    for r in rows:
        cols = list(r.keys())
        db_exec(f"INSERT OR REPLACE INTO samples ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                [r[c] for c in cols])


def load_settings():
    s = json.loads(json.dumps(DEFAULT_SETTINGS))
    row = get_conn().execute("SELECT value FROM settings WHERE key='app'").fetchone()
    if row:
        try:
            deep_merge(s, json.loads(row[0]))
        except Exception:
            pass
    return s


def save_settings(s):
    db_exec("INSERT OR REPLACE INTO settings (key, value) VALUES ('app', ?)", (json.dumps(s, ensure_ascii=False),))


@st.cache_data(max_entries=600, show_spinner=False)
def load_df(sid, version):
    row = get_conn().execute('SELECT data FROM samples WHERE id=?', (int(sid),)).fetchone()
    if row is None or row[0] is None:
        return pd.DataFrame()
    return pd.read_parquet(io.BytesIO(row[0]))


# =====================================================================
# 파일명 해석 (yymmdd APIM 10-1 K4 60R 18C #1 기타 (1))
# =====================================================================
def parse_filename(fname, S=None):
    S = S or DEFAULT_SETTINGS
    base = re.sub(r'\.(xlsx|xls|txt|csv|tsv|dat)$', '', str(fname), flags=re.I).strip()
    rep = 0
    m = re.search(r'\s*[(（]\s*(\d+)\s*[)）]\s*$', base)
    if m:
        rep = int(m.group(1))
        base = base[:m.start()]
    base = re.sub(r'\s+', ' ', base).strip()
    info = dict(lot=base, rep=rep, date=None, code=None, change=None, recipe=None, batch_kg=None, variant=None,
                feed=None, rpm=None, set_temp=None, batch=None, extras=[])
    toks = [t for t in re.split(r'[\s_]+', re.sub(r'(\S)#(\d+)', r'\1 #\2', base)) if t]
    for tk in toks:
        if info['date'] is None and re.fullmatch(r'\d{6}', tk):
            mo, d = int(tk[2:4]), int(tk[4:6])
            if 1 <= mo <= 12 and 1 <= d <= 31:
                info['date'] = f'20{tk[:2]}-{tk[2:4]}-{tk[4:6]}'
                continue
        if info['code'] is None and (x := re.fullmatch(r'AP([A-Z]*)M', tk, re.I)):
            info['code'] = tk.upper()
            info['change'] = x.group(1).upper() or None
            continue
        if info['recipe'] is None and (x := re.fullmatch(r'(\d+)-(\d+)', tk)):
            info['recipe'] = tk
            info['batch_kg'] = int(x.group(1)) * S['name_batch_mul']
            info['variant'] = int(x.group(2))
            continue
        if info['feed'] is None and (x := re.fullmatch(r'K(\d+(?:\.\d+)?)', tk, re.I)):
            info['feed'] = float(x.group(1)) * S['name_feed_mul']
            continue
        if info['rpm'] is None and (x := re.fullmatch(r'(\d+(?:\.\d+)?)R', tk, re.I)):
            info['rpm'] = float(x.group(1))
            continue
        if info['set_temp'] is None and (x := re.fullmatch(r'(\d+(?:\.\d+)?)C', tk, re.I)):
            info['set_temp'] = float(x.group(1)) * S['name_temp_mul']
            continue
        if info['batch'] is None and (x := re.fullmatch(r'#(\d+)', tk)):
            info['batch'] = int(x.group(1))
            continue
        info['extras'].append(tk)
    knead = info['feed'] is not None or info['rpm'] is not None or info['set_temp'] is not None or bool(re.search('니딩|knead', base, re.I))
    info['proc_guess'] = 'Kneading' if knead else 'Mixing'
    info['cond'] = ' '.join(t for t in toks if not re.fullmatch(r'#\d+', t))
    return info


def rep_label(rep):
    return '원본' if not rep else f'({rep})'


# =====================================================================
# Raw 파일 읽기 + 헤더 자동 탐지 (116행 이후 'Temperature' & 'Heat Flow' 행)
# =====================================================================
def _to_text(b):
    if b[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return b.decode('utf-16')
    for enc in ('utf-8-sig', 'cp949', 'latin-1'):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            continue
    return b.decode('utf-8', errors='replace')


def read_raw(name, data):
    """파일을 header=None 형태의 표로 읽는다."""
    if re.search(r'\.xlsx?$', name, re.I):
        return pd.read_excel(io.BytesIO(data), header=None, sheet_name=0)
    text = _to_text(data)
    lines = text.splitlines()
    head = '\n'.join(lines[:600])
    if '\t' in head:
        rows = [ln.split('\t') for ln in lines]
    elif head.count(';') and head.count(';') >= head.count(',') * 0.5:
        rows = [ln.split(';') for ln in lines]
    else:
        rows = [ln.split(',') for ln in lines]
    raw = pd.DataFrame(rows)
    return raw.replace({'': np.nan, None: np.nan})


def _num(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.number)):
        return float(v)
    t = str(v).strip().replace(' ', '')
    if not t:
        return np.nan
    t = t.replace(',', '') if '.' in t else t.replace(',', '.')
    try:
        return float(t)
    except ValueError:
        return np.nan


def _numeric_count(row):
    return int(sum(1 for v in row if not np.isnan(_num(v))))


def _row_has_keywords(row):
    s = ' '.join(str(v) for v in row if pd.notna(v)).lower()
    return any(k in s for k in TEMP_KW) and any(k in s for k in HF_KW)


def detect_header(raw, start_row=116):
    """start_row(엑셀 기준 1-based)부터 아래로 'Temperature'와 'Heat Flow'가 함께 있는 첫 행을 찾는다.
    못 찾으면 파일 앞부분도 검색 (TRIOS txt 대응). 반환: 0-based 인덱스 또는 None"""
    n = len(raw)
    start = max(0, int(start_row) - 1)

    def scan(rng):
        for i in rng:
            row = raw.iloc[i].tolist()
            if _row_has_keywords(row):
                nxt = [raw.iloc[k].tolist() for k in range(i + 1, min(n, i + 5))]
                if any(_numeric_count(r) >= 2 for r in nxt):
                    return i
        return None

    h = scan(range(start, n))
    if h is None:
        h = scan(range(0, min(start, n)))
    return h


def _dedupe(names):
    seen, out = {}, []
    for nme in names:
        if nme in seen:
            seen[nme] += 1
            out.append(f'{nme}.{seen[nme]}')
        else:
            seen[nme] = 0
            out.append(nme)
    return out


def _coerce(df):
    for c in df.columns:
        num = df[c].map(_num)
        if num.notna().mean() > 0.5:
            df[c] = num.astype(float)
        else:
            df[c] = df[c].astype(str).replace({'nan': ''})
    return df


def _meta_from_rows(raw, upto):
    meta = {'mass_mg': None, 'sample_name': '', 'run_date': ''}
    for i in range(0, min(upto, len(raw))):
        row = [v for v in raw.iloc[i].tolist()]
        if not row or pd.isna(row[0]):
            continue
        key = str(row[0]).strip()
        rest = ' '.join(str(v).strip() for v in row[1:] if pd.notna(v) and str(v).strip())
        if meta['mass_mg'] is None and re.search(r'sample\s*mass|^mass$|^weight$|sample\s*weight|시료\s*(질량|무게)|^질량', key, re.I):
            m = re.search(r'(-?[\d.,]+)\s*(mg|g)?', rest, re.I)
            if m:
                v = _num(m.group(1))
                if m.group(2) and m.group(2).lower() == 'g':
                    v *= 1000
                if v > 0:
                    meta['mass_mg'] = float(v)
        if not meta['sample_name'] and re.search(r'^sample\s*name|^시료\s*(명|이름)', key, re.I):
            meta['sample_name'] = rest
        if not meta['run_date'] and re.search(r'run\s*date|^date$|측정\s*일', key, re.I):
            meta['run_date'] = rest
    return meta


def parse_raw_file(name, data, start_row=116):
    """반환: dict(df, header_row, unit_row, units, meta) — 실패 시 ValueError"""
    raw = read_raw(name, data)
    h = detect_header(raw, start_row)
    units = {}
    unit_row = None
    if h is not None:
        header = [str(v).strip() if pd.notna(v) and str(v).strip() else f'Column{j + 1}'
                  for j, v in enumerate(raw.iloc[h].tolist())]
        header = _dedupe(header)
        nxt = raw.iloc[h + 1].tolist() if h + 1 < len(raw) else []
        if nxt and _numeric_count(nxt) == 0 and any(pd.notna(v) and str(v).strip() for v in nxt):
            unit_row = h + 1                                  # 헤더 바로 다음 행 = 단위 행 → 제외
            units = {header[j]: str(v).strip() for j, v in enumerate(nxt) if j < len(header) and pd.notna(v)}
            body = raw.iloc[h + 2:]
        else:
            body = raw.iloc[h + 1:]
        df = body.copy()
        df.columns = header[:df.shape[1]] + [f'Column{j + 1}' for j in range(len(header), df.shape[1])]
    else:
        # TRIOS 'Signal 1 / Signal 2 ...' 형식
        sig = {}
        for i in range(len(raw)):
            k = str(raw.iat[i, 0]).strip() if pd.notna(raw.iat[i, 0]) else ''
            m = re.fullmatch(r'Sig(?:nal)?\s*#?\s*(\d+)', k, re.I)
            if m and raw.shape[1] > 1:
                sig[int(m.group(1))] = str(raw.iat[i, 1]).strip()
        if not sig:
            raise ValueError("'Temperature'와 'Heat Flow' 항목 행을 찾지 못했습니다.")
        names = _dedupe([sig[k] for k in sorted(sig)])
        start = next((i for i in range(len(raw)) if _numeric_count(raw.iloc[i].tolist()) >= min(3, len(names))), None)
        if start is None:
            raise ValueError('숫자 데이터를 찾지 못했습니다.')
        df = raw.iloc[start:, :len(names)].copy()
        df.columns = names
        for nme in names:
            mm = re.search(r'\(([^)]+)\)\s*$', nme)
            if mm:
                units[nme] = mm.group(1)
        h = start - 1
        unit_row = -1   # Signal 형식 표시용
    df.columns = [str(c).strip() for c in df.columns]           # 컬럼명 앞뒤 공백 제거
    df = df.dropna(how='all')                                    # 완전히 빈 행 제거
    df = df.loc[:, df.notna().any()]
    df = _coerce(df.reset_index(drop=True))
    meta = _meta_from_rows(raw, h if h is not None else 0)
    return dict(df=df, header_row=h, unit_row=unit_row, units=units, meta=meta)


def default_columns(cols):
    """컬럼명에 온도/열흐름 키워드가 포함된 인덱스를 기본값으로 (없으면 0, 1)"""
    low = [c.lower() for c in cols]
    t_idx = next((i for i, c in enumerate(low) if any(k in c for k in TEMP_KW) and 'deriv' not in c), 0)
    h_idx = next((i for i, c in enumerate(low) if any(k in c for k in HF_KW) and not any(x in c for x in ('deriv', 'rev', 'phase', 'temp'))),
                 next((i for i, c in enumerate(low) if any(k in c for k in HF_KW) and i != t_idx), 1 if len(cols) > 1 else 0))
    return t_idx, h_idx


def hf_unit_of(col, units):
    u = (units.get(col) or (re.search(r'\(([^)]*)\)\s*$', col).group(1) if re.search(r'\(([^)]*)\)\s*$', col) else '')).lower().replace(' ', '')
    c = col.lower()
    if '/mg' in u:
        return 'W/g'
    if u in ('w/g',) or 'normaliz' in c:
        return 'W/g'
    if u == 'mw/g':
        return 'mW/g'
    if u == 'mw':
        return 'mW'
    if u == 'w':
        return 'W'
    return 'W/g'


def ingest_file(name, data, S):
    """업로드 파일을 해석해 DB에 저장(같은 파일명은 덮어쓰기). 반환: (성공여부, 메시지)"""
    p = parse_raw_file(name, data, S['header_start_row'])
    df, units = p['df'], p['units']
    signal_fmt = p['unit_row'] == -1
    if signal_fmt:
        p['unit_row'] = None
    if df.shape[1] < 2:
        raise ValueError('데이터 컬럼이 2개 미만입니다.')
    cols = list(df.columns)
    ti, hi = default_columns(cols)
    tcol, hcol = cols[ti], cols[hi]
    unit = hf_unit_of(hcol, units)
    info = parse_filename(name, S)
    is_blank = 1 if BLANK_RE.search(name) or BLANK_RE.search(p['meta'].get('sample_name') or '') else 0
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    meta = dict(p['meta'], units=units)
    existing = get_conn().execute('SELECT id, proc, ptfe, t_start, t_end, include, mass, version FROM samples WHERE filename=?',
                                  (name,)).fetchone()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if existing:
        sid, proc, ptfe, ts, te, inc, mass, ver = existing
        db_update(sid, lot=info['lot'], rep=info['rep'], temp_col=tcol, hf_col=hcol, hf_unit=unit,
                  mass=p['meta']['mass_mg'] or mass, header_row=p['header_row'], unit_row=p['unit_row'],
                  is_blank=is_blank, uploaded_at=now, version=(ver or 1) + 1,
                  meta=json.dumps(meta, ensure_ascii=False), data=buf.getvalue())
        replaced = True
    else:
        proc = info['proc_guess']
        d = S['proc'][proc]
        db_exec("""INSERT INTO samples (filename, lot, rep, proc, ptfe, t_start, t_end, include, temp_col, hf_col, hf_unit,
                   mass, header_row, unit_row, is_blank, uploaded_at, version, meta, data)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (name, info['lot'], info['rep'], proc, d['ptfe'], d['t_start'], d['t_end'], 1, tcol, hcol, unit,
                 p['meta']['mass_mg'], p['header_row'], p['unit_row'], is_blank, now, 1,
                 json.dumps(meta, ensure_ascii=False), buf.getvalue()))
        replaced = False
    h = p['header_row']
    start = int(S['header_start_row'])
    where = f'{start}행 이후 탐색 결과 -> ' if h is not None and h + 1 >= start else ''
    if signal_fmt:
        msg = f'파일 업로드 성공! (TRIOS Signal 형식 인식, {h + 2}행부터 데이터)'
    elif p['unit_row'] is not None:
        msg = f'파일 업로드 성공! ({where}{h + 1}행을 헤더로, {h + 2}행을 단위로 인식 완료)'
    else:
        msg = f'파일 업로드 성공! ({where}{h + 1}행을 헤더로 인식, 단위 행 없음)'
    if is_blank:
        extra = ['Blank(빈 팬) 파일로 등록']
    else:
        extra = [f'로트 [{info["lot"]}] {rep_label(info["rep"])}', f'Heat Flow 열: {hcol} ({unit})']
        if not replaced:
            extra.append(f'공정 자동 판별: {info["proc_guess"]}')
    if replaced:
        extra.append('같은 파일명 → 데이터 교체 (설정 유지)')
    return msg + '  \n' + ' · '.join(extra)


# =====================================================================
# 분석 알고리즘
# =====================================================================
def clean_xy(df, tcol, hcol, heating_only=True):
    """두 컬럼을 숫자로 변환하고 둘 다 NaN이 아닌 행만 사용"""
    t = pd.to_numeric(df[tcol], errors='coerce')
    h = pd.to_numeric(df[hcol], errors='coerce')
    m = t.notna() & h.notna()
    temp = t[m].to_numpy(float)
    hf = h[m].to_numpy(float)
    n_valid = len(temp)
    if heating_only and n_valid:
        imax = int(np.argmax(temp))            # 최고 온도 이후(냉각 구간)는 제외 → 1st Heating만 사용
        temp, hf = temp[:imax + 1], hf[:imax + 1]
    if len(temp):
        g = pd.DataFrame({'t': temp, 'h': hf}).groupby('t', as_index=False).mean()   # 정렬 + 등온 중복 제거
        temp, hf = g['t'].to_numpy(), g['h'].to_numpy()
    return temp, hf, n_valid


def run_analysis(temp, hf, p):
    """프롬프트 사양의 1~5 단계 계산. temp(°C), hf(W/g) 배열."""
    res = dict(ok=False, error=None, warnings=[])
    n = len(temp)
    if n < p['min_points']:
        res['error'] = f'유효 데이터가 {n}개로 {p["min_points"]}개 미만입니다. 컬럼 선택 또는 파일 구조를 확인해 주세요.'
        return res
    endo_down = p['endo'].startswith('아래')

    # 1. 스무딩 & 미분 (Savitzky-Golay, window 15 / polyorder 3)
    wl = int(p['sg_window'])
    if wl > n:
        wl = n
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 3)
    poly = min(int(p['sg_poly']), wl - 1)
    smoothed = savgol_filter(hf, window_length=wl, polyorder=poly)
    deriv = np.gradient(smoothed, temp)

    # 2. 용융 피크 탐색 (기본 250~400 °C 구간의 흡열 최솟값)
    #    베이스라인이 기울어 있으면 단순 최솟값이 구간 끝(예: 400 °C)에 걸리므로,
    #    적분 구간 안에서는 '직선 베이스라인 아래로 가장 깊은 점'을 피크로 사용
    t_start, t_end = float(p['t_start']), float(p['t_end'])
    mpk = (temp >= p['peak_min']) & (temp <= p['peak_max'])
    mint = (temp >= t_start) & (temp <= t_end)
    cand = mpk & mint
    if cand.sum() > 2:
        it = np.where(mint)[0]
        y0, y1, x0, x1 = smoothed[it[0]], smoothed[it[-1]], temp[it[0]], temp[it[-1]]
        dev = smoothed - (y0 + (y1 - y0) * (temp - x0) / (x1 - x0))
        idx = np.where(cand)[0]
        ip = idx[np.argmin(dev[idx])] if endo_down else idx[np.argmax(dev[idx])]
    else:
        if not mpk.any():
            res['warnings'].append(f'{p["peak_min"]:.0f}~{p["peak_max"]:.0f} °C 구간 데이터가 없어 전체 범위에서 피크를 탐색했습니다.')
            idx = np.arange(n)
        else:
            idx = np.where(mpk)[0]
        ip = idx[np.argmin(smoothed[idx])] if endo_down else idx[np.argmax(smoothed[idx])]
    t_peak = float(temp[ip])

    # 3. Onset 온도 (접선 외삽법) — 적분 시작 온도 ~ T_peak 구간의 최대 기울기점(변곡점)
    lead = np.where((temp >= t_start) & (temp <= t_peak))[0]
    if len(lead) <= 5:
        lead = np.where(temp <= t_peak)[0]
    t_onset, t_infl, slope, h_infl = t_peak - 10.0, None, None, None
    mb = (temp >= t_start - p['onset_width']) & (temp <= t_start)
    baseline_y = float(smoothed[mb].mean()) if mb.any() else (float(smoothed[lead[0]]) if len(lead) else float(smoothed[0]))
    if len(lead) > 5:
        ii = lead[np.argmin(deriv[lead])] if endo_down else lead[np.argmax(deriv[lead])]
        slope, h_infl, t_infl = float(deriv[ii]), float(smoothed[ii]), float(temp[ii])
        if slope != 0:
            t_onset = t_infl + (baseline_y - h_infl) / slope
        else:
            res['warnings'].append('변곡점 기울기가 0이어서 T_onset = T_peak − 10 °C로 표시합니다.')
    else:
        res['warnings'].append('Onset 계산 데이터가 부족하여 T_onset = T_peak − 10 °C로 표시합니다.')

    res.update(temp=temp, raw=hf, smoothed=smoothed, deriv=deriv, wl=wl, t_peak=t_peak, t_onset=float(t_onset),
               t_infl=t_infl, h_infl=h_infl, slope=slope, baseline_y=baseline_y, t_start=t_start, t_end=t_end)

    # 4. 엔탈피 적분 — 직선 베이스라인 차감, |차이| 적분 ÷ 승온속도
    mi = (temp >= t_start) & (temp <= t_end)
    if mi.sum() <= 2:
        res['error'] = '적분 구간의 데이터가 부족합니다. 사이드바의 적분 시작/종료 온도 범위를 확인해 주세요.'
        return res
    ti, hi = temp[mi], smoothed[mi]
    base = hi[0] + (hi[-1] - hi[0]) * (ti - ti[0]) / (ti[-1] - ti[0])
    area = float(trapezoid(np.abs(hi - base), ti))                 # W/g·°C
    rate = float(p['rate'])
    dh = area / rate * 60.0                                        # ÷ 승온속도(°C/min) → W·min/g, ×60 → J/g
    # 5. 결정화도
    w = float(p['ptfe']) / 100.0
    xc = dh / (w * float(p['dh0'])) * 100.0 if w > 0 else 0.0
    res.update(ok=True, ti=ti, hi=hi, base=base, area=area, dh=dh, xc=xc)

    if not (t_start <= t_peak <= t_end):
        res['warnings'].append(f'피크({t_peak:.1f} °C)가 적분 구간 밖에 있습니다. 적분 구간을 확인하세요.')
    elif t_peak - t_start < 2 or t_end - t_peak < 2:
        res['warnings'].append('피크가 적분 경계에 붙어 있습니다. 적분 구간이 피크 전체를 포함하는지 확인하세요.')
    if xc > 100:
        res['warnings'].append('결정화도가 100%를 넘습니다. PTFE 함량·ΔH₀·Heat Flow 단위를 확인하세요.')
    if w <= 0:
        res['warnings'].append('PTFE 함량이 0이라 결정화도를 0으로 표시합니다.')
    return res


@st.cache_data(max_entries=100, show_spinner=False)
def blank_curve(bid, version, tcol, hcol, unit, heating_only):
    df = load_df(bid, version)
    if df.empty or tcol not in df.columns or hcol not in df.columns:
        return None
    t, h, _ = clean_xy(df, tcol, hcol, heating_only)
    return t, h * (1000.0 if unit == 'W' else 1.0)


@st.cache_data(max_entries=3000, show_spinner=False)
def analyze_sample(sid, version, pjson):
    p = json.loads(pjson)
    df = load_df(sid, version)
    out = dict(ok=False, error=None, warnings=[], n_valid=0, t_range=None, blank_applied=False)
    if df.empty:
        out['error'] = '저장된 데이터를 읽을 수 없습니다.'
        return out
    if p['tcol'] not in df.columns or p['hcol'] not in df.columns:
        out['error'] = '선택한 컬럼이 데이터에 없습니다.'
        return out
    temp, hf, n_valid = clean_xy(df, p['tcol'], p['hcol'], p['heating_only'])
    out['n_valid'] = n_valid
    if len(temp):
        out['t_range'] = (float(temp.min()), float(temp.max()))
    unit = p['unit']
    if unit in ('mW', 'W'):
        mass = p.get('mass')
        if not mass or mass <= 0:
            out['error'] = f'Heat Flow 단위가 {unit}입니다. W/g로 환산하려면 시료 질량(mg)을 입력하세요.'
            return out
        mw = hf * (1000.0 if unit == 'W' else 1.0)
        if p.get('blank'):
            b = p['blank']
            bc = blank_curve(b['id'], b['version'], b['tcol'], b['hcol'], b['unit'], p['heating_only'])
            if bc is not None and len(bc[0]) > 2:
                mw = mw - np.interp(temp, bc[0], bc[1])
                out['blank_applied'] = True
        hf = mw / mass                                       # mW / mg = W/g
    elif unit == 'mW/g':
        hf = hf / 1000.0
    res = run_analysis(temp, hf, p)
    res['warnings'] = out['warnings'] + res['warnings']
    out.update(res)
    return out


def sample_params(row, S, blank_info):
    proc = row['proc'] if row['proc'] in PROCESSES else 'Mixing'
    p = dict(tcol=row['temp_col'], hcol=row['hf_col'], unit=row['hf_unit'] or 'W/g',
             mass=float(row['mass']) if pd.notna(row['mass']) else None,
             t_start=float(row['t_start']), t_end=float(row['t_end']), ptfe=float(row['ptfe']),
             dh0=float(S['dh0']), rate=float(S['rate']), sg_window=int(S['sg_window']), sg_poly=int(S['sg_poly']),
             peak_min=float(S['peak_min']), peak_max=float(S['peak_max']), onset_width=float(S['onset_width']),
             endo=S['endo'], heating_only=bool(S['heating_only']), min_points=int(S['min_points']), proc=proc,
             blank=blank_info if (row['hf_unit'] in ('mW', 'W')) else None)
    return json.dumps(p, sort_keys=True, ensure_ascii=False)


def get_blank_info(S, samples):
    if not S['use_blank'] or samples.empty:
        return None
    blanks = samples[samples['is_blank'] == 1]
    if blanks.empty:
        return None
    bid = S.get('active_blank')
    if bid not in blanks.index:
        bid = blanks.sort_values('uploaded_at').index[-1]
    b = blanks.loc[bid]
    if b['hf_unit'] not in ('mW', 'W'):
        return None
    return dict(id=int(bid), version=int(b['version']), tcol=b['temp_col'], hcol=b['hf_col'], unit=b['hf_unit'], name=b['filename'])


def status_of(r):
    if r.get('error'):
        return '오류'
    if r.get('warnings'):
        return '확인 필요'
    return '정상'


def build_results(samples, S):
    """모든 시료(Blank 제외)의 분석 결과 표"""
    rows = []
    if samples.empty:
        return pd.DataFrame()
    binfo = get_blank_info(S, samples)
    for sid, row in samples[samples['is_blank'] == 0].iterrows():
        r = analyze_sample(int(sid), int(row['version']), sample_params(row, S, binfo))
        info = parse_filename(row['filename'], S)
        meta = json.loads(row['meta'] or '{}')
        rows.append(dict(
            id=int(sid), 파일명=row['filename'], 로트=row['lot'], 반복=int(row['rep']), 시료=rep_label(int(row['rep'])),
            공정=row['proc'], 포함=bool(row['include']), PTFE=float(row['ptfe']), 적분시작=float(row['t_start']),
            적분종료=float(row['t_end']), 질량=row['mass'], 단위=row['hf_unit'],
            T_onset=r.get('t_onset') if r.get('ok') else np.nan, T_peak=r.get('t_peak') if r.get('ok') else np.nan,
            dH=r.get('dh') if r.get('ok') else np.nan, Xc=r.get('xc') if r.get('ok') else np.nan,
            ok=bool(r.get('ok')), 상태=status_of(r), 메시지=' / '.join(([r['error']] if r.get('error') else []) + r.get('warnings', [])),
            측정일=info['date'] or (meta.get('run_date') or '')[:10], 조건=info['cond'], 배치=info['batch'],
            투입량=info['feed'], RPM=info['rpm'], 설정온도=info['set_temp'], 공정변수=info['variant'],
            업로드=row['uploaded_at']))
    return pd.DataFrame(rows).set_index('id', drop=False) if rows else pd.DataFrame()


def stats(values):
    v = np.array([x for x in values if x is not None and np.isfinite(x)], float)
    n = len(v)
    if n == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, se=np.nan, cv=np.nan, min=np.nan, max=np.nan)
    sd = float(v.std(ddof=1)) if n > 1 else np.nan
    mean = float(v.mean())
    return dict(n=n, mean=mean, sd=sd, se=sd / np.sqrt(n) if n > 1 else np.nan,
                cv=sd / abs(mean) * 100 if n > 1 and mean else np.nan, min=float(v.min()), max=float(v.max()))


def lot_summary(res):
    rows = []
    if res.empty:
        return pd.DataFrame()
    for lot, g in res.groupby('로트', sort=False):
        valid = g[g['포함'] & g['ok']]
        sx = stats(valid['Xc'])
        rows.append({'로트': lot, '공정': g['공정'].iloc[0], '측정일': g['측정일'].iloc[0], 'N(유효/전체)': f'{sx["n"]}/{len(g)}',
                     'Xc 평균(%)': sx['mean'], 'SD': sx['sd'], 'CV(%)': sx['cv'],
                     'T_onset 평균(°C)': valid['T_onset'].mean() if len(valid) else np.nan,
                     'T_peak 평균(°C)': valid['T_peak'].mean() if len(valid) else np.nan,
                     'ΔHm 평균(J/g)': valid['dH'].mean() if len(valid) else np.nan,
                     'PTFE(wt%)': ', '.join(f'{x:g}' for x in sorted(g['PTFE'].unique()))})
    return pd.DataFrame(rows)


# =====================================================================
# 그래프 (matplotlib) + 이미지 복사/저장
# =====================================================================
def trunc_mid(s, n):
    s = str(s)
    if len(s) <= n:
        return s
    head = min(7, n // 3)
    return s[:head] + '…' + s[-(n - head - 1):]


def view_range(r, S, proc):
    d = S['proc'].get(proc, S['proc']['Mixing'])
    lo, hi = float(d['view_min']), float(d['view_max'])
    return min(lo, r['t_start'] - 5), max(hi, r['t_end'] + 5)


def make_report_fig(r, S, proc, sample_name):
    """프롬프트 사양: 2행 1열, figsize (8.5, 6.0), sharex, dpi 300"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True, dpi=300)
    temp, sm, dv = r['temp'], r['smoothed'], r['deriv']
    x0, x1 = view_range(r, S, proc)
    vm = (temp >= x0) & (temp <= x1)
    # 상단: 스무딩 Heat Flow
    ax1.plot(temp[vm], sm[vm], color='blue', lw=1.5, label='Smoothed Heat Flow')
    if r.get('ok'):
        ax1.fill_between(r['ti'], r['hi'], r['base'], color='orange', alpha=0.4,
                         label=f'Integration {r["t_start"]:.1f}–{r["t_end"]:.1f} °C (ΔHm {r["dh"]:.4f} J/g)')
        ax1.plot(r['ti'], r['base'], color='darkorange', lw=1.0, ls='-')
    ax1.axvline(r['t_peak'], color='red', ls='--', lw=1.2, label=f'T_peak = {r["t_peak"]:.2f} °C')
    ax1.axvline(r['t_onset'], color='purple', ls=':', lw=1.5, label=f'T_onset = {r["t_onset"]:.2f} °C')
    if vm.sum() > 10:   # 데이터 끝(승온→냉각 전환점)의 한 점 튐이 축 범위를 망가뜨리지 않도록
        lo, hi = np.percentile(sm[vm], 0.5), np.percentile(sm[vm], 99.5)
        if r.get('ok'):
            lo, hi = min(lo, r['hi'].min()), max(hi, r['hi'].max())
        pad = (hi - lo) * 0.12 or 1e-3
        ax1.set_ylim(lo - pad, hi + pad)
    ax1.set_ylabel('Heat Flow (W/g)')
    ax1.set_title('PTFE Melting Behavior & DSC Curve', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, ls='--', alpha=0.6)
    ax1.annotate(sample_name, xy=(0.01, 0.97), xycoords='axes fraction', ha='left', va='top', fontsize=8, color='#555555')
    # 하단: 1차 미분
    ax2.plot(temp[vm], dv[vm], color='green', lw=1.5, label='1st Derivative (dHF/dT)')
    ax2.axhline(0, color='gray', ls='--', lw=1)
    if S['deriv_ylim_mode'] == '고정':
        lim = float(S['deriv_ylim'])
    else:
        seg = dv[vm] if vm.any() else dv
        lim = max(abs(np.percentile(seg, 0.5)), abs(np.percentile(seg, 99.5))) * 1.25 or 1.0
    ax2.set_ylim(-lim, lim)
    ax2.set_xlim(x0, x1)
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Derivative')
    ax2.legend(loc='upper right')
    ax2.grid(True, ls='--', alpha=0.6)
    plt.tight_layout()
    return fig


def fig_bytes(fig, fmt='png', dpi=None):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight', facecolor='white')
    return buf.getvalue()


def show_figure(fig, base_name, key, dpi=None):
    """그래프 표시 + [이미지 복사] [PNG 저장] [SVG 저장]"""
    png = fig_bytes(fig, 'png', dpi)
    svg = fig_bytes(fig, 'svg')
    plt.close(fig)
    st.image(png, width='stretch')
    fname = re.sub(r'[\\/:*?"<>|]+', '_', base_name).strip().replace(' ', '_')[:120]
    c1, c2, c3 = st.columns([1, 1, 2.2])
    c1.download_button('💾 PNG', png, file_name=f'{fname}.png', mime='image/png', key=f'png_{key}',
                       on_click='ignore', width='stretch')
    c2.download_button('📐 SVG', svg, file_name=f'{fname}.svg', mime='image/svg+xml', key=f'svg_{key}',
                       on_click='ignore', width='stretch')
    with c3:
        copy_button(png)


def copy_button(png):
    b64 = base64.b64encode(png).decode()
    components.html(f"""
<style>
 body {{ margin:0; font-family: 'Malgun Gothic','Apple SD Gothic Neo',sans-serif; }}
 button {{ font: inherit; font-size:14px; padding:6px 14px; border:1px solid rgba(49,51,63,.2); border-radius:8px;
          background:#fff; cursor:pointer; color:#31333f; }}
 button:hover {{ border-color:#ff4b4b; color:#ff4b4b; }}
 span {{ font-size:12.5px; color:#555; margin-left:8px; }}
</style>
<button id="b">📋 이미지 복사</button><span id="m"></span>
<script>
const b64 = "{b64}";
document.getElementById('b').onclick = async () => {{
  const m = document.getElementById('m');
  try {{
    const bin = atob(b64); const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    await navigator.clipboard.write([new ClipboardItem({{'image/png': new Blob([arr], {{type: 'image/png'}})}})]);
    m.textContent = '✅ 복사 완료 — PPT·엑셀·메신저에 Ctrl+V';
  }} catch (e) {{
    m.textContent = '⚠️ 브라우저가 복사를 막았습니다. 그래프를 우클릭 → [이미지 복사]를 이용하세요.';
  }}
}};
</script>""", height=42)


def endo_curve(r):
    """스무딩 곡선 − 직선 베이스라인(적분 구간 양 끝점 연장)"""
    temp, ti, hi = r['temp'], r['ti'], r['hi']
    return r['smoothed'] - (hi[0] + (hi[-1] - hi[0]) * (temp - ti[0]) / (ti[-1] - ti[0]))


def overlay_fig(items, S, title):
    """items: [(label, color, r, proc)]"""
    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=200)
    x0s, x1s, ylo, yhi = [], [], [], []
    sign = -1 if S['endo'].startswith('아래') else 1
    for label, color, r, proc in items:
        x0, x1 = view_range(r, S, proc)
        x0s.append(x0)
        x1s.append(x1)
        m = (r['temp'] >= x0) & (r['temp'] <= x1)
        yv = sign * endo_curve(r)
        ax.plot(r['temp'][m], yv[m], color=color, lw=1.4, label=label)
        core = (r['temp'] >= r['t_start'] - 15) & (r['temp'] <= r['t_end'] + 15)   # 데이터 끝단 스무딩 튐 제외
        if core.any():
            ylo.append(float(yv[core].min()))
            yhi.append(float(yv[core].max()))
    ax.axhline(0, color='gray', ls='--', lw=1)
    if ylo:
        span = max(yhi) - min(ylo) or 1e-3
        ax.set_ylim(min(ylo) - 0.12 * span, max(yhi) + 0.12 * span)
    ax.set_xlim(min(x0s), max(x1s))
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Heat Flow − Baseline (W/g, endo ↑)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, ls='--', alpha=0.6)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if len(uniq) <= 12:
        ax.legend(uniq.values(), uniq.keys(), loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    plt.tight_layout()
    return fig


def err_bounds(s, mode):
    if s['n'] < 2 or mode == '표시 안 함':
        return np.nan, np.nan
    if mode == '표준편차 (SD)':
        return s['mean'] - s['sd'], s['mean'] + s['sd']
    if mode == '표준오차 (SE)':
        return s['mean'] - s['se'], s['mean'] + s['se']
    return s['min'], s['max']


def bar_fig(groups, view, err_mode, ref):
    """groups: list of dict(key, proc, items[(label, xc)], stats)"""
    n_bars = len(groups) if view == '로트별 (평균 ± 오차)' else sum(len(g['items']) for g in groups)
    fig, ax = plt.subplots(figsize=(max(8.5, min(18, 0.55 * n_bars + 4)), 4.8), dpi=200)
    rng = np.random.default_rng(0)
    ymax = 0
    if view == '로트별 (평균 ± 오차)':
        xs = np.arange(len(groups))
        for i, g in enumerate(groups):
            s = g['stats']
            ax.bar(i, s['mean'], width=0.62, color=PROC_COLOR[g['proc']], zorder=2)
            lo, hi = err_bounds(s, err_mode)
            if np.isfinite(lo):
                ax.errorbar(i, s['mean'], yerr=[[s['mean'] - lo], [hi - s['mean']]], fmt='none', ecolor='#1d2125',
                            elinewidth=1.4, capsize=5, zorder=4)
            vals = [v for _, v in g['items']]
            jit = (np.arange(len(vals)) - (len(vals) - 1) / 2) * min(0.08, 0.5 / max(1, len(vals)))
            ax.scatter(i + jit, vals, s=16, color='#1d2125', alpha=0.75, edgecolors='white', linewidths=0.8, zorder=5)
            top = max([s['mean'], hi if np.isfinite(hi) else 0] + vals)
            ymax = max(ymax, top)
            if len(groups) <= 20:
                ax.text(i, top + 0.8, f'{s["mean"]:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(xs)
        ax.set_xticklabels([trunc_mid(g['key'], 30) for g in groups], rotation=30, ha='right')
        title = f'Crystallinity by Lot (mean {"" if err_mode == "표시 안 함" else "± " + err_mode.split("(")[-1].rstrip(")")}, dots = samples)'
    else:
        x, ticks, labels, centers = 0.0, [], [], []
        for g in groups:
            start = x
            for lab, v in g['items']:
                ax.bar(x, v, width=0.7, color=PROC_COLOR[g['proc']], zorder=2)
                if n_bars <= 30:
                    ax.text(x, v + 0.6, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
                ticks.append(x)
                labels.append(lab)
                ymax = max(ymax, v)
                x += 1
            centers.append(((start + x - 1) / 2, g['key']))
            x += 0.8
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        for cx, key in centers:
            ax.annotate(trunc_mid(key, 26), xy=(cx, 0), xycoords=('data', 'axes fraction'), xytext=(0, -24),
                        textcoords='offset points', ha='center', va='top', fontsize=8, fontweight='bold', rotation=12)
        title = 'Crystallinity by Sample'
    if ref is not None and np.isfinite(ref[1]):
        ax.axhline(ref[1], color='#52514e', ls='--', lw=1.2, zorder=3)
        ax.annotate(f'Ref. {ref[1]:.1f}', xy=(1, ref[1]), xycoords=('axes fraction', 'data'), xytext=(-4, 4),
                    textcoords='offset points', ha='right', fontsize=8, color='#52514e')
        ymax = max(ymax, ref[1])
    ax.set_ylim(0, ymax * 1.15 if ymax > 0 else 1)
    ax.set_ylabel('Crystallinity Xc (%)')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, axis='y', ls='--', alpha=0.6, zorder=0)
    procs = [p for p in PROCESSES if any(g['proc'] == p for g in groups)]
    ax.legend(handles=[Patch(color=PROC_COLOR[p], label=p) for p in procs], loc='upper right')
    plt.tight_layout()
    return fig


XPARAMS = {'투입량 (kg/hr)': '투입량', 'RPM': 'RPM', '설정온도 (°C)': '설정온도', '측정일': '측정일',
           '공정변수 번호 (10-n)': '공정변수', '배치 번호 (#n)': '배치', 'PTFE 함량 (wt%)': 'PTFE'}


def scatter_fig(groups, xname, err_mode):
    col = XPARAMS[xname]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=200)
    pts = []
    for g in groups:
        xv = g['x'].get(col)
        if xv is None or (isinstance(xv, float) and not np.isfinite(xv)) or xv == '':
            continue
        if col == '측정일':
            try:
                xv = pd.Timestamp(xv)
            except Exception:
                continue
        pts.append((xv, g))
    for xv, g in pts:
        s = g['stats']
        lo, hi = err_bounds(s, err_mode)
        if np.isfinite(lo):
            ax.errorbar([xv], [s['mean']], yerr=[[s['mean'] - lo], [hi - s['mean']]], fmt='none',
                        ecolor=PROC_COLOR[g['proc']], capsize=4, elinewidth=1.3, zorder=2)
        ax.scatter([xv], [s['mean']], s=46, color=PROC_COLOR[g['proc']], edgecolors='white', linewidths=1.2, zorder=3)
        if len(pts) <= 10:
            ax.annotate(trunc_mid(g['key'], 22), (xv, s['mean']), xytext=(5, 5), textcoords='offset points', fontsize=7.5,
                        color='#52514e')
    ax.margins(x=0.18, y=0.15)
    ax.set_xlabel(xname)
    ax.set_ylabel('Crystallinity Xc (%)')
    ax.set_title(f'{xname} vs Crystallinity', fontweight='bold')
    ax.grid(True, ls='--', alpha=0.6)
    procs = [p for p in PROCESSES if any(g['proc'] == p for _, g in pts)]
    if procs:
        ax.legend(handles=[Patch(color=PROC_COLOR[p], label=p) for p in procs], loc='best')
    if col == '측정일':
        fig.autofmt_xdate()
    plt.tight_layout()
    return fig, len(pts)


# =====================================================================
# 화면 — 콜백 (변경 즉시 자동 저장)
# =====================================================================
def cb_sidebar(field):
    sid = st.session_state.get('cur_sid')
    if sid is None:
        return
    val = st.session_state[f'sb_{field}']
    if field == 'proc':
        d = load_settings()['proc'][val]
        db_update(sid, proc=val, t_start=d['t_start'], t_end=d['t_end'])
    else:
        db_update(sid, **{field: float(val)})
    st.session_state['_sb_sig'] = None


def cb_global():
    S = load_settings()
    S['dh0'] = float(st.session_state['sb_dh0'])
    S['rate'] = float(st.session_state['sb_rate'])
    save_settings(S)


def cb_sop_range():
    sid = st.session_state.get('cur_sid')
    row = get_conn().execute('SELECT proc FROM samples WHERE id=?', (sid,)).fetchone()
    if row:
        d = load_settings()['proc'][row[0]]
        db_update(sid, t_start=d['t_start'], t_end=d['t_end'])
        st.session_state['_sb_sig'] = None


def cb_apply_lot():
    sid = st.session_state.get('cur_sid')
    row = get_conn().execute('SELECT lot, proc, ptfe, t_start, t_end FROM samples WHERE id=?', (sid,)).fetchone()
    if row:
        db_exec('UPDATE samples SET proc=?, ptfe=?, t_start=?, t_end=? WHERE lot=? AND is_blank=0', (*row[1:], row[0]))
        st.session_state['_flash'] = ('success', f'로트 [{row[0]}]의 모든 시료에 공정·PTFE·적분 구간을 적용했습니다.')


def cb_nav(step):
    ids = st.session_state.get('_nav_ids', [])
    cur = st.session_state.get('cur_sid')
    if ids and cur in ids:
        st.session_state['cur_sid'] = ids[(ids.index(cur) + step) % len(ids)]


def cb_cols():
    sid = st.session_state.get('cur_sid')
    tcol, hcol = st.session_state['rep_tcol'], st.session_state['rep_hcol']
    meta = json.loads(get_conn().execute('SELECT meta FROM samples WHERE id=?', (sid,)).fetchone()[0] or '{}')
    db_update(sid, temp_col=tcol, hf_col=hcol, hf_unit=hf_unit_of(hcol, meta.get('units', {})))


def cb_mass():
    sid = st.session_state.get('cur_sid')
    v = st.session_state['rep_mass']
    db_update(sid, mass=float(v) if v and v > 0 else None)


def cb_unit():
    db_update(st.session_state.get('cur_sid'), hf_unit=st.session_state['rep_unit'])


def bump(name):
    st.session_state[name] = st.session_state.get(name, 0) + 1


def apply_editor(orig, edited, mapping, S):
    """data_editor 편집 내용을 DB에 저장. mapping: {표 컬럼: DB 필드}"""
    changed = 0
    for sid in edited.index:
        upd = {}
        for col, field in mapping.items():
            a, b = orig.at[sid, col], edited.at[sid, col]
            same = (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and a == b)
            if not same:
                if field == 'include':
                    upd[field] = int(bool(b))
                elif field == 'proc':
                    upd[field] = str(b)
                else:
                    upd[field] = None if pd.isna(b) else float(b)
        if 'proc' in upd and 't_start' not in upd and 't_end' not in upd:
            d = S['proc'][upd['proc']]
            upd.update(t_start=d['t_start'], t_end=d['t_end'])
        if upd:
            db_update(int(sid), **upd)
            changed += 1
    return changed


# =====================================================================
# 메인 화면
# =====================================================================
def main():
    st.set_page_config(page_title='건식전극 PTFE 결정화도 자동 분석기', layout='wide', page_icon='🧪')
    S = load_settings()
    samples = db_samples()
    res = build_results(samples, S)

    # ---------------- 사이드바 ----------------
    with st.sidebar:
        st.header('📂 분석 대상 시료')
        cur_row = None
        if res.empty:
            st.info('먼저 [📊 데이터 분석] 탭에서 Raw 파일을 업로드하세요.')
            st.session_state['cur_sid'] = None
        else:
            lots = list(dict.fromkeys(res.sort_values(['측정일', '로트'], ascending=[False, True])['로트']))
            if st.session_state.get('sb_lotfilter') not in ['(전체 로트)'] + lots:
                st.session_state['sb_lotfilter'] = '(전체 로트)'
            lot_f = st.selectbox('로트 필터', ['(전체 로트)'] + lots, key='sb_lotfilter')
            pool = res if lot_f == '(전체 로트)' else res[res['로트'] == lot_f]
            pool = pool.sort_values(['로트', '반복'])
            ids = [int(i) for i in pool.index]
            st.session_state['_nav_ids'] = ids
            if st.session_state.get('cur_sid') not in ids:
                st.session_state['cur_sid'] = ids[0]
            st.selectbox('시료 선택', ids, key='cur_sid',
                         format_func=lambda i: res.at[i, '파일명'] if i in res.index else str(i))
            st_now = res.at[st.session_state['cur_sid'], '상태']
            if st_now != '정상':
                st.caption(f'⚠️ 상태: {st_now} — {res.at[st.session_state["cur_sid"], "메시지"]}')
            cur_row = samples.loc[st.session_state['cur_sid']]

        # 위젯 값 동기화 (선택 시료가 바뀌거나 다른 화면에서 값이 바뀐 경우)
        if cur_row is not None:
            sig = (int(cur_row['id']), cur_row['proc'], float(cur_row['ptfe']), float(cur_row['t_start']), float(cur_row['t_end']))
            vals = dict(proc=cur_row['proc'] if cur_row['proc'] in PROCESSES else 'Mixing', ptfe=float(cur_row['ptfe']),
                        t_start=float(cur_row['t_start']), t_end=float(cur_row['t_end']))
        else:
            d = S['proc']['Mixing']
            sig = ('none',)
            vals = dict(proc='Mixing', ptfe=d['ptfe'], t_start=d['t_start'], t_end=d['t_end'])
        if st.session_state.get('_sb_sig') != sig:
            for k, v in vals.items():
                st.session_state[f'sb_{k}'] = v
            st.session_state['_sb_sig'] = sig
        gsig = (float(S['dh0']), float(S['rate']))
        if st.session_state.get('_sb_gsig') != gsig:
            st.session_state['sb_dh0'], st.session_state['sb_rate'] = gsig
            st.session_state['_sb_gsig'] = gsig
        no_sample = cur_row is None

        st.header('⚙️ 분석 파라미터 설정')
        st.selectbox('공정 선택', PROCESSES, key='sb_proc', on_change=cb_sidebar, args=('proc',), disabled=no_sample,
                     help='공정을 바꾸면 SOP 설정 탭의 해당 공정 적분 구간이 자동 적용됩니다.')
        st.number_input('PTFE 함량 (wt%)', min_value=0.0, step=0.1, key='sb_ptfe', on_change=cb_sidebar,
                        args=('ptfe',), disabled=no_sample, help='입력값 ÷ 100 → 분율로 계산에 사용')
        st.number_input('100% 결정질 PTFE 표준 엔탈피 ΔH₀ (J/g)', min_value=0.1, step=0.1, key='sb_dh0',
                        on_change=cb_global, help='모든 시료에 공통 적용')
        st.number_input('승온 속도 (°C/min)', min_value=0.1, step=1.0, key='sb_rate', on_change=cb_global,
                        help='모든 시료에 공통 적용')
        st.divider()
        st.subheader('📐 적분 및 온도 구간 설정')
        st.number_input('적분 시작 온도 (°C)', step=1.0, key='sb_t_start', on_change=cb_sidebar, args=('t_start',),
                        disabled=no_sample)
        st.number_input('적분 종료 온도 (°C)', step=1.0, key='sb_t_end', on_change=cb_sidebar, args=('t_end',),
                        disabled=no_sample)
        if st.session_state['sb_t_end'] <= st.session_state['sb_t_start']:
            st.error('적분 종료 온도가 시작 온도보다 커야 합니다.')
        c1, c2 = st.columns(2)
        c1.button('SOP 기본값', on_click=cb_sop_range, disabled=no_sample, width='stretch',
                  help='선택 시료의 적분 구간을 SOP 설정값으로 되돌립니다.')
        c2.button('로트 적용', on_click=cb_apply_lot, disabled=no_sample, width='stretch',
                  help='공정·PTFE·적분 구간을 같은 로트의 모든 시료에 적용합니다.')
        st.caption('💾 모든 값은 변경 즉시 자동 저장됩니다.')

    # ---------------- 본문 ----------------
    st.title('🧪 건식전극 PTFE 결정화도 자동 분석 프로그램')
    st.markdown(f"DSC 장비에서 내보낸 엑셀(`.xlsx`) Raw 파일의 **{S['header_start_row']}행 이후에서** "
                "'Temperature'와 'Heat Flow' 항목 행을 자동으로 찾아 PTFE 바인더의 용융 거동(Onset/Peak 온도)과 "
                "결정화도(X<sub>c</sub>)를 분석합니다. TRIOS `.txt` / `.csv` 파일도 사용할 수 있습니다.", unsafe_allow_html=True)
    if '_flash' in st.session_state:
        kind, msg = st.session_state.pop('_flash')
        getattr(st, kind)(msg)

    tab_an, tab_dash, tab_sop, tab_help = st.tabs(['📊 데이터 분석', '📈 대시보드', '⚙️ SOP 설정', '📖 분석 원리 / 사용법'])

    with tab_an:
        render_upload(S)
        sub1, sub2, sub3 = st.tabs(['🔬 샘플별 분석', '📦 로트별 분석', '🗂️ 전체 데이터 관리'])
        with sub1:
            render_sample_report(S, samples, res, cur_row)
        with sub2:
            render_lot_view(S, samples, res)
        with sub3:
            render_data_manager(S, samples, res)
    with tab_dash:
        render_dashboard(S, samples, res)
    with tab_sop:
        render_sop(S, samples)
    with tab_help:
        render_help(S)


def render_upload(S):
    st.subheader('📤 Raw 데이터 업로드')
    nonce = st.session_state.get('up_nonce', 0)
    files = st.file_uploader('DSC Raw 파일 (.xlsx, .xls, .txt, .csv) — 여러 파일을 한 번에 올릴 수 있습니다. '
                             '파일명에 BLANK/BLANC가 있으면 Blank(빈 팬)로 등록됩니다.',
                             type=['xlsx', 'xls', 'txt', 'csv'], accept_multiple_files=True, key=f'up_{nonce}')
    if files:
        msgs = []
        with st.spinner('파일을 분석하고 저장하는 중...'):
            for f in files:
                try:
                    msgs.append(('success', f'**{f.name}** — ' + ingest_file(f.name, f.getvalue(), S)))
                except Exception as e:
                    msgs.append(('error', f'**{f.name}** — 엑셀 파일을 처리하는 도중 오류가 발생했습니다: {e}'))
        st.session_state['upload_msgs'] = msgs
        bump('up_nonce')
        st.rerun()
    msgs = st.session_state.get('upload_msgs')
    if msgs:
        ok = sum(1 for k, _ in msgs if k == 'success')
        with st.expander(f'✅ 업로드 결과: 성공 {ok}개 / 실패 {len(msgs) - ok}개 (자동 저장 완료) — 눌러서 상세 보기',
                         expanded=(ok < len(msgs)) or len(msgs) <= 2):
            for kind, m in msgs:
                getattr(st, kind)(m)
            if st.button('업로드 결과 닫기', key='close_msgs'):
                st.session_state.pop('upload_msgs')
                st.rerun()


def render_sample_report(S, samples, res, cur_row):
    if cur_row is None:
        st.info('업로드된 시료가 없습니다. 위에서 Raw 파일을 업로드하면 이곳에 분석 리포트가 나타납니다. '
                '사용 방법은 [📖 분석 원리 / 사용법] 탭을 참고하세요.')
        return
    sid = int(cur_row['id'])
    df = load_df(sid, int(cur_row['version']))
    binfo = get_blank_info(S, samples)
    r = analyze_sample(sid, int(cur_row['version']), sample_params(cur_row, S, binfo))
    info = parse_filename(cur_row['filename'], S)

    h1, h2, h3 = st.columns([6, 1, 1])
    with h1:
        st.markdown(f'#### {cur_row["filename"]}')
        chips = [f'로트 **{cur_row["lot"]}**', f'시료 {rep_label(int(cur_row["rep"]))}', f'공정 **{cur_row["proc"]}**']
        for label, key, unit in (('측정일', 'date', ''), ('투입량', 'feed', ' kg/hr'), ('RPM', 'rpm', ''),
                                 ('설정온도', 'set_temp', ' °C'), ('배치', 'batch', '')):
            if info[key] is not None:
                v = info[key]
                chips.append(f'{label} {("#" if key == "batch" else "")}{v:g}{unit}' if isinstance(v, (int, float)) else f'{label} {v}')
        st.caption(' · '.join(chips))
    h2.button('◀ 이전', on_click=cb_nav, args=(-1,), width='stretch', key='nav_prev')
    h3.button('다음 ▶', on_click=cb_nav, args=(1,), width='stretch', key='nav_next')

    hr = cur_row['header_row']
    if pd.notna(hr):
        hr = int(hr)
        st.success(f'파일 업로드 성공! ({hr + 1}행을 헤더로'
                   + (f', {int(cur_row["unit_row"]) + 1}행을 단위로 인식 완료)' if pd.notna(cur_row['unit_row']) else ' 인식, 단위 행 없음)'))

    # 0. 진단 정보
    if r.get('t_range'):
        st.info(f'진단 정보: 읽어온 온도 범위 {r["t_range"][0]:.1f} ~ {r["t_range"][1]:.1f} °C, 유효 데이터 {r["n_valid"]:,}행'
                + (f' (1st Heating {len(r["temp"]):,}점 사용)' if r.get('temp') is not None and S['heating_only'] else ''))

    # 컬럼 선택
    cols = list(df.columns)
    mass_v = float(cur_row['mass']) if pd.notna(cur_row['mass']) else 0.0
    rep_sig = (sid, int(cur_row['version']), cur_row['temp_col'], cur_row['hf_col'], cur_row['hf_unit'], mass_v)
    if st.session_state.get('_rep_sid') != rep_sig:
        st.session_state['rep_tcol'] = cur_row['temp_col'] if cur_row['temp_col'] in cols else cols[0]
        st.session_state['rep_hcol'] = cur_row['hf_col'] if cur_row['hf_col'] in cols else cols[min(1, len(cols) - 1)]
        st.session_state['rep_unit'] = cur_row['hf_unit'] or 'W/g'
        st.session_state['rep_mass'] = mass_v
        st.session_state['_rep_sid'] = rep_sig
    c1, c2, c3, c4 = st.columns([3, 3, 1.4, 1.4])
    c1.selectbox('온도(Temperature) 컬럼 선택', cols, key='rep_tcol', on_change=cb_cols)
    c2.selectbox('열 플로우(Heat Flow) 컬럼 선택', cols, key='rep_hcol', on_change=cb_cols)
    c3.selectbox('Heat Flow 단위', ['W/g', 'mW/g', 'mW', 'W'], key='rep_unit', on_change=cb_unit,
                 help='W/g(정규화)면 그대로 사용합니다. mW·W이면 시료 질량으로 나눠 W/g로 환산합니다.')
    c4.number_input('시료 질량 (mg)', min_value=0.0, step=0.01, key='rep_mass', on_change=cb_mass,
                    disabled=st.session_state['rep_unit'] not in ('mW', 'W'),
                    help='Heat Flow 단위가 mW 또는 W일 때만 사용')
    if r.get('blank_applied'):
        st.caption(f'🧪 Blank 차감 적용: {binfo["name"]}')

    if r.get('error'):
        st.error(r['error'])
        return
    for w in r.get('warnings', []):
        st.warning(w)

    st.divider()
    st.subheader('📊 자동 분석 결과 리포트')
    m1, m2, m3 = st.columns(3)
    m1.metric('용융 시작 온도 (T_onset)', f'{r["t_onset"]:.2f} °C', border=True)
    m2.metric('용융 피크 온도 (T_peak)', f'{r["t_peak"]:.2f} °C', border=True)
    m3.metric('산출된 결정화도 (X_c)', f'{r["xc"]:.2f} %', border=True)
    st.info(f'적분 적용 구간: **{r["t_start"]:.1f} ~ {r["t_end"]:.1f} °C** · 적용 표준 엔탈피(ΔH₀): **{S["dh0"]} J/g** · '
            f'산출 융해 엔탈피(ΔHm): **{r["dh"]:.4f} J/g** · PTFE {float(cur_row["ptfe"]):g} wt% · 승온 속도 {S["rate"]:g} °C/min')

    fig = make_report_fig(r, S, cur_row['proc'], cur_row['filename'])
    show_figure(fig, f'DSC_{cur_row["filename"].rsplit(".", 1)[0]}', key=f'rep_{sid}')


def render_lot_view(S, samples, res):
    if res.empty:
        st.info('업로드된 시료가 없습니다.')
        return
    summ = lot_summary(res)
    st.markdown('##### 로트별 결정화도 요약')
    st.dataframe(summ, hide_index=True, width='stretch',
                 column_config={'Xc 평균(%)': st.column_config.NumberColumn(format='%.2f'),
                                'SD': st.column_config.NumberColumn(format='%.2f'),
                                'CV(%)': st.column_config.NumberColumn(format='%.1f'),
                                'T_onset 평균(°C)': st.column_config.NumberColumn(format='%.1f'),
                                'T_peak 평균(°C)': st.column_config.NumberColumn(format='%.1f'),
                                'ΔHm 평균(J/g)': st.column_config.NumberColumn(format='%.4f')})
    st.divider()
    lots = list(summ['로트'])
    if st.session_state.get('lot_pick') not in lots:
        st.session_state['lot_pick'] = lots[0]
    lot = st.selectbox('상세 확인할 로트', lots, key='lot_pick')
    g = res[res['로트'] == lot].sort_values('반복')
    valid = g[g['포함'] & g['ok']]
    sx = stats(valid['Xc'])
    a, b, c, d = st.columns(4)
    a.metric('결정화도 평균 (X_c)', f'{sx["mean"]:.2f} %' if sx['n'] else '-', border=True)
    b.metric('표준편차 (SD)', f'{sx["sd"]:.2f} %' if sx['n'] > 1 else '-', border=True)
    c.metric('변동계수 (CV)', f'{sx["cv"]:.1f} %' if sx['n'] > 1 else '-', border=True,
             delta=('기준 초과' if sx['n'] > 1 and sx['cv'] > S['cv_warn'] else None), delta_color='inverse')
    d.metric('유효 시료 수 (N)', f'{sx["n"]} / {len(g)}', border=True)

    # 로트 일괄 설정
    with st.container(border=True):
        st.markdown('**로트 일괄 설정** — 이 로트의 모든 시료에 적용됩니다.')
        k = re.sub(r'\W', '_', lot)
        c1, c2, c3, c4 = st.columns([1.2, 1, 3, 1])
        cur_proc = g['공정'].iloc[0] if g['공정'].iloc[0] in PROCESSES else 'Mixing'
        new_proc = c1.selectbox('공정', PROCESSES, index=PROCESSES.index(cur_proc), key=f'lp_{k}')
        new_ptfe = c2.number_input('PTFE 함량 (wt%)', value=float(g['PTFE'].iloc[0]), step=0.1, min_value=0.0, key=f'lw_{k}')
        new_memo = c3.text_input('메모', value=S['lot_memo'].get(lot, ''), key=f'lm_{k}')
        c4.write('')
        if c4.button('적용', key=f'la_{k}', type='primary', width='stretch'):
            if new_proc != cur_proc:
                d_ = S['proc'][new_proc]
                db_exec('UPDATE samples SET proc=?, t_start=?, t_end=? WHERE lot=? AND is_blank=0',
                        (new_proc, d_['t_start'], d_['t_end'], lot))
            db_exec('UPDATE samples SET ptfe=? WHERE lot=? AND is_blank=0', (float(new_ptfe), lot))
            S['lot_memo'][lot] = new_memo
            save_settings(S)
            st.session_state['_sb_sig'] = None
            st.session_state['_flash'] = ('success', f'로트 [{lot}] 설정을 저장했습니다.')
            st.rerun()

    # 로트 내 시료 표 (편집 → 자동 저장)
    st.markdown('**시료별 결과** — 포함 체크, PTFE, 적분 구간은 표에서 바로 수정할 수 있습니다 (자동 저장).')
    show = g[['포함', '시료', '파일명', 'PTFE', '적분시작', '적분종료', 'T_onset', 'T_peak', 'dH', 'Xc', '상태', '메시지']].copy()
    nonce = st.session_state.get('ed_lot', 0)
    edited = st.data_editor(
        show, hide_index=True, width='stretch', key=f'edlot_{k}_{nonce}',
        disabled=['시료', '파일명', 'T_onset', 'T_peak', 'dH', 'Xc', '상태', '메시지'],
        column_config={
            '포함': st.column_config.CheckboxColumn('포함', help='해제하면 로트 평균·대시보드에서 제외'),
            'PTFE': st.column_config.NumberColumn('PTFE (wt%)', format='%.2f', step=0.1, min_value=0.0),
            '적분시작': st.column_config.NumberColumn('적분 시작 (°C)', format='%.1f', step=1.0),
            '적분종료': st.column_config.NumberColumn('적분 종료 (°C)', format='%.1f', step=1.0),
            'T_onset': st.column_config.NumberColumn('T_onset (°C)', format='%.2f'),
            'T_peak': st.column_config.NumberColumn('T_peak (°C)', format='%.2f'),
            'dH': st.column_config.NumberColumn('ΔHm (J/g)', format='%.4f'),
            'Xc': st.column_config.NumberColumn('X_c (%)', format='%.2f'),
        })
    if apply_editor(show, edited, {'포함': 'include', 'PTFE': 'ptfe', '적분시작': 't_start', '적분종료': 't_end'}, S):
        bump('ed_lot')
        st.session_state['_sb_sig'] = None
        st.rerun()

    # 로트 내 곡선 겹쳐보기
    binfo = get_blank_info(S, samples)
    items = []
    for i, (sid, row) in enumerate(g.iterrows()):
        srow = samples.loc[sid]
        r = analyze_sample(int(sid), int(srow['version']), sample_params(srow, S, binfo))
        if r.get('ok'):
            items.append((f'{row["시료"]}{" (제외)" if not row["포함"] else ""}', SERIES_COLORS[i % 8], r, row['공정']))
    if items:
        st.markdown('**로트 내 시료 곡선 겹쳐보기** — 재현성 확인용 (베이스라인 차감, 흡열이 위쪽)')
        show_figure(overlay_fig(items, S, f'DSC Overlay — {trunc_mid(lot, 50)}'), f'Overlay_{lot}', key=f'ov_{k}')


def render_data_manager(S, samples, res):
    if samples.empty:
        st.info('업로드된 데이터가 없습니다.')
        return
    st.markdown('##### 전체 시료 목록')
    st.caption('표에서 **공정(드롭다운)·PTFE·적분 구간·포함 여부**를 바로 수정할 수 있으며 즉시 자동 저장됩니다. '
               '필요 없는 데이터는 **삭제** 칸을 체크한 뒤 아래 삭제 버튼을 누르세요.')
    q = st.text_input('🔍 검색 (파일명·로트, 띄어쓰기로 여러 단어 AND 검색)', key='dm_q')
    data = res.copy() if not res.empty else pd.DataFrame()
    if not data.empty:
        if q.strip():
            terms = q.lower().split()
            data = data[data['파일명'].str.lower().apply(lambda s: all(t in s for t in terms))]
        data = data.sort_values(['업로드', '파일명'], ascending=[False, True])
        show = data[['포함', '파일명', '로트', '시료', '공정', 'PTFE', '적분시작', '적분종료', 'Xc', 'T_onset', 'T_peak', 'dH', '상태', '업로드']].copy()
        show.insert(0, '삭제', False)
        nonce = st.session_state.get('ed_all', 0)
        edited = st.data_editor(
            show, hide_index=True, width='stretch', key=f'edall_{nonce}', height=min(600, 38 + 35 * len(show)),
            disabled=['파일명', '로트', '시료', 'Xc', 'T_onset', 'T_peak', 'dH', '상태', '업로드'],
            column_config={
                '삭제': st.column_config.CheckboxColumn('삭제', help='체크 후 아래 [선택 삭제] 버튼'),
                '포함': st.column_config.CheckboxColumn('포함'),
                '공정': st.column_config.SelectboxColumn('공정', options=PROCESSES, required=True),
                'PTFE': st.column_config.NumberColumn('PTFE (wt%)', format='%.2f', step=0.1, min_value=0.0),
                '적분시작': st.column_config.NumberColumn('적분 시작', format='%.1f', step=1.0),
                '적분종료': st.column_config.NumberColumn('적분 종료', format='%.1f', step=1.0),
                'Xc': st.column_config.NumberColumn('X_c (%)', format='%.2f'),
                'T_onset': st.column_config.NumberColumn('T_onset', format='%.2f'),
                'T_peak': st.column_config.NumberColumn('T_peak', format='%.2f'),
                'dH': st.column_config.NumberColumn('ΔHm (J/g)', format='%.4f'),
            })
        if apply_editor(show, edited, {'포함': 'include', '공정': 'proc', 'PTFE': 'ptfe', '적분시작': 't_start', '적분종료': 't_end'}, S):
            bump('ed_all')
            st.session_state['_sb_sig'] = None
            st.rerun()
        to_del = [int(i) for i in edited.index[edited['삭제']]]
        c1, c2, c3 = st.columns([1.3, 1.3, 4])
        if c1.button(f'🗑️ 선택 삭제 ({len(to_del)}개)', disabled=not to_del, width='stretch'):
            st.session_state['_confirm_del'] = to_del
        csv = res.drop(columns=['id', 'ok']).to_csv(index=False).encode('utf-8-sig')
        c2.download_button('📥 전체 결과 CSV', csv, file_name=f'DSC_PTFE_results_{datetime.now():%Y%m%d}.csv',
                           mime='text/csv', on_click='ignore', width='stretch')
        pend = st.session_state.get('_confirm_del')
        if pend:
            st.warning(f'{len(pend)}개 시료를 삭제합니다. 계속할까요? (삭제 직후 되돌리기 가능)')
            k1, k2, _ = st.columns([1, 1, 5])
            if k1.button('삭제 확인', type='primary', width='stretch'):
                st.session_state['_undo'] = db_fetch_full(pend)
                db_delete(pend)
                st.session_state.pop('_confirm_del')
                bump('ed_all')
                st.rerun()
            if k2.button('취소', width='stretch'):
                st.session_state.pop('_confirm_del')
                st.rerun()
    undo = st.session_state.get('_undo')
    if undo:
        u1, u2 = st.columns([4, 1])
        u1.info(f'{len(undo)}개 데이터를 삭제했습니다.')
        if u2.button('↩️ 되돌리기', width='stretch'):
            db_restore(undo)
            st.session_state.pop('_undo')
            bump('ed_all')
            st.rerun()

    # Blank
    blanks = samples[samples['is_blank'] == 1]
    st.markdown('##### 🧪 Blank (빈 팬) 데이터')
    if blanks.empty:
        st.caption('등록된 Blank가 없습니다. 파일명에 BLANK/BLANC가 들어간 빈 팬 측정 파일을 올리면 자동 등록됩니다. '
                   '(Heat Flow 단위가 mW/W인 시료에만 차감됩니다.)')
    else:
        for bid, b in blanks.iterrows():
            c1, c2 = st.columns([5, 1])
            act = (S.get('active_blank') == int(bid)) or (S.get('active_blank') not in blanks.index and bid == blanks.sort_values('uploaded_at').index[-1])
            c1.write(f'{"✅ " if act and S["use_blank"] else ""}{b["filename"]} · 단위 {b["hf_unit"]} · 업로드 {b["uploaded_at"]}')
            if c2.button('삭제', key=f'bdel_{bid}', width='stretch'):
                st.session_state['_undo'] = db_fetch_full([int(bid)])
                db_delete([int(bid)])
                st.rerun()

    with st.expander('💽 데이터 백업'):
        st.write(f'모든 데이터는 `{DB_PATH}` 파일에 저장됩니다. 이 파일을 복사해 두면 백업이 되고, '
                 '다른 PC의 app.py 옆에 같은 이름으로 넣으면 그대로 이어서 사용할 수 있습니다.')
        with open(DB_PATH, 'rb') as fh:
            st.download_button('📦 데이터베이스 백업 파일 다운로드', fh.read(), file_name=f'dsc_ptfe_data_{datetime.now():%Y%m%d}.db',
                               mime='application/octet-stream', on_click='ignore')


def render_dashboard(S, samples, res):
    if res.empty:
        st.info('업로드된 시료가 없습니다. [📊 데이터 분석] 탭에서 Raw 파일을 업로드하세요.')
        return
    st.subheader('📈 로트·샘플 결정화도 비교')
    # 필터
    f1, f2, f3, f4 = st.columns([2.2, 1.5, 1.6, 1.6])
    q = f1.text_input('🔍 검색 (로트명·메모, 띄어쓰기 = AND)', key='d_q', placeholder='예: K4 18C')
    procs = f2.multiselect('공정', PROCESSES, default=PROCESSES, key='d_proc')
    dates = sorted(d for d in res['측정일'].unique() if d)
    use_date = f3.checkbox('측정일 필터 사용', key='d_use_date', disabled=not dates)
    drange = None
    if use_date and dates:
        drange = f3.date_input('측정일 범위', value=(pd.Timestamp(dates[0]).date(), pd.Timestamp(dates[-1]).date()), key='d_dates')
    view = f4.radio('비교 단위', ['로트별 (평균 ± 오차)', '샘플별 (개별 값)'], key='d_view')
    g1, g2, g3, _ = st.columns([1.6, 1.6, 2.4, 1.6])
    group_by = g1.selectbox('묶음 기준', ['로트 (파일명 전체)', '조건 (배치 #번호 통합)'], key='d_group')
    err_mode = g2.selectbox('오차막대', ['표준편차 (SD)', '표준오차 (SE)', '최소~최대', '표시 안 함'], key='d_err')

    lots = lot_summary(res)
    lots['메모'] = lots['로트'].map(lambda x: S['lot_memo'].get(x, ''))
    if q.strip():
        terms = q.lower().split()
        lots = lots[(lots['로트'] + ' ' + lots['메모']).str.lower().apply(lambda s: all(t in s for t in terms))]
    lots = lots[lots['공정'].isin(procs)]
    if drange and len(drange) == 2:
        d0, d1 = str(drange[0]), str(drange[1])
        lots = lots[lots['측정일'].apply(lambda d: (not d) or (d0 <= d <= d1))]

    left, right = st.columns([1.5, 3])
    with left:
        st.markdown('**비교할 로트 선택**')
        b1, b2 = st.columns(2)
        if b1.button('전체 선택', width='stretch', key='d_all', help='검색·필터 결과에 보이는 로트를 모두 선택'):
            for lt in lots['로트']:
                S['dash_lot_sel'][lt] = True
            save_settings(S)
            bump('ed_dash')
            st.rerun()
        if b2.button('전체 해제', width='stretch', key='d_none'):
            S['dash_lot_sel'] = {}
            save_settings(S)
            bump('ed_dash')
            st.rerun()
        sel_df = pd.DataFrame({'선택': [bool(S['dash_lot_sel'].get(lt, False)) for lt in lots['로트']],
                               '로트': lots['로트'].values, 'Xc(%)': lots['Xc 평균(%)'].values,
                               'N': lots['N(유효/전체)'].values})
        nonce = st.session_state.get('ed_dash', 0)
        edited = st.data_editor(sel_df, hide_index=True, width='stretch', key=f'dsel_{nonce}', disabled=['로트', 'Xc(%)', 'N'],
                                height=min(520, 38 + 35 * max(1, len(sel_df))),
                                column_config={'선택': st.column_config.CheckboxColumn('선택', width='small'),
                                               '로트': st.column_config.TextColumn(width='medium'),
                                               'Xc(%)': st.column_config.NumberColumn(format='%.1f', width='small'),
                                               'N': st.column_config.TextColumn(width='small')})
        if not edited['선택'].equals(sel_df['선택']):
            for lt, v in zip(edited['로트'], edited['선택']):
                S['dash_lot_sel'][lt] = bool(v)
            save_settings(S)
            bump('ed_dash')
            st.rerun()
        chosen = [lt for lt, v in zip(edited['로트'], edited['선택']) if v]
        off = set(S.get('dash_sample_off', []))
        cand = res[res['로트'].isin(chosen) & res['포함'] & res['ok']].sort_values(['로트', '반복'])
        with st.expander(f'샘플 단위 선택 ({len(cand) - len(off & set(cand.index))}/{len(cand)})'):
            st.caption('비교에서 뺄 시료의 체크를 해제하세요. (데이터 분석 탭에서 제외했거나 오류인 시료는 목록에 없습니다)')
            sdf = pd.DataFrame({'선택': [int(i) not in off for i in cand.index], '로트': cand['로트'].values,
                                '시료': cand['시료'].values, 'Xc(%)': cand['Xc'].values}, index=cand.index)
            snonce = st.session_state.get('ed_dsamp', 0)
            sed = st.data_editor(sdf, hide_index=True, width='stretch', key=f'dsamp_{snonce}', disabled=['로트', '시료', 'Xc(%)'],
                                 column_config={'Xc(%)': st.column_config.NumberColumn(format='%.2f')})
            if not sed['선택'].equals(sdf['선택']):
                for sid, v in zip(sed.index, sed['선택']):
                    if v:
                        off.discard(int(sid))
                    else:
                        off.add(int(sid))
                S['dash_sample_off'] = sorted(off)
                save_settings(S)
                bump('ed_dsamp')
                st.rerun()

    use = res[res['로트'].isin(chosen) & res['포함'] & res['ok'] & ~res.index.isin(list(off))]
    with right:
        if use.empty:
            st.info('왼쪽 목록에서 비교할 로트를 체크하세요.')
            return
        key_col = '조건' if group_by.startswith('조건') else '로트'
        groups = []
        for key, g in use.groupby(key_col, sort=True):
            g = g.sort_values(['로트', '반복'])
            items = [((f'#{int(b)} ' if key_col == '조건' and pd.notna(b) and g['로트'].nunique() > 1 else '') + s, x)
                     for s, x, b in zip(g['시료'], g['Xc'], g['배치'])]
            groups.append(dict(key=key, proc=g['공정'].iloc[0], items=items, stats=stats(g['Xc']), df=g,
                               x={'투입량': g['투입량'].iloc[0], 'RPM': g['RPM'].iloc[0], '설정온도': g['설정온도'].iloc[0],
                                  '측정일': g['측정일'].min(), '공정변수': g['공정변수'].iloc[0], '배치': g['배치'].iloc[0],
                                  'PTFE': g['PTFE'].mean()}))
        sort_by = g3.selectbox('정렬', ['이름순', '측정일순', 'Xc 높은순', 'Xc 낮은순'], key='d_sort')
        if sort_by == '측정일순':
            groups.sort(key=lambda g: (g['x']['측정일'] or '9', g['key']))
        elif sort_by == 'Xc 높은순':
            groups.sort(key=lambda g: -g['stats']['mean'])
        elif sort_by == 'Xc 낮은순':
            groups.sort(key=lambda g: g['stats']['mean'])
        ref_opts = ['없음'] + [g['key'] for g in groups]
        if st.session_state.get('d_ref') not in ref_opts:
            st.session_state['d_ref'] = '없음'
        ref_key = st.selectbox('기준 로트 (Δ 비교)', ref_opts, key='d_ref')
        ref = next(((g['key'], g['stats']['mean']) for g in groups if g['key'] == ref_key), None)

        show_figure(bar_fig(groups, view, err_mode, ref), 'Xc_compare', key='dash_bar')
        st.caption('막대 = 평균, 점 = 개별 시료, 오차막대 = ' + err_mode + ' (N ≥ 2일 때). 파랑 = Mixing, 주황 = Kneading.')

        # 결과 표
        st.markdown('##### 비교 결과 표')
        if view.startswith('로트별'):
            rows = []
            for g in groups:
                s = g['stats']
                rows.append({key_col: g['key'], '공정': g['proc'], '측정일': g['x']['측정일'], '투입량(kg/hr)': g['x']['투입량'],
                             'RPM': g['x']['RPM'], '설정온도(°C)': g['x']['설정온도'], 'PTFE(wt%)': g['x']['PTFE'], 'N': s['n'],
                             'Xc 평균(%)': s['mean'], 'SD': s['sd'], 'SE': s['se'], 'CV(%)': s['cv'], 'Min': s['min'], 'Max': s['max'],
                             '기준 대비 Δ(%p)': (s['mean'] - ref[1]) if ref else np.nan,
                             'T_onset 평균(°C)': g['df']['T_onset'].mean(), 'T_peak 평균(°C)': g['df']['T_peak'].mean(),
                             'ΔHm 평균(J/g)': g['df']['dH'].mean()})
            table = pd.DataFrame(rows)
        else:
            parts = []
            for g in groups:
                d = g['df'][['로트', '시료', '파일명', '공정', '측정일', '투입량', 'RPM', '설정온도', 'PTFE', '적분시작', '적분종료',
                             'T_onset', 'T_peak', 'dH', 'Xc']].copy()
                d['기준 대비 Δ(%p)'] = (d['Xc'] - ref[1]) if ref else np.nan
                parts.append(d)
            table = pd.concat(parts).rename(columns={'dH': 'ΔHm(J/g)', 'Xc': 'Xc(%)', '투입량': '투입량(kg/hr)',
                                                    '설정온도': '설정온도(°C)', 'PTFE': 'PTFE(wt%)'})
        num_fmt = {c: st.column_config.NumberColumn(format='%.2f') for c in table.columns
                   if table[c].dtype.kind == 'f' and not c.startswith('ΔHm')}
        num_fmt.update({c: st.column_config.NumberColumn(format='%.4f') for c in table.columns if c.startswith('ΔHm')})
        st.dataframe(table, hide_index=True, width='stretch', column_config=num_fmt)
        st.download_button('📥 비교 표 CSV 저장', table.to_csv(index=False).encode('utf-8-sig'),
                           file_name=f'DSC_PTFE_compare_{datetime.now():%Y%m%d}.csv', mime='text/csv', on_click='ignore')
        st.caption('표 안의 셀을 드래그로 선택해 Ctrl+C 하면 엑셀에 바로 붙여넣을 수 있습니다.')

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            xname = st.selectbox('공정 조건 vs 결정화도 — X축', list(XPARAMS), key='d_x')
            fig, npts = scatter_fig(groups, xname, err_mode)
            if npts:
                show_figure(fig, f'Xc_vs_{XPARAMS[xname]}', key='dash_sc')
            else:
                plt.close(fig)
                st.info(f"선택한 항목에 '{xname}' 정보가 없습니다 (파일명에서 읽습니다).")
        with c2:
            st.markdown('**DSC 곡선 겹쳐보기** (베이스라인 차감, 흡열 ↑)')
            binfo = get_blank_info(S, samples)
            items, legend_done = [], set()
            for gi, g in enumerate(groups):
                color = SERIES_COLORS[gi] if gi < len(SERIES_COLORS) else OTHER_COLOR
                for sid in g['df'].index[:12]:
                    srow = samples.loc[sid]
                    r = analyze_sample(int(sid), int(srow['version']), sample_params(srow, S, binfo))
                    if r.get('ok'):
                        label = trunc_mid(g['key'], 30) if gi < len(SERIES_COLORS) else '기타'
                        items.append((label, color, r, g['proc']))
            if items:
                show_figure(overlay_fig(items, S, 'DSC Overlay (selected)'), 'DSC_overlay', key='dash_ov')
                if len(groups) > len(SERIES_COLORS):
                    st.caption(f'색 구분은 처음 {len(SERIES_COLORS)}개 그룹까지이며 나머지는 회색입니다.')


def render_sop(S, samples):
    st.subheader('⚙️ 분석 SOP 설정')
    st.caption('여기서 저장한 값은 **새로 업로드하는 시료의 기본값**이 됩니다. 기존 시료에도 적용하려면 아래 체크박스를 선택하세요.')
    with st.form('sop_form'):
        st.markdown('##### 공정별 기준')
        hdr = st.columns([2, 1, 1])
        hdr[1].markdown('**Mixing**')
        hdr[2].markdown('**Kneading**')
        fields = [('t_start', '적분 시작 온도 (°C)', 1.0), ('t_end', '적분 종료 온도 (°C)', 1.0),
                  ('view_min', '그래프·미분 표시 구간 시작 (°C)', 5.0), ('view_max', '그래프·미분 표시 구간 종료 (°C)', 5.0),
                  ('ptfe', '기본 PTFE 함량 (wt%)', 0.1)]
        vals = {p: {} for p in PROCESSES}
        for f, label, step in fields:
            c = st.columns([2, 1, 1])
            c[0].markdown(label)
            for j, p in enumerate(PROCESSES):
                vals[p][f] = c[j + 1].number_input(label, value=float(S['proc'][p][f]), step=step, key=f'sop_{p}_{f}',
                                                   label_visibility='collapsed')
        st.markdown('##### 계산 상수')
        a, b, c = st.columns(3)
        dh0 = a.number_input('100% 결정질 PTFE 표준 엔탈피 ΔH₀ (J/g)', value=float(S['dh0']), step=0.1, min_value=0.1,
                             help='사이드바 값과 같은 값입니다.')
        rate = b.number_input('승온 속도 (°C/min)', value=float(S['rate']), step=1.0, min_value=0.1)
        endo = c.selectbox('흡열 피크 방향', ['아래 (흡열 = 최솟값)', '위 (흡열 = 최댓값)'],
                           index=0 if S['endo'].startswith('아래') else 1, help='TRIOS 기본(Exo Up)은 흡열이 아래쪽')
        a, b, c = st.columns(3)
        sgw = a.number_input('Savitzky-Golay window_length (홀수)', value=int(S['sg_window']), step=2, min_value=3)
        sgp = b.number_input('Savitzky-Golay polyorder', value=int(S['sg_poly']), step=1, min_value=1, max_value=5)
        onw = c.number_input('Onset 베이스라인 평균 폭 (°C)', value=float(S['onset_width']), step=1.0, min_value=0.5,
                             help='(적분 시작 온도 − 폭) ~ 적분 시작 온도 구간의 평균을 베이스라인 y값으로 사용')
        a, b, c = st.columns(3)
        pmin = a.number_input('피크 탐색 구간 시작 (°C)', value=float(S['peak_min']), step=5.0)
        pmax = b.number_input('피크 탐색 구간 종료 (°C)', value=float(S['peak_max']), step=5.0)
        hs = c.number_input('헤더 탐색 시작 행 (엑셀 기준)', value=int(S['header_start_row']), step=1, min_value=1)
        a, b, c = st.columns(3)
        dym = a.selectbox('미분 그래프 y축 범위', ['자동', '고정'], index=0 if S['deriv_ylim_mode'] == '자동' else 1)
        dyl = b.number_input('고정 시 y축 ± 값', value=float(S['deriv_ylim']), step=0.1, min_value=0.0001, format='%.4f')
        heat = c.checkbox('승온(1st Heating) 구간만 사용', value=bool(S['heating_only']),
                          help='최고 온도 이후의 냉각·재가열 데이터를 제외합니다.')
        st.markdown('##### Blank · 판정 · 파일명 해석')
        a, b, c = st.columns(3)
        use_blank = a.checkbox('Blank(빈 팬) 차감 사용 (mW 단위 데이터에만 적용)', value=bool(S['use_blank']))
        blanks = samples[samples['is_blank'] == 1] if not samples.empty else pd.DataFrame()
        bopts = [None] + [int(i) for i in blanks.index] if not blanks.empty else [None]
        cur_b = S.get('active_blank') if S.get('active_blank') in bopts else None
        act_b = b.selectbox('적용할 Blank', bopts, index=bopts.index(cur_b),
                            format_func=lambda i: '가장 최근 Blank 자동' if i is None else blanks.at[i, 'filename'])
        cvw = c.number_input('로트 CV 경고 기준 (%)', value=float(S['cv_warn']), step=0.5)
        a, b, c = st.columns(3)
        bm = a.number_input('레시피 번호 배수 (10 → 1000 kg)', value=int(S['name_batch_mul']), step=10)
        fm = b.number_input('투입량 배수 (K4 → 400 kg/hr)', value=int(S['name_feed_mul']), step=10)
        tm = c.number_input('설정온도 배수 (18C → 180 °C)', value=int(S['name_temp_mul']), step=1)
        st.markdown('##### 기존 데이터에 적용')
        a, b = st.columns(2)
        reset_range = a.checkbox('기존 시료의 적분 구간을 공정별 새 SOP 값으로 일괄 변경')
        reset_ptfe = b.checkbox('기존 시료의 PTFE 함량을 공정별 새 기본값으로 일괄 변경')
        submitted = st.form_submit_button('💾 설정 저장 및 전체 재계산', type='primary')
    if submitted:
        errs = [f'{p}: 적분 종료 온도가 시작 온도보다 커야 합니다.' for p in PROCESSES if vals[p]['t_end'] <= vals[p]['t_start']]
        errs += [f'{p}: 표시 구간 종료가 시작보다 커야 합니다.' for p in PROCESSES if vals[p]['view_max'] <= vals[p]['view_min']]
        if pmax <= pmin:
            errs.append('피크 탐색 구간 종료가 시작보다 커야 합니다.')
        if errs:
            for e in errs:
                st.error(e)
        else:
            for p in PROCESSES:
                S['proc'][p].update(vals[p])
            S.update(dh0=dh0, rate=rate, endo=endo, sg_window=int(sgw) | 1, sg_poly=int(sgp), onset_width=onw,
                     peak_min=pmin, peak_max=pmax, header_start_row=int(hs), deriv_ylim_mode=dym, deriv_ylim=dyl,
                     heating_only=heat, use_blank=use_blank, active_blank=act_b, cv_warn=cvw,
                     name_batch_mul=int(bm), name_feed_mul=int(fm), name_temp_mul=int(tm))
            save_settings(S)
            for p in PROCESSES:
                if reset_range:
                    db_exec('UPDATE samples SET t_start=?, t_end=? WHERE proc=? AND is_blank=0',
                            (S['proc'][p]['t_start'], S['proc'][p]['t_end'], p))
                if reset_ptfe:
                    db_exec('UPDATE samples SET ptfe=? WHERE proc=? AND is_blank=0', (S['proc'][p]['ptfe'], p))
            st.session_state['_sb_sig'] = None
            st.session_state['_flash'] = ('success', 'SOP 설정을 저장하고 전체 시료를 다시 계산했습니다.')
            st.rerun()
    if st.button('↺ 기본값으로 초기화', help='SOP 설정만 초기화합니다 (데이터는 유지)'):
        keep = {k: S[k] for k in ('lot_memo', 'dash_lot_sel', 'dash_sample_off')}
        save_settings(dict(json.loads(json.dumps(DEFAULT_SETTINGS)), **keep))
        st.session_state['_flash'] = ('success', 'SOP 설정을 기본값으로 되돌렸습니다.')
        st.rerun()


def render_help(S):
    st.subheader('📖 분석 원리 / 사용법')
    st.markdown("""
#### 1. 빠른 시작 (처음 쓰는 분은 이 순서대로)
1. **TRIOS에서 데이터 내보내기** — 엑셀(.xlsx) 또는 텍스트(.txt)로 저장합니다. 파일 이름은 아래 *파일명 규칙*을 따르세요.
   같은 로트를 여러 번 측정했다면 `… (1).xlsx`, `… (2).xlsx` 처럼 끝에 번호를 붙이면 자동으로 한 로트로 묶입니다.
2. **[📊 데이터 분석] 탭에 파일 올리기** — 여러 파일을 한 번에 끌어다 놓아도 됩니다. 업로드와 동시에 **자동 저장**됩니다.
3. **왼쪽 사이드바에서 시료 선택** — [🔬 샘플별 분석]에 결과 리포트(T_onset, T_peak, X_c)와 그래프가 나타납니다.
4. **공정과 PTFE 함량 확인** — 사이드바의 *공정 선택*(Mixing/Kneading)과 *PTFE 함량*이 실제 배합과 맞는지 확인하세요.
5. **그래프에서 적분 구간 확인** — 주황색 영역이 용융 피크 전체를 덮고 있으면 정상입니다. 어긋나 있으면 사이드바의
   *적분 시작/종료 온도*를 조정하세요. 같은 로트 전체에 적용하려면 **[로트 전체 적용]** 을 누르세요.
6. **[📈 대시보드]에서 비교** — 비교할 로트를 체크하면 결정화도 막대그래프(오차막대 포함)와 결과 표가 나옵니다.
   그래프 아래 **📋 이미지 복사** 후 PPT에 Ctrl+V, 표는 **CSV 저장** 또는 셀 선택 후 Ctrl+C로 엑셀에 붙여넣습니다.
""")
    st.markdown("""
#### 2. 화면 구성
| 위치 | 내용 |
|---|---|
| **사이드바** | 분석할 시료 선택, 공정 선택, PTFE 함량, ΔH₀, 승온 속도, 적분 시작/종료 온도 — 바꾸는 즉시 자동 저장 |
| **📊 데이터 분석 → 🔬 샘플별 분석** | 선택한 시료 하나의 자동 분석 리포트와 그래프 (◀ 이전 / 다음 ▶ 로 넘기기) |
| **📊 데이터 분석 → 📦 로트별 분석** | 로트별 평균 ± SD 요약표, 로트 일괄 설정, 시료 포함/제외, 로트 내 곡선 겹쳐보기 |
| **📊 데이터 분석 → 🗂️ 전체 데이터 관리** | 전체 시료 표(공정 드롭다운·PTFE·적분 구간 바로 수정), 선택 삭제·되돌리기, 결과 CSV, Blank 관리, 백업 |
| **📈 대시보드** | 검색·공정·측정일 필터, 로트/샘플 체크박스 선택, 결정화도 비교 그래프 + 오차막대, 결과 표, 조건 vs X_c, 곡선 겹쳐보기 |
| **⚙️ SOP 설정** | 공정별 적분 구간·표시(미분) 구간·기본 PTFE 함량, 계산 상수, 스무딩, 파일명 해석 규칙 |
""")
    st.markdown("""
#### 3. 파일명 규칙
| 공정 | 형식 | 예시 |
|---|---|---|
| Mixing (믹싱) | `yymmdd APIM 10-1` [#배치] [기타] | `261117 APIM 10-1 #2 증속평가 (1).xlsx` |
| Kneading (니딩) | `yymmdd APIM 10-1 K4 60R 18C` [#배치] [기타] | `261117 APIM 10-1 K6 95R 18C #1 30Hz.xlsx` |

| 항목 | 의미 |
|---|---|
| `yymmdd` | 연월일 (261117 → 2026-11-17) |
| `APIM` | **A** = Anode(고정), **P** = Pilot(고정), **I** = 변경 코드(레시피·공정조건·소재 변경 시 알파벳이 바뀜), **M** = Mixing(고정) |
| `10-1` | 10 = 1000 kg(고정), `-` 뒤 숫자 = 같은 레시피 안에서 공정 변수 번호 |
| `K4` | 투입량 (4 → 400 kg/hr). 이 항목이 있으면 **Kneading**으로 자동 판별 |
| `60R` / `18C` | 니딩 RPM (60 rpm) / 설정온도 (18 → 180 °C) |
| `#1` | 배치 번호 (Mixing은 10-1 뒤, Kneading은 18C 뒤) |
| 기타 문자 | 증속평가, 50Hz 등 — 로트 이름의 일부로 남아 서로 다른 로트로 구분 |
| `(1)`, `(2)` | 맨 끝 괄호 번호 = **같은 로트의 반복 측정 시료** (괄호 없음 = 원본) |
""")
    test = st.text_input('🧪 파일명 해석 테스트 — 파일명을 입력하면 프로그램이 어떻게 읽는지 보여줍니다',
                         value='261117 APIM 10-1 K4 60R 18C #1 증속평가 (2).xlsx', key='help_name')
    i = parse_filename(test, S)
    st.dataframe(pd.DataFrame([
        ('로트명 (묶음 기준)', i['lot']), ('반복 측정', rep_label(i['rep'])), ('자동 판별 공정', i['proc_guess']),
        ('측정일', i['date'] or '인식 안 됨'), ('코드', f'{i["code"]} (변경 코드 {i["change"]})' if i['code'] else '인식 안 됨'),
        ('레시피', f'{i["recipe"]} → {i["batch_kg"]} kg, 공정 변수 {i["variant"]}' if i['recipe'] else '인식 안 됨'),
        ('투입량', f'{i["feed"]:g} kg/hr' if i['feed'] is not None else '-'), ('RPM', f'{i["rpm"]:g}' if i['rpm'] is not None else '-'),
        ('설정온도', f'{i["set_temp"]:g} °C' if i['set_temp'] is not None else '-'),
        ('배치 번호', f'#{i["batch"]}' if i['batch'] is not None else '-'), ('기타 문자', ', '.join(i['extras']) or '-'),
        ('Blank 여부', '예 (Blank로 등록)' if BLANK_RE.search(test) else '아니오'),
    ], columns=['항목', '해석 결과']), hide_index=True, width='stretch')

    st.markdown(f"""
#### 4. 분석 원리
DSC는 시료를 일정 속도로 가열하며 드나드는 열량을 측정합니다. PTFE 결정은 약 327 °C 부근에서 녹으며 열을 흡수(흡열 피크)하고,
이 피크 면적이 **결정이 녹는 데 필요한 에너지(융해 엔탈피 ΔHm)** 입니다. 결정이 많을수록 면적이 커집니다.

**0. 헤더 자동 탐지** — 엑셀 **{S['header_start_row']}행 이후**에서 온도(`temp`/`온도`)와 열흐름(`heat`/`flow`/`hf`/`열`) 키워드가 함께 있는
첫 행을 헤더로, 바로 다음 행을 **단위 행**으로 인식합니다. 완전히 빈 행은 제거하고, 온도·Heat Flow가 모두 숫자인 행만 사용합니다.
""")
    st.markdown('**1. 스무딩 & 미분** — Savitzky-Golay 필터(window 15, 3차)로 노이즈를 줄이고 1차 미분을 구합니다.')
    st.markdown('**2. 용융 피크 (T_peak)** — 피크 탐색 구간(기본 250~400 °C)에서 스무딩된 Heat Flow가 가장 낮은 점(흡열 피크)')
    st.markdown('**3. Onset 온도 (접선 외삽법)** — 적분 시작 온도 ~ T_peak 구간에서 기울기가 가장 큰 점(변곡점)의 접선이 '
                '베이스라인 높이(적분 시작 온도 − 5 °C ~ 적분 시작 온도의 평균)와 만나는 온도')
    st.latex(r'T_{onset} = T_{infl} + \frac{y_{baseline} - h_{infl}}{slope}')
    st.markdown('**4. 융해 엔탈피** — 적분 구간 양 끝점을 잇는 직선 베이스라인을 빼고 그 차이를 적분한 뒤 승온 속도로 나눕니다.')
    st.latex(r'\Delta H_m\ (\mathrm{J/g}) = \frac{60}{\beta}\int_{T_1}^{T_2} \left| HF(T) - BL(T) \right| dT')
    st.caption('HF 단위 W/g, β = 승온 속도(°C/min). ∫ … dT ÷ β 의 단위는 W·min/g 이므로 ×60 으로 J/g 로 환산합니다.')
    st.markdown('**5. 결정화도** — 측정한 전극 1 g 안에 PTFE는 w(=wt%/100) 만큼만 들어 있으므로 함량으로 보정합니다.')
    st.latex(r'X_c\ (\%) = \frac{\Delta H_m}{w_{PTFE} \times \Delta H_0} \times 100')
    st.caption('예) ΔHm = 0.47 J/g, PTFE 1 wt%, ΔH₀ = 85.4 J/g → 0.47 / (0.01 × 85.4) × 100 = 55.0 %')
    st.markdown("""
**Heat Flow 단위** — 엑셀이 W/g(정규화)이면 그대로 사용합니다. mW로 내보낸 파일은 시료 질량(mg)으로 나눠 W/g로 바꾸며,
이때 Blank(빈 팬) 파일이 있으면 먼저 빼 줍니다.

#### 5. 상태 표시 의미
| 상태 | 의미 |
|---|---|
| **정상** | 문제 없이 계산됨 |
| **확인 필요** | 결과가 틀렸을 수 있음 — 피크가 적분 구간 밖/경계에 붙음, 결정화도 100% 초과 등. 그래프를 열어 구간을 확인하세요. |
| **오류** | 계산 불가 — 유효 데이터 15개 미만, 적분 구간 데이터 부족, mW 단위인데 질량 없음 등. 오류 시료는 평균에서 자동 제외됩니다. |

#### 6. 자주 묻는 질문
- **결정화도가 100%를 넘거나 너무 낮아요.** → PTFE 함량(wt%), ΔH₀, Heat Flow 단위(W/g인지 mW인지)를 확인하세요.
- **“'Temperature'와 'Heat Flow' 항목 행을 찾지 못했습니다” 오류** → 내보낼 때 Temperature와 Heat Flow 신호가 포함되게 하세요.
- **같은 파일을 다시 올리면?** → 같은 파일명은 새 데이터로 교체되고, 공정·PTFE·적분 구간 등 설정은 유지됩니다.
- **잘못 지웠어요.** → 삭제 직후 나타나는 **↩️ 되돌리기** 버튼으로 복구할 수 있습니다.
- **데이터는 어디에 저장되나요?** → `app.py` 옆의 `dsc_ptfe_data.db` 파일입니다. 이 파일을 복사해 두면 백업이 됩니다.
- **그래프 이미지 복사가 안 돼요.** → 브라우저가 막는 경우입니다. 그래프를 우클릭 → *이미지 복사* 또는 **PNG 저장**을 이용하세요.
""")


if __name__ == '__main__':
    main()
