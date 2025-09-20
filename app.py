# -*- coding: utf-8 -*-
import io, re, hashlib
from io import StringIO
from collections import Counter
from datetime import date, timedelta, time, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from zoneinfo import ZoneInfo
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype

# =========================================================
# 基本設定
# =========================================================
st.set_page_config(page_title="デイトレ結果ダッシュボード", page_icon="📈", layout="wide")
st.title("📈 デイトレ結果ダッシュボード（VWAP/MA対応・3分足＋IN/OUT）")

TZ = ZoneInfo("Asia/Tokyo")
MAIN_CHART_HEIGHT = 560
LARGE_CHART_HEIGHT = 820

# 市場時間（前場/後場）
MORNING_START_SEC = 9*3600
MORNING_END_SEC   = 11*3600 + 30*60
AFTERNOON_START_SEC = 12*3600 + 30*60
AFTERNOON_END_SEC   = 15*3600 + 30*60

# 線色
COLOR_VWAP = "#808080"    # グレー
COLOR_MA1  = "#2ca02c"    # 緑
COLOR_MA2  = "#ff7f0e"    # オレンジ
COLOR_MA3  = "#1f77b4"    # 青

# =========================================================
# ユーティリティ
# =========================================================
def _clean_colname(name: str) -> str:
    if name is None: return ""
    s = str(name).replace("\ufeff","").replace("\u3000"," ")
    return re.sub(r"\s+"," ", s.strip())

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    return df.rename(columns={c:_clean_colname(c) for c in df.columns})

def to_numeric_jp(x):
    """日本語CSVの数値表記を数値化。 (123)->-123, 全角-, 桁区切り, 円/株/% 除去"""
    if isinstance(x, pd.Series):
        s = (x.astype(str)
               .str.replace(r"\((\s*[\d,\.]+)\)", r"-\1", regex=True)
               .str.replace("−", "-", regex=False)
               .str.replace(",", "", regex=False)
               .str.replace("円", "", regex=False)
               .str.replace("株", "", regex=False)
               .str.replace("%", "", regex=False)
               .str.strip())
        return pd.to_numeric(s, errors="coerce")
    if pd.isna(x): return np.nan
    if isinstance(x, str):
        x = re.sub(r"\((\s*[\d,\.]+)\)", r"-\1", x)
        x = x.replace("−","-").replace(",","").replace("円","").replace("株","").replace("%","").strip()
    return pd.to_numeric(x, errors="coerce")

@st.cache_data(show_spinner=False)
def read_table_from_upload(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    # Excel
    if file_name.lower().endswith(".xlsx"):
        try:
            return clean_columns(pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl"))
        except Exception:
            return pd.DataFrame()

    # 文字コード自動
    text = None
    for enc in ["utf-8-sig","utf-8","cp932","shift_jis","euc_jp"]:
        try:
            text = file_bytes.decode(enc); break
        except Exception: continue
    if text is None:
        try:
            text = file_bytes.decode("utf-8", errors="replace")
        except Exception:
            return pd.DataFrame()

    # 区切り推定
    sample = text[:20000]
    sep = None
    if sample.count("\t") >= 2 and sample.count("\t") > sample.count(","): sep = "\t"
    elif sample.count(";") >= 2 and sample.count(";") > sample.count(","): sep = ";"
    elif sample.count("|") >= 2 and sample.count("|") > sample.count(","): sep = "|"

    try:
        df = pd.read_csv(StringIO(text), sep=sep, engine="python")
        return clean_columns(df)
    except Exception:
        try:
            df = pd.read_csv(StringIO(text), engine="python")
            return clean_columns(df)
        except Exception:
            return pd.DataFrame()

def files_signature(files) -> str:
    if not files: return ""
    parts = []
    for f in files:
        b = f.getvalue()
        h = hashlib.md5(b).hexdigest()
        parts.append(f"{f.name}|{len(b)}|{h[:8]}")
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()

@st.cache_data(show_spinner=False)
def concat_uploaded_tables(files, sig: str, add_source_col: bool=True) -> pd.DataFrame:
    if not files: return pd.DataFrame()
    frames = []
    for f in files:
        df = read_table_from_upload(f.name, f.getvalue())
        if df.empty: continue
        if add_source_col:
            df = df.copy(); df["__source_file__"] = f.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

# ---- JSTのSeriesに強制変換する安全ヘルパー
def _to_jst_series(obj, index) -> pd.Series:
    """
    どんな入力でも、必ず tz-aware（JST）の pandas.Series[datetime64[ns, Asia/Tokyo]] を返す。
    """
    if isinstance(obj, pd.Series):
        s = pd.to_datetime(obj, errors="coerce", utc=False)
    else:
        s = pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")

    if is_datetime64tz_dtype(s):
        return s.dt.tz_convert(TZ)
    if is_datetime64_any_dtype(s):
        return s.dt.tz_localize(TZ)
    s = pd.to_datetime(s, errors="coerce")
    return s.dt.tz_localize(TZ)

# =========================================================
# 銘柄コード/名称 正規化
# =========================================================
def normalize_symbol_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    d = df.copy()

    def extract_code_from_series(s: pd.Series) -> pd.Series:
        if s is None:
            return pd.Series([pd.NA]*len(d), index=d.index, dtype="object")
        ss = s.astype(str).str.strip().str.upper().str.replace(".0","",regex=False)
        s1 = ss.str.extract(r'(?i)(\d{4,5}[A-Z])')[0]
        s2 = ss.str.extract(r'(\d{4,5})')[0]
        return s1.fillna(s2)

    code = pd.Series([pd.NA]*len(d), index=d.index, dtype="object")
    for col in ["コード4","銘柄コード","コード","コード番号","コード(数値)","コード "]:
        if col in d.columns: code = code.fillna(extract_code_from_series(d[col]))
    for col in ["銘柄名","名称","銘柄"]:
        if col in d.columns: code = code.fillna(extract_code_from_series(d[col]))
    if "__source_file__" in d.columns:
        sf = d["__source_file__"].astype(str)
        s1 = sf.str.extract(r'_(?i:(\d{4,5}[A-Z]))(?=[,_\.])')[0]
        s2 = sf.str.extract(r'_(\d{4,5})(?=[,_\.])')[0]
        code = code.fillna(s1.fillna(s2))

    d["code_key"] = pd.Series(code, dtype="string").str.upper().str.strip()
    d["code_key"] = d["code_key"].replace({"":"<NA>","NAN":"<NA>"}).replace("<NA>", pd.NA)

    name = None
    for col in ["銘柄名","名称","銘柄"]:
        if col in d.columns:
            s = d[col].astype(str).str.replace("\u3000"," ", regex=False).str.strip()
            name = s if name is None else name.fillna(s)
    d["name_key"] = name if name is not None else ""
    return d

def representative_name(df: pd.DataFrame) -> str:
    names = [x for x in df["name_key"].dropna().astype(str).tolist() if x and x.lower()!="nan"]
    if not names: return ""
    return Counter(names).most_common(1)[0][0]

def build_code_to_name_map(*dfs: pd.DataFrame) -> dict:
    mp = {}
    for df in dfs:
        if df is None or df.empty: continue
        d = normalize_symbol_cols(df)
        if "code_key" not in d.columns: continue
        grp = d[d["code_key"].notna() & (d["code_key"]!="")]
        if grp.empty: continue
        name_map = grp.groupby("code_key").apply(representative_name)
        for ck, nm in name_map.items():
            if isinstance(ck, str) and ck and nm:
                mp.setdefault(ck, nm)
    return mp

# =========================================================
# 実現損益・約定の正規化
# =========================================================
def pick_dt_col(df: pd.DataFrame, preferred=None) -> str | None:
    if df is None or df.empty: return None
    cands = preferred or ["約定日","約定日時","約定日付","日時","日付","年月日","決済日","受渡日"]
    for c in cands:
        if c in df.columns: return c
    for c in df.columns:
        if re.search(r"(約定|決済|受渡)?(日付|日時|年月日)", str(c)): return c
    return None

def pick_time_col(df: pd.DataFrame, preferred=None) -> str | None:
    if df is None or df.empty: return None
    cands = preferred or ["約定時刻","約定時間","時刻","時間","約定時刻(JST)","約定時間(日本)","時間(JST)"]
    for c in cands:
        if c in df.columns: return c
    for c in df.columns:
        if re.search(r"(約定)?(時刻|時間)", str(c)): return c
    return None

def parse_time_only_to_timedelta(s: pd.Series) -> pd.Series:
    ss = s.astype(str).str.strip().str.replace("：",":", regex=False)
    out = pd.Series(pd.NaT, index=ss.index, dtype="timedelta64[ns]")
    as_num = pd.to_numeric(ss, errors="coerce")
    mask_frac = as_num.notna() & (as_num>=0) & (as_num<=1)
    if mask_frac.any():
        secs = (as_num[mask_frac]*86400).round().astype(int)
        out.loc[mask_frac] = pd.to_timedelta(secs, unit="s")
    mask_hms = ss.str.match(r"^\d{1,2}:\d{1,2}(:\d{1,2})?$")
    out.loc[mask_hms] = pd.to_timedelta(ss.loc[mask_hms])
    mask_kanji = ss.str.match(r"^\d{1,2}時\d{1,2}分(\d{1,2})?秒?$")
    if mask_kanji.any():
        def jp_to_hms(x):
            m = re.match(r"^(\d{1,2})時(\d{1,2})分(?:(\d{1,2})秒)?$", x)
            hh,mm,ss_ = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
            return pd.to_timedelta(f"{hh:02d}:{mm:02d}:{ss_:02d}")
        out.loc[mask_kanji] = ss.loc[mask_kanji].map(jp_to_hms)
    mask_num = ss.str.match(r"^\d{3,6}$")
    if mask_num.any():
        def num_to_hms(x):
            x = x.zfill(6); hh,mm,ss_ = int(x[:2]), int(x[2:4]), int(x[4:6])
            return pd.to_timedelta(f"{hh:02d}:{mm:02d}:{ss_:02d}")
        out.loc[mask_num] = ss.loc[mask_num].map(num_to_hms)
    return out

def combine_date_time_cols(df: pd.DataFrame, date_col: str, time_col: str) -> pd.Series:
    d = pd.to_datetime(df[date_col], errors="coerce", infer_datetime_format=True)
    td = parse_time_only_to_timedelta(df[time_col]) if time_col in df.columns else pd.Series(pd.NaT, index=df.index)
    dt_str = df[date_col].astype(str)
    tail_num = dt_str.str.extract(r"(\d{3,6})\s*$")[0]
    mask_fill = td.isna() & tail_num.notna()
    if mask_fill.any():
        td.loc[mask_fill] = parse_time_only_to_timedelta(tail_num.loc[mask_fill])
    ts = d.dt.floor("D") + td
    ts = pd.to_datetime(ts, errors="coerce")
    try: ts = ts.dt.tz_localize(TZ)
    except Exception: ts = ts.dt.tz_convert(TZ)
    return ts

def parse_datetime_from_dtcol(df: pd.DataFrame, dtcol: str) -> pd.Series:
    s = df[dtcol].astype(str).str.strip().str.replace("：",":", regex=False)
    date_part = s.str.extract(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})")[0]
    d = pd.to_datetime(date_part, errors="coerce")
    t_hms  = s.str.extract(r"\b(\d{1,2}:\d{1,2}(?::\d{1,2})?)\b")[0]
    t_kan  = s.str.extract(r"\b(\d{1,2}時\d{1,2}分(?:\d{1,2})?秒?)\b")[0]
    t_tail = s.str.extract(r"\s(\d{3,6})\s*$")[0]
    t_str = t_hms.fillna(t_kan).fillna(t_tail).fillna("")
    td = parse_time_only_to_timedelta(t_str)
    ts = d.dt.floor("D") + td
    ts = pd.to_datetime(ts, errors="coerce")
    try: ts = ts.dt.tz_localize(TZ)
    except Exception: ts = ts.dt.tz_convert(TZ)
    return ts

def detect_pl_column(d: pd.DataFrame) -> str | None:
    strong = ["実現損益[円]","実現損益（円）","実現損益", "損益[円]","損益額","損益", "差引金額","損益合計"]
    for c in strong:
        if c in d.columns: return c
    candidates = []
    for c in d.columns:
        s = str(c).lower()
        if any(k in s for k in ["損益","pnl","profit","realized","pl","差引"]):
            candidates.append(c)
    if not candidates: return None
    best, best_ratio = None, 0.0
    for c in candidates:
        s = to_numeric_jp(d[c]); ratio = s.notna().mean()
        if ratio > best_ratio: best_ratio, best = ratio, c
    return best

def normalize_realized(df: pd.DataFrame) -> pd.DataFrame:
    """'約定日時','約定日','実現損益[円]' を生成。時刻が無い場合は 00:00（後で推定補完）。"""
    if df is None or df.empty: 
        return df
    d = clean_columns(df.copy())

    # 実現損益列
    pl_col = detect_pl_column(d)
    d["実現損益[円]"] = to_numeric_jp(d[pl_col]) if pl_col else pd.Series(dtype="float64")

    # 約定日時/約定日
    date_col = pick_dt_col(d)
    time_col = pick_time_col(d)

    ts = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    if date_col and time_col:
        ts = combine_date_time_cols(d, date_col, time_col)
    elif date_col:
        try:
            ts = pd.to_datetime(d[date_col], errors="coerce", infer_datetime_format=True)
        except Exception:
            for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
                ts = pd.to_datetime(d[date_col], format=fmt, errors="coerce")
                if ts.notna().any(): break
        try: ts = ts.dt.tz_localize(TZ)
        except Exception: ts = ts.dt.tz_convert(TZ)

    d["約定日時"] = ts
    d["約定日"]  = pd.to_datetime(ts, errors="coerce").dt.date

    # 追加情報
    d = normalize_symbol_cols(d)
    if "売却/決済単価[円]" in d.columns: d["__決済単価__"] = to_numeric_jp(d["売却/決済単価[円]"])
    if "数量[株]" in d.columns:         d["__数量__"]   = to_numeric_jp(d["数量[株]"])
    d["__action__"] = d.get("取引", pd.Series(index=d.index, dtype="object"))

    # 時刻ありフラグ（0:00:00は無し扱い）
    has_time = d["約定日時"].notna() & ((d["約定日時"].dt.hour + d["約定日時"].dt.minute + d["約定日時"].dt.second) > 0)
    d["約定時刻あり"] = has_time.fillna(False)
    return d

def normalize_yakujyou(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    return normalize_symbol_cols(df.copy())

# ==== ここから：時刻統合と推定の関数 ====
def pick_dt_with_optional_time(df: pd.DataFrame, dt_candidates=None, tm_candidates=None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(pd.NaT, index=[], dtype="datetime64[ns, Asia/Tokyo]")
    dtcol = pick_dt_col(df, preferred=dt_candidates)
    tmcol = pick_time_col(df, preferred=tm_candidates)
    if dtcol and tmcol and (tmcol in df.columns):
        ts = combine_date_time_cols(df, dtcol, tmcol)
        return ts
    if dtcol:
        ts = parse_datetime_from_dtcol(df, dtcol)
        return ts
    return _to_jst_series(pd.Series(pd.NaT, index=df.index), df.index)

def attach_exec_time_from_yak(realized_df: pd.DataFrame, yak_df: pd.DataFrame) -> pd.DataFrame:
    if realized_df.empty or yak_df.empty:
        realized_df = realized_df.copy()
        realized_df["約定日時_推定"] = pd.NaT
        return realized_df
    d = realized_df.copy()
    y = normalize_symbol_cols(clean_columns(yak_df.copy()))
    y["約定日時"] = pick_dt_with_optional_time(y)
    y["__day__"]   = y["約定日時"].dt.date
    price_col = next((c for c in ["約定単価(円)","約定単価（円）","約定価格","価格","約定単価"] if c in y.columns), None)
    qty_col   = next((c for c in ["約定数量(株/口)","約定数量","出来数量","数量","株数","出来高","口数"] if c in y.columns), None)
    side_col  = next((c for c in ["売買","売買区分","売買種別","Side","取引"] if c in y.columns), None)
    if price_col is None:
        for c in y.columns:
            if re.search(r"(約定)?.*(単価|価格)", str(c)): price_col = c; break
    if qty_col is None:
        for c in y.columns:
            if any(k in str(c) for k in ["数量","株数","口数","出来高"]): qty_col = c; break
    y["__price__"]  = to_numeric_jp(y[price_col]) if price_col else np.nan
    y["__qty__"]    = to_numeric_jp(y[qty_col])   if qty_col else np.nan
    y["__action__"] = y[side_col] if side_col else pd.NA

    d["__day__"] = pd.to_datetime(d.get("約定日", d.get("約定日_final")), errors="coerce").dt.date
    y_grp = y.groupby(["__day__","code_key","__action__"])

    est = []
    matched = 0
    for i, row in d.iterrows():
        if row.get("約定時刻あり", False):
            est.append(pd.NaT); continue
        act = row.get("__action__")
        if act not in ("買埋","売埋"):
            est.append(pd.NaT); continue
        key = (row["__day__"], str(row.get("code_key","")).upper(), act)
        if key not in y_grp.groups:
            est.append(pd.NaT); continue
        g = y_grp.get_group(key)
        if g.empty or g["約定日時"].isna().all():
            est.append(pd.NaT); continue
        tp = row.get("__決済単価__", np.nan)
        tq = row.get("__数量__", np.nan)
        score = (g["__price__"] - tp).abs()
        if pd.notna(tq):
            score = score + (g["__qty__"] - tq).abs()*0.001
        idx = score.idxmin()
        est_time = g.loc[idx, "約定日時"]
        est.append(est_time); matched += 1

    d["約定日時_推定"] = pd.Series(est, index=d.index)
    st.caption(f"🧩 実現損益に時刻を推定付与：{matched} 件マッチ（買埋/売埋のみ対象）")
    return d

def enrich_times_lenient(realized_df: pd.DataFrame, yak_df: pd.DataFrame) -> pd.DataFrame:
    if realized_df.empty or yak_df.empty:
        return realized_df
    d = realized_df.copy()
    dt_final = _to_jst_series(d["約定日時_final"] if "約定日時_final" in d.columns else None, d.index)
    no_time = dt_final.isna() | ((dt_final.dt.hour==0) & (dt_final.dt.minute==0) & (dt_final.dt.second==0))
    if not no_time.any():
        return d
    if "約定日_final" in d.columns:
        day_base = pd.to_datetime(d["約定日_final"], errors="coerce")
    elif "約定日" in d.columns:
        day_base = pd.to_datetime(d["約定日"], errors="coerce")
    else:
        day_base = _to_jst_series(d.get("約定日時_final", pd.Series(pd.NaT, index=d.index)), d.index)
    d["__day__"] = (day_base.dt.tz_convert(TZ) if hasattr(day_base.dtype, "tz") else pd.to_datetime(day_base, errors="coerce")).dt.date

    y = normalize_symbol_cols(clean_columns(yak_df.copy()))
    y_dt = pick_dt_with_optional_time(y)
    y = y.assign(__day__=y_dt.dt.date, __dt__=y_dt)

    base = y.dropna(subset=["__dt__"]).copy()
    if base.empty:
        d["約定日時_final"] = dt_final
        return d

    rep_code = (base.dropna(subset=["code_key"])
                    .groupby(["__day__","code_key"])["__dt__"]
                    .apply(lambda s: s.sort_values().iloc[len(s)//2])
                    .rename("__rep_dt_code__"))
    rep_name = (base.dropna(subset=["name_key"])
                    .groupby(["__day__","name_key"])["__dt__"]
                    .apply(lambda s: s.sort_values().iloc[len(s)//2])
                    .rename("__rep_dt_name__"))
    rep_day = (base.groupby(["__day__"])["__dt__"]
                    .apply(lambda s: s.sort_values().iloc[len(s)//2])
                    .rename("__rep_dt_day__"))

    d["__ck__"] = d.get("code_key", pd.Series([""]*len(d), index=d.index)).astype(str).str.upper()
    d["__nk__"] = d.get("name_key", pd.Series([""]*len(d), index=d.index)).astype(str)

    m = d.merge(rep_code.reset_index(), how="left",
                left_on=["__day__","__ck__"], right_on=["__day__","code_key"])
    fill1 = _to_jst_series(m["__rep_dt_code__"], m.index)
    dt_step1 = dt_final.where(~no_time, fill1)

    still_no = dt_step1.isna() | ((dt_step1.dt.hour==0) & (dt_step1.dt.minute==0) & (dt_step1.dt.second==0))
    m2 = d.merge(rep_name.reset_index(), how="left",
                 left_on=["__day__","__nk__"], right_on=["__day__","name_key"])
    fill2 = _to_jst_series(m2["__rep_dt_name__"], m2.index)
    dt_step2 = dt_step1.where(~still_no, fill2)

    still_no2 = dt_step2.isna() | ((dt_step2.dt.hour==0) & (dt_step2.dt.minute==0) & (dt_step2.dt.second==0))
    m3 = d.merge(rep_day.reset_index(), how="left", on="__day__")
    fill3 = _to_jst_series(m3["__rep_dt_day__"], m3.index)
    dt_final_new = dt_step2.where(~still_no2, fill3)

    d["約定日時_final"] = dt_final_new
    return d

def session_of(dt_series: pd.Series) -> pd.Series:
    dt_local = _to_jst_series(dt_series, dt_series.index)
    sec = dt_local.dt.hour*3600 + dt_local.dt.minute*60 + dt_local.dt.second
    out = pd.Series(pd.NA, index=dt_series.index, dtype="object")
    out[(sec >= MORNING_START_SEC) & (sec <= MORNING_END_SEC)]  = "前場"
    out[(sec >= AFTERNOON_START_SEC) & (sec <= AFTERNOON_END_SEC)] = "後場"
    return out

# =========================================================
# 3分足ロード
# =========================================================
@st.cache_data(show_spinner=False)
def load_ohlc_map_from_uploads(files, sig: str):
    ohlc_map = {}
    if not files: return ohlc_map
    for f in files:
        df = read_table_from_upload(f.name, f.getvalue())
        if df.empty: continue

        col_rename_map, found_cols = {}, {}
        CANDIDATES = {
            "time":  ["time","日時","date","datetime","timestamp","Time","Date","Datetime","Timestamp","日付","日付時刻","時刻","時間"],
            "open":  ["open","始値","Open","始","O"],
            "high":  ["high","高値","High","高","H"],
            "low":   ["low","安値","Low","安","L"],
            "close": ["close","終値","Close","終","C"],
        }
        original_cols = {c.lower(): c for c in df.columns}
        for std, cands in CANDIDATES.items():
            for cand in cands:
                lc = cand.lower()
                if lc in original_cols:
                    col_rename_map[original_cols[lc]] = std
                    found_cols[std] = True
                    break
        if len(found_cols) < len(CANDIDATES):  # 必須列が揃わなければスキップ
            continue

        df = df.rename(columns=col_rename_map)
        t = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
        t = _to_jst_series(t, t.index)
        df = df.copy(); df["time"] = t

        def pick_one(df, names):
            for n in names:
                if n in df.columns: return df[n]
            return None
        for key, names in [("VWAP",["VWAP","vwap","Vwap"]),
                           ("MA1", ["MA1","ma1","Ma1"]),
                           ("MA2", ["MA2","ma2","Ma2"]),
                           ("MA3", ["MA3","ma3","Ma3"])]:
            s = pick_one(df, names)
            if s is not None: df[key] = pd.to_numeric(s, errors="coerce")

        df = df.dropna(subset=["time"]).sort_values("time")
        df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)

        key = f.name.rsplit(".",1)[0]
        ohlc_map[key] = df
    return ohlc_map

def extract_code_from_ohlc_key(key: str):
    m = re.search(r'_(?i:(\d{3,5}[a-z]))(?=[,_])', key)
    if m: return m.group(1).upper()
    m2 = re.search(r'_(\d{4,5})(?=[,_])', key)
    if m2: return m2.group(1)
    return None

def guess_name_for_ohlc_key(key: str, code_to_name: dict) -> str | None:
    """
    3分足ファイル名から code を推定し、実現損益/約定履歴から作った CODE_TO_NAME で銘柄名を返す。
    見つからない場合は指数・先物などをヒューリスティックに命名。
    """
    code = extract_code_from_ohlc_key(key)
    name = None
    if code:
        name = code_to_name.get(code.upper())
    if not name:
        ku = key.upper()
        if "NK2251" in ku or "OSE_NK2251" in ku:
            name = "日経225先物"
        elif "NI225" in ku or "TVC_NI225" in ku:
            name = "日経平均"
    return name

def build_ohlc_code_index(ohlc_map: dict):
    idx = {}
    for k in ohlc_map.keys():
        c = extract_code_from_ohlc_key(k)
        if c:
            idx.setdefault(c.upper(), []).append(k)
    return idx

def ohlc_global_date_range(ohlc_map: dict):
    if not ohlc_map: return None, None
    mins, maxs = [], []
    for df in ohlc_map.values():
        if df is None or df.empty or "time" not in df.columns: continue
        t = _to_jst_series(df["time"], df.index)
        if t.notna().any():
            mins.append(t.min().date()); maxs.append(t.max().date())
    if not mins or not maxs: return None, None
    return min(mins), max(maxs)

def download_button_df(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def compute_max_drawdown(series: pd.Series) -> float:
    if series is None: return np.nan
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return np.nan
    arr = s.to_numpy(dtype=float)
    peak, max_dd = -np.inf, 0.0
    for x in arr:
        peak = max(peak, x)
        max_dd = max(max_dd, peak - x)
    return float(max_dd)

# =========================================================
# サイドバー：データアップロード & 設定
# =========================================================
st.sidebar.header("① データアップロード（複数ファイルOK）")
realized_files = st.sidebar.file_uploader("実現損益 CSV/Excel", type=["csv","txt","xlsx"], accept_multiple_files=True)
yakujyou_files = st.sidebar.file_uploader("約定履歴 CSV/Excel", type=["csv","txt","xlsx"], accept_multiple_files=True)
ohlc_files     = st.sidebar.file_uploader("3分足 OHLC CSV/Excel（ファイル名に _7974 等）", type=["csv","txt","xlsx"], accept_multiple_files=True)

sig_realized = files_signature(realized_files)
sig_yakujyou = files_signature(yakujyou_files)
sig_ohlc     = files_signature(ohlc_files)

yakujyou_all = concat_uploaded_tables(yakujyou_files, sig_yakujyou)
yakujyou_all = normalize_yakujyou(clean_columns(yakujyou_all))

realized = concat_uploaded_tables(realized_files, sig_realized)
realized = normalize_realized(clean_columns(realized))

ohlc_map = load_ohlc_map_from_uploads(ohlc_files, sig_ohlc)
CODE_TO_NAME = build_code_to_name_map(realized, yakujyou_all)

# --- 実現損益に約定履歴から時刻推定 → 最終列を作成
realized = attach_exec_time_from_yak(realized, yakujyou_all)

dt_real = _to_jst_series(realized["約定日時"]       if "約定日時" in realized.columns else None, realized.index)
dt_est  = _to_jst_series(realized["約定日時_推定"]   if "約定日時_推定" in realized.columns else None, realized.index)
has_real_clock = dt_real.notna() & ((dt_real.dt.hour + dt_real.dt.minute + dt_real.dt.second) > 0)
realized["約定日時_final"] = dt_real.where(has_real_clock, dt_est)

# ゆるめ補完（code→name→day）
realized = enrich_times_lenient(realized, yakujyou_all)

# 約定日_final
if "約定日" in realized.columns:
    day_raw_date = pd.to_datetime(realized["約定日"], errors="coerce").dt.date
else:
    day_raw_date = pd.Series([pd.NaT]*len(realized), index=realized.index, dtype="object")
realized["約定日_final"] = np.where(pd.notna(day_raw_date),
                                  day_raw_date,
                                  _to_jst_series(realized["約定日時_final"], realized.index).dt.date)

# ===== 診断 =====
with st.expander("🛠 実現損益 正規化の診断"):
    st.write("行数:", len(realized))
    if not realized.empty:
        cols = [c for c in ["約定日","約定日_final","約定日時","約定日時_推定","約定日時_final",
                            "約定時刻あり","実現損益[円]","取引","銘柄名","銘柄コード","code_key",
                            "__決済単価__","__数量__"] if c in realized.columns]
        st.write("検出列:", cols)
        st.write(realized[cols].head(12))

# フィルタ
st.sidebar.header("② トレード種別フィルタ")
trade_type = st.sidebar.radio("対象", ["全体","デイトレード（一般）","スイングトレード（制度）"], index=0)

def apply_trade_type_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "信用区分" not in df.columns: return df
    if trade_type.startswith("全体"): return df
    s = df["信用区分"].astype(str)
    if "デイトレ" in trade_type or "一般" in trade_type:
        return df[s.str.contains("一般", na=False)]
    if "スイング" in trade_type or "制度" in trade_type:
        return df[s.str.contains("制度", na=False)]
    return df

# === 時間別集計のオプション ===
st.sidebar.header("③ 集計オプション")
drop_noclock = st.sidebar.checkbox("約定時刻なしを除外（時間別）", value=True)
include_off_hours = st.sidebar.checkbox("市場時間外も含める（時間別）", value=False,
                                        help="チェックすると 9:00〜15:30 以外の時刻も時間別集計に含めます")

# 期間フィルタ
st.subheader("期間フィルタ")
c1,c2,c3 = st.columns([2,2,3])
with c1:
    span = st.radio("クイック選択", ["全期間","今日","今週","今月","今年","カスタム"], horizontal=True, index=0, key="span")
with c2:
    start = st.date_input("開始日", value=date.today()-timedelta(days=30), disabled=(span!="カスタム"))
with c3:
    end = st.date_input("終了日", value=date.today(), disabled=(span!="カスタム"))

def filter_by_span(df, dt_col):
    if df.empty: return df
    if dt_col not in df.columns: return df
    dt = pd.to_datetime(df[dt_col], errors="coerce")
    today = date.today()
    if span=="全期間": return df
    if span=="今日": s,e = today, today
    elif span=="今週":
        mon = today - timedelta(days=today.weekday()); s,e = mon, mon+timedelta(days=6)
    elif span=="今月":
        first = today.replace(day=1)
        next_f = date(first.year+1,1,1) if first.month==12 else date(first.year, first.month+1, 1)
        s,e = first, next_f - timedelta(days=1)
    elif span=="今年":
        s,e = date(today.year,1,1), date(today.year,12,31)
    else:
        s,e = start, end
    mask = (dt.dt.date>=s) & (dt.dt.date<=e)
    return df.loc[mask]

realized_f = apply_trade_type_filter(filter_by_span(realized, "約定日_final"))

# =========================================================
# KPI
# =========================================================
st.subheader("KPI")
if not realized_f.empty and "実現損益[円]" in realized_f.columns:
    realized_f = realized_f.copy()
    realized_f["実現損益[円]"] = to_numeric_jp(realized_f["実現損益[円]"])

pl = None
if not realized_f.empty and "実現損益[円]" in realized_f.columns:
    pl = to_numeric_jp(realized_f["実現損益[円]"]).dropna()

c1,c2,c3 = st.columns(3)
with c1:
    total_pl = pl.sum() if pl is not None and not pl.empty else np.nan
    st.metric("実現損益（選択期間）", f"{int(total_pl):,} 円" if pd.notna(total_pl) else "—")
with c2:
    n_trades = int(pl.shape[0]) if pl is not None else 0
    st.metric("取引回数", f"{n_trades:,}" if n_trades else "—")
with c3:
    avg_pl = pl.mean() if pl is not None and not pl.empty else np.nan
    st.metric("平均損益（/回）", f"{int(round(avg_pl)):,} 円" if pd.notna(avg_pl) else "—")

# =========================================================
# タブ
# =========================================================
tab1, tab1b, tab2, tab3, tab4, tab5 = st.tabs([
    "集計（期間別）",
    "集計（時間別）",
    "累計損益",
    "個別銘柄",
    "ランキング",
    "3分足 IN/OUT + 指標"
])

# ---- 1) 期間別
with tab1:
    st.markdown("### 実現損益（期間別集計）")
    if realized_f.empty or "実現損益[円]" not in realized_f.columns:
        st.info("実現損益データが必要です。")
    else:
        r = realized_f.copy()
        r = r[r["実現損益[円]"].notna()]
        if r.empty:
            st.info("実現損益の数値が見つかりません。")
        else:
            r["日"] = pd.to_datetime(r["約定日_final"], errors="coerce").dt.date
            r["週"] = (pd.to_datetime(r["日"]) - pd.to_timedelta(pd.to_datetime(r["日"]).dt.weekday, unit="D")).dt.date
            r["月"] = pd.to_datetime(r["日"]).dt.to_period("M").dt.to_timestamp().dt.date
            r["年"] = pd.to_datetime(r["日"]).dt.to_period("Y").dt.to_timestamp().dt.date

            for label,col in [("日別","日"),("週別","週"),("月別","月"),("年別","年")]:
                g = r.groupby(col, as_index=False)["実現損益[円]"].sum().sort_values(col)
                # 表示整形：数値はカンマ区切り、日付は YYYY-MM-DD 文字列
                disp = g.copy()
                disp["日付"] = pd.to_datetime(disp[col]).dt.strftime("%Y-%m-%d")
                disp["実現損益[円]"] = disp["実現損益[円]"].round(0).astype("Int64").map(lambda x: f"{x:,}")
                st.write(f"**{label}**")
                st.dataframe(disp[["日付","実現損益[円]"]], use_container_width=True, hide_index=True)
                download_button_df(disp[["日付","実現損益[円]"]], f"⬇ CSVダウンロード（{label}）", f"{col}_pl.csv")

                # グラフ：X軸は時間表記なしの 'YYYY-MM-DD'
                fig_bar = go.Figure([go.Bar(x=disp["日付"], y=g["実現損益[円]"], name=f"{label} 実現損益")])
                fig_bar.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=300,
                                      xaxis_title="日付", yaxis_title="実現損益[円]")
                st.plotly_chart(fig_bar, use_container_width=True)

# ---- 1b) 時間別
def pick_best_exec_time_series(df: pd.DataFrame, index=None) -> pd.Series:
    """final→推定→元 の順で“時刻あり”だけを採用（JST, tz-aware）"""
    idx = df.index if index is None else index
    base = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns, Asia/Tokyo]")
    def fill_from(colname: str, base: pd.Series) -> pd.Series:
        if colname not in df.columns: return base
        s = _to_jst_series(df[colname], df.index)
        has_clock = s.notna() & ((s.dt.hour + s.dt.minute + s.dt.second) > 0)
        return base.mask(base.isna() & has_clock, s)
    base = fill_from("約定日時_final", base)
    base = fill_from("約定日時_推定", base)
    base = fill_from("約定日時",     base)
    return base

with tab1b:
    st.markdown("### 実現損益（時間別・1時間ごと）")
    if realized_f.empty or "実現損益[円]" not in realized_f.columns:
        st.info("実現損益データが必要です。")
    else:
        d0 = realized_f.copy()
        d0 = d0[d0["実現損益[円]"].notna()]
        if d0.empty:
            st.info("実現損益の数値が見つかりません。")
        else:
            # final→推定→元 の順で“時刻あり”だけを採用（JST, tz-aware）
            dt0 = pick_best_exec_time_series(d0, index=d0.index)

            # 診断
            def count_has_clock(col):
                if col not in d0.columns: return 0
                s = _to_jst_series(d0[col], d0.index)
                return int((s.notna() & ((s.dt.hour + s.dt.minute + s.dt.second) > 0)).sum())
            with st.expander("🧪 時間別 集計の診断", expanded=False):
                st.write("候補別“時刻あり”件数：",
                         {"約定日時_final": count_has_clock("約定日時_final"),
                          "約定日時_推定": count_has_clock("約定日時_推定"),
                          "約定日時":     count_has_clock("約定日時")})
                st.write("dt0 サンプル（先頭5件）:")
                st.write(pd.DataFrame({"dt0": dt0}).head())

            # 「真に時刻あり」
            hh0, mm0, ss0 = dt0.dt.hour, dt0.dt.minute, dt0.dt.second
            has_clock0 = dt0.notna() & ((hh0.fillna(0)*3600 + mm0.fillna(0)*60 + ss0.fillna(0)) > 0)

            cnt_all      = len(d0)
            cnt_time     = int(has_clock0.sum())
            cnt_midnight = int(((dt0.notna()) & ~has_clock0).sum())
            st.caption(f"⏱️ 真に時刻あり: {cnt_time}/{cnt_all} | 00:00扱い: {cnt_midnight}")

            # ① 約定時刻なしを除外
            if drop_noclock:
                d, dt = d0.loc[has_clock0].copy(), dt0.loc[has_clock0]
            else:
                d, dt = d0.copy(), dt0.copy()

            if d.empty:
                st.info("“約定時刻あり”のレコードがありません（00:00やNaTは除外）。")
            else:
                # ② 市場時間フィルタ
                if include_off_hours:
                    d_in, dt_in = d, dt
                    info_suffix = "（市場時間外も含む）"
                else:
                    sec = (dt.dt.hour*3600 + dt.dt.minute*60 + dt.dt.second).astype(int)
                    mask_mkt = (sec >= MORNING_START_SEC) & (sec <= AFTERNOON_END_SEC)
                    d_in, dt_in = d.loc[mask_mkt].copy(), dt.loc[mask_mkt]
                    info_suffix = ""

                if d_in.empty:
                    st.info("市場時間内の“約定時刻あり”レコードがありませんでした。" + ("（オプションで市場時間外を含められます）" if not include_off_hours else ""))
                else:
                    d_in["PL"]  = to_numeric_jp(d_in["実現損益[円]"])
                    d_in["win"] = d_in["PL"] > 0

                    hour_floor = dt_in.dt.floor("H")
                    hour_x = pd.to_datetime([datetime(2000,1,1,h.hour,0,0, tzinfo=TZ) for h in hour_floor])
                    d_in["hour_x"] = hour_x

                    x_range = [datetime(2000,1,1,9,0, tzinfo=TZ), datetime(2000,1,1,15,30, tzinfo=TZ)]
                    x_ticks = pd.date_range(x_range[0], x_range[1], freq="60min", inclusive="both")
                    base = pd.DataFrame({"hour_x": x_ticks})

                    by = d_in.groupby("hour_x", as_index=False).agg(
                        収支=("PL","sum"),
                        取引回数=("PL","count"),
                        勝率=("win","mean"),
                        平均損益=("PL","mean")
                    )
                    by = base.merge(by, on="hour_x", how="left")

                    # 表示整形（ご要望）
                    disp = by.copy()
                    disp["時間"] = disp["hour_x"].dt.strftime("%H:%M")
                    disp["収支"] = disp["収支"].fillna(0).round(0).astype("Int64").map(lambda x: f"{x:,}")
                    disp["取引回数"] = disp["取引回数"].fillna(0).astype("Int64")
                    disp["勝率"] = (disp["勝率"].fillna(0)*100).round(1).map(lambda x: f"{x:.1f}%")
                    disp["平均損益"] = disp["平均損益"].fillna(0).round(0).astype("Int64").map(lambda x: f"{x:,}")
                    st.dataframe(disp[["時間","収支","取引回数","勝率","平均損益"]],
                                 use_container_width=True, hide_index=True)
                    download_button_df(disp[["時間","収支","取引回数","勝率","平均損益"]],
                                       "⬇ CSVダウンロード（時間別）", "hourly_stats.csv")

                    # グラフは数値のまま（X軸は時刻表示）
                    fig_h_pl = go.Figure([go.Bar(x=by["hour_x"], y=by["収支"], name="収支（合計）")])
                    fig_h_pl.update_layout(title=f"時間別 収支（合計）{info_suffix}", xaxis_title="時間", yaxis_title="円",
                                           margin=dict(l=10,r=10,t=30,b=10), height=300,
                                           xaxis=dict(tickformat="%H:%M", range=x_range))
                    st.plotly_chart(fig_h_pl, use_container_width=True)

                    fig_h_wr = go.Figure([go.Scatter(x=by["hour_x"], y=by["勝率"]*100, mode="lines+markers", name="勝率")])
                    fig_h_wr.update_layout(title=f"時間別 勝率（全体）{info_suffix}", xaxis_title="時間", yaxis_title="勝率（%）",
                                           margin=dict(l=10,r=10,t=30,b=10), height=300,
                                           yaxis=dict(range=[0,100]),
                                           xaxis=dict(tickformat="%H:%M", range=x_range))
                    st.plotly_chart(fig_h_wr, use_container_width=True)

                    fig_h_cnt = go.Figure([go.Bar(x=by["hour_x"], y=by["取引回数"], name="取引回数")])
                    fig_h_cnt.update_layout(title=f"時間別 取引回数{info_suffix}", xaxis_title="時間", yaxis_title="回",
                                            margin=dict(l=10,r=10,t=30,b=10), height=300,
                                            xaxis=dict(tickformat="%H:%M", range=x_range))
                    st.plotly_chart(fig_h_cnt, use_container_width=True)

                    fig_h_avg = go.Figure([go.Bar(x=by["hour_x"], y=by["平均損益"], name="平均損益（/回）")])
                    fig_h_avg.update_layout(title=f"時間別 平均損益（/回）{info_suffix}", xaxis_title="時間", yaxis_title="円/回",
                                            margin=dict(l=10,r=10,t=30,b=10), height=300,
                                            xaxis=dict(tickformat="%H:%M", range=x_range))
                    st.plotly_chart(fig_h_avg, use_container_width=True)

                    # 前場 / 後場 比較
                    st.markdown("### 前場 / 後場 比較" + info_suffix)
                    ses = session_of(dt_in)
                    d_in["セッション"] = ses
                    cmp = d_in.dropna(subset=["セッション"]).groupby("セッション").agg(
                        収支=("PL","sum"), 取引回数=("PL","count"),
                        勝率=("win","mean"), 平均損益=("PL","mean")
                    ).reset_index()
                    cmp_disp = cmp.copy()
                    cmp_disp["収支"] = cmp_disp["収支"].round(0).astype("Int64").map(lambda x: f"{x:,}")
                    cmp_disp["取引回数"] = cmp_disp["取引回数"].astype("Int64")
                    cmp_disp["勝率"] = (cmp_disp["勝率"]*100).round(1).map(lambda x: f"{x:.1f}%")
                    cmp_disp["平均損益"] = cmp_disp["平均損益"].round(0).astype("Int64").map(lambda x: f"{x:,}")
                    st.dataframe(cmp_disp, use_container_width=True, hide_index=True)
                    download_button_df(cmp_disp, "⬇ CSVダウンロード（前場後場比較）", "am_pm_compare.csv")

                    # 累積勝率（5分）
                    st.markdown("### 累積勝率の時間推移（全期間・5分ビン）" + info_suffix)
                    five = dt_in.dt.floor("5min")
                    x_five = pd.to_datetime([datetime(2000,1,1,t.hour,t.minute,0, tzinfo=TZ) for t in five.dt.time])
                    tmp = pd.DataFrame({"x": x_five, "win": d_in["win"].astype(float), "cnt": 1.0})
                    grid = pd.DataFrame({"x": pd.date_range(datetime(2000,1,1,9,0, tzinfo=TZ),
                                                            datetime(2000,1,1,15,30, tzinfo=TZ),
                                                            freq="5min", inclusive="both")})
                    agg5 = tmp.groupby("x").agg(win_sum=("win","sum"), cnt=("cnt","sum")).reset_index()
                    grid = grid.merge(agg5, on="x", how="left").fillna(0.0)
                    grid["cum_wr"] = np.where(grid["cnt"].cumsum()>0,
                                              grid["win_sum"].cumsum()/grid["cnt"].cumsum()*100.0, np.nan)
                    fig_cum = go.Figure([go.Scatter(x=grid["x"], y=grid["cum_wr"], mode="lines", name="累積勝率")])
                    fig_cum.update_layout(title=f"累積勝率（時間の経過とともに）{info_suffix}", xaxis_title="時間", yaxis_title="勝率（%）",
                                          margin=dict(l=10,r=10,t=30,b=10), height=320,
                                          yaxis=dict(range=[0,100]),
                                          xaxis=dict(tickformat="%H:%M",
                                                     range=[datetime(2000,1,1,9,0, tzinfo=TZ),
                                                            datetime(2000,1,1,15,30, tzinfo=TZ)]))
                    st.plotly_chart(fig_cum, use_container_width=True)

# ---- 2) 累計損益
with tab2:
    st.markdown("### 累計実現損益（選択期間内、日次ベース）")
    if realized_f.empty or "実現損益[円]" not in realized_f.columns:
        st.info("実現損益データが必要です。")
    else:
        d = realized_f.copy()
        d = d[d["実現損益[円]"].notna()]
        if d.empty:
            st.info("実現損益の数値が見つかりません。")
        else:
            d["日"] = pd.to_datetime(d["約定日_final"], errors="coerce").dt.date
            seq = d.groupby("日", as_index=False)["実現損益[円]"].sum().sort_values("日")
            seq["累計"] = pd.to_numeric(seq["実現損益[円]"], errors="coerce").cumsum()

            # 表示整形：テーブルはカンマ区切り、グラフのXはYYYY-MM-DD文字列
            seq_disp = seq.copy()
            seq_disp["日付"] = pd.to_datetime(seq_disp["日"]).dt.strftime("%Y-%m-%d")
            seq_disp["実現損益[円]"] = seq_disp["実現損益[円]"].round(0).astype("Int64").map(lambda x: f"{x:,}")
            seq_disp["累計"] = seq_disp["累計"].round(0).astype("Int64").map(lambda x: f"{x:,}")
            st.dataframe(seq_disp[["日付","実現損益[円]","累計"]], use_container_width=True, hide_index=True)
            download_button_df(seq_disp[["日付","実現損益[円]","累計"]], "⬇ CSVダウンロード（累計・日次）", "cumulative_daily_pl.csv")

            left,right = st.columns(2)
            with left:
                fig_bar = go.Figure([go.Bar(x=seq_disp["日付"], y=seq["実現損益[円]"], name="日次 実現損益")])
                fig_bar.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=350, xaxis_title="日付", yaxis_title="実現損益[円]")
                st.plotly_chart(fig_bar, use_container_width=True)
            with right:
                fig_line = go.Figure([go.Scatter(x=seq_disp["日付"], y=seq["累計"], mode="lines", name="累計")])
                fig_line.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=350, xaxis_title="日付", yaxis_title="累計実現損益[円]")
                st.plotly_chart(fig_line, use_container_width=True)

# ---- 3) 個別銘柄
def per_symbol_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["銘柄コード","銘柄名","実現損益合計","取引回数","1回平均損益","勝率"])
    d = normalize_symbol_cols(df.copy())
    if "実現損益[円]" in d.columns:
        d["実現損益"] = to_numeric_jp(d["実現損益[円]"])
    else:
        cand = next((c for c in d.columns if ("実現" in str(c) and "損益" in str(c))), None)
        if cand is None: 
            return pd.DataFrame(columns=["銘柄コード","銘柄名","実現損益合計","取引回数","1回平均損益","勝率"])
        d["実現損益"] = to_numeric_jp(d[cand])
    d = d[d["実現損益"].notna()]
    if d.empty:
        return pd.DataFrame(columns=["銘柄コード","銘柄名","実現損益合計","取引回数","1回平均損益","勝率"])
    d["win"] = d["実現損益"]>0
    d["group_key"] = np.where(d["code_key"].notna()&(d["code_key"]!=""), d["code_key"], "NAMEONLY::"+d["name_key"].astype(str))
    agg = d.groupby("group_key").agg({"実現損益":["sum","count","mean"], "win":["mean"]})
    agg.columns = ["実現損益合計","取引回数","1回平均損益","勝率"]
    rep_name = d.groupby("group_key").apply(representative_name).rename("銘柄名")
    code_col = d.groupby("group_key")["code_key"].first().rename("銘柄コード")
    out = agg.join(rep_name).join(code_col).reset_index(drop=True).sort_values("実現損益合計", ascending=False)
    out["銘柄コード"] = out["銘柄コード"].fillna("—")
    return out

with tab3:
    st.markdown("### 個別銘柄（勝率・実現損益）")
    if realized_f.empty or "実現損益[円]" not in realized_f.columns:
        st.info("実現損益データが必要です。")
    else:
        sym = per_symbol_stats(realized_f)
        if not sym.empty:
            disp = sym.copy()
            if "実現損益合計" in disp.columns: disp["実現損益合計"] = disp["実現損益合計"].round(0).astype("Int64").map(lambda x: f"{x:,}")
            if "1回平均損益" in disp.columns: disp["1回平均損益"] = disp["1回平均損益"].round(0).astype("Int64").map(lambda x: f"{x:,}")
            if "勝率" in disp.columns: disp["勝率"] = (disp["勝率"]*100).round(1).map(lambda x: f"{x:.1f}%")
            order = ["銘柄コード","銘柄名","実現損益合計","取引回数","1回平均損益","勝率"]
            cols = [c for c in order if c in disp.columns] + [c for c in disp.columns if c not in order]
            st.dataframe(disp[cols], use_container_width=True, hide_index=True)
            download_button_df(disp[cols], "⬇ CSVダウンロード（個別銘柄）", "per_symbol_stats.csv")
        else:
            st.info("集計できるデータがありません。")

# ---- 4) ランキング
with tab4:
    st.markdown("### ランキング（選択期間）")
    if realized_f.empty or "実現損益[円]" not in realized_f.columns:
        st.info("実現損益データが必要です。")
    else:
        d = normalize_symbol_cols(realized_f.copy())
        d["実現損益"] = to_numeric_jp(d["実現損益[円]"]) if "実現損益[円]" in d.columns else np.nan
        d = d[d["実現損益"].notna()]
        if d.empty:
            st.info("実現損益の数値が見つかりません。")
        else:
            d["group_key"] = np.where(d["code_key"].notna()&(d["code_key"]!=""), d["code_key"], "NAMEONLY::"+d["name_key"].astype(str))
            by_symbol = d.groupby("group_key").agg({"実現損益":["count","sum","mean"]})
            by_symbol.columns = ["取引回数","実現損益合計","1回平均損益"]
            rep_name = d.groupby("group_key").apply(representative_name).rename("銘柄名")
            code_col = d.groupby("group_key")["code_key"].first().rename("銘柄コード")
            out = by_symbol.join(rep_name).join(code_col).reset_index(drop=True)
            out["銘柄コード"] = out["銘柄コード"].fillna("—")
            if "1回平均損益" in out.columns: out["1回平均損益"] = out["1回平均損益"].round(0).astype("Int64").map(lambda x: f"{x:,}")
            if "実現損益合計" in out.columns: out["実現損益合計"] = out["実現損益合計"].round(0).astype("Int64").map(lambda x: f"{x:,}")
            left,right = st.columns(2)
            with left:
                sort_key = st.selectbox("ソート指標", ["実現損益合計","取引回数","1回平均損益"], index=0)
            with right:
                topn = st.slider("表示件数", 5, 100, 20)
            out = out.sort_values(sort_key, ascending=False).head(topn)
            order = ["銘柄コード","銘柄名","実現損益合計","取引回数","1回平均損益"]
            cols = [c for c in order if c in out.columns] + [c for c in out.columns if c not in order]
            st.dataframe(out[cols], use_container_width=True, hide_index=True)
            download_button_df(out[cols], "⬇ CSVダウンロード（ランキング）", "ranking.csv")

# =========================================================
# 5) 3分足 IN/OUT + 指標（先に日付を選び、その日にデータがある銘柄だけ選択）
# =========================================================

# --- 既に定義済みであれば再定義しないヘルパー ---
try:
    guess_name_for_ohlc_key
except NameError:
    def guess_name_for_ohlc_key(key: str, code_to_name: dict) -> str | None:
        code = extract_code_from_ohlc_key(key)
        name = None
        if code:
            name = code_to_name.get(str(code).upper())
        if not name:
            ku = key.upper()
            if "NK2251" in ku or "OSE_NK2251" in ku:
                name = "日経225先物"
            elif "NI225" in ku or "TVC_NI225" in ku:
                name = "日経平均"
        return name

# --- IN/OUTを4分類（買建/売建/売埋/買埋）で近傍バーへスナップ ---
def align_trades_to_ohlc(ohlc: pd.DataFrame, trades: pd.DataFrame, max_gap_min=6):
    """約定（買建/売建/売埋/買埋）をOHLCの最も近いバーに結びつける。"""
    if ohlc is None or ohlc.empty or trades is None or trades.empty:
        return pd.DataFrame(columns=["time","price","side","qty","label4"])

    tdf = trades.copy()
    tdf["約定日時"] = _to_jst_series(tdf["約定日時"] if "約定日時" in tdf.columns else None, tdf.index)

    price_col = next((c for c in ["約定単価(円)","約定単価（円）","約定価格","価格","約定単価"] if c in tdf.columns), None)
    qty_col   = next((c for c in ["約定数量(株/口)","約定数量","出来数量","数量","株数","出来高","口数"] if c in tdf.columns), None)
    side_col  = next((c for c in ["売買","売買区分","売買種別","Side","取引"] if c in tdf.columns), None)
    if price_col is None:
        for c in tdf.columns:
            if re.search(r"(約定)?.*(単価|価格)", str(c)): price_col = c; break
    if qty_col is None:
        for c in tdf.columns:
            if any(k in str(c) for k in ["数量","株数","口数","出来高"]): qty_col = c; break

    tdf["price"] = to_numeric_jp(tdf[price_col]) if price_col else np.nan
    tdf["qty"]   = to_numeric_jp(tdf[qty_col])   if qty_col else np.nan
    tdf["side"]  = tdf[side_col].astype(str) if side_col else ""

    def classify_side4(s: str) -> str | None:
        s = str(s)
        if "買建" in s: return "買建"
        if "売建" in s: return "売建"
        if "売埋" in s: return "売埋"
        if "買埋" in s: return "買埋"
        # 英語・記号のヒューリスティック
        if ("買" in s and ("新規" in s or "建" in s)) or re.search(r"\bBUY\b.*\b(OPEN|NEW)\b", s, re.I): return "買建"
        if ("売" in s and ("新規" in s or "建" in s)) or re.search(r"\bSELL\b.*\b(OPEN|NEW)\b", s, re.I): return "売建"
        if ("売" in s and ("返済" in s or "決済" in s)) or re.search(r"\bSELL\b.*\b(CLOSE)\b", s, re.I): return "売埋"
        if ("買" in s and ("返済" in s or "決済" in s)) or re.search(r"\bBUY\b.*\b(CLOSE|COVER)\b", s, re.I): return "買埋"
        return None

    tdf["label4"] = tdf["side"].map(classify_side4)

    odf = ohlc.copy()
    tt = _to_jst_series(odf["time"], odf.index)
    odf = odf.set_index(tt)

    out_rows = []
    for _, row in tdf.iterrows():
        t0 = row["約定日時"]
        if pd.isna(t0) or not row.get("label4"):
            continue
        lo = t0 - pd.Timedelta(minutes=max_gap_min)
        hi = t0 + pd.Timedelta(minutes=max_gap_min)
        window = odf.loc[lo:hi]
        if window.empty:
            continue
        idx = (window.index - t0).abs().argmin()
        near_time = window.index[idx]
        price_on_bar = window.loc[near_time, "close"]
        out_rows.append({
            "time": near_time, "price": price_on_bar,
            "side": row["side"], "qty": row["qty"], "label4": row["label4"]
        })
    return pd.DataFrame(out_rows)

# --- ローソク＋指標（時間レンジ＆高さを外から指定可） ---
def make_candle_with_indicators(df: pd.DataFrame, title="", height=560, x_range=None):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df["time"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC", showlegend=False
    ))
    if "VWAP" in df.columns and df["VWAP"].notna().any():
        fig.add_trace(go.Scatter(x=df["time"], y=df["VWAP"], name="VWAP", mode="lines",
                                 line=dict(color=COLOR_VWAP, width=1.3)))
    if "MA1" in df.columns and df["MA1"].notna().any():
        fig.add_trace(go.Scatter(x=df["time"], y=df["MA1"], name="MA1", mode="lines",
                                 line=dict(color=COLOR_MA1, width=1.3)))
    if "MA2" in df.columns and df["MA2"].notna().any():
        fig.add_trace(go.Scatter(x=df["time"], y=df["MA2"], name="MA2", mode="lines",
                                 line=dict(color=COLOR_MA2, width=1.3)))
    if "MA3" in df.columns and df["MA3"].notna().any():
        fig.add_trace(go.Scatter(x=df["time"], y=df["MA3"], name="MA3", mode="lines",
                                 line=dict(color=COLOR_MA3, width=1.3)))

    fig.update_layout(
        title=title, height=height, margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        xaxis=dict(showgrid=False, range=x_range),
        yaxis=dict(showgrid=True)
    )
    return fig

with tab5:
    st.markdown("### 3分足 IN/OUT + 指標（VWAP/MA1/MA2/MA3）")
    if not ohlc_map:
        st.info("3分足OHLCファイルをアップロードしてください。")
    else:
        # まず全ファイルから利用可能な日付範囲を推定
        dmin, dmax = ohlc_global_date_range(ohlc_map)
        if dmin is None or dmax is None:
            st.info("有効な日時列が見つかりませんでした。")
        else:
            c1, c2, c3 = st.columns([2,2,1])
            with c1:
                sel_date = st.date_input("表示日を選択", value=dmin, min_value=dmin, max_value=dmax)
            with c2:
                enlarge = st.toggle("🔍 拡大表示", value=False, help="チェックでチャートを大きくします")
            with c3:
                ht = LARGE_CHART_HEIGHT if enlarge else MAIN_CHART_HEIGHT

            # 表示時間レンジ（固定 9:00〜15:30）
            t0 = pd.Timestamp(f"{sel_date} 09:00", tz=TZ)
            t1 = pd.Timestamp(f"{sel_date} 15:30", tz=TZ)
            x_range = [t0, t1]

            # 選択日のデータがある銘柄（ファイルキー）だけを提示
            keys_that_day = []
            for k, df in ohlc_map.items():
                if df is None or df.empty: 
                    continue
                vw = df[(df["time"]>=t0) & (df["time"]<=t1)]
                if not vw.empty:
                    keys_that_day.append(k)

            if not keys_that_day:
                st.info("選択日のデータがある銘柄が見つかりません。別の日付を選んでください。")
            else:
                # ラベルに銘柄名を表示
                options = sorted(keys_that_day)
                def _fmt_label(k):
                    nm = guess_name_for_ohlc_key(k, CODE_TO_NAME)
                    return f"{k}（{nm}）" if nm else k

                sel_key = st.selectbox("銘柄（ファイル名）を選択", options=options, index=0, format_func=_fmt_label)
                sel_name = guess_name_for_ohlc_key(sel_key, CODE_TO_NAME)
                if sel_name:
                    st.caption(f"想定銘柄名: **{sel_name}**")

                view = ohlc_map[sel_key]
                view = view[(view["time"]>=t0) & (view["time"]<=t1)].copy()
                if view.empty:
                    st.info(f"{sel_key}：{sel_date} の3分足が見つかりません。")
                else:
                    # 該当コードの約定を当日抽出
                    yak = yakujyou_all.copy()
                    if "code_key" in yak.columns:
                        this_code = extract_code_from_ohlc_key(sel_key) or ""
                        if this_code:
                            yak = yak[yak["code_key"].astype(str).str.upper()==this_code.upper()]
                    y_dtcol = pick_dt_col(yak) or "約定日"
                    yak = yak.copy()
                    yak["約定日時"] = pick_dt_with_optional_time(yak) if y_dtcol in yak.columns else _to_jst_series(pd.Series(pd.NaT, index=yak.index), yak.index)
                    yak = yak[yak["約定日時"].notna()]
                    yak = yak[(yak["約定日時"]>=t0) & (yak["約定日時"]<=t1)]

                    # 近傍バーへスナップ（買建/売建/売埋/買埋）
                    trades = align_trades_to_ohlc(view, yak, max_gap_min=6) if not yak.empty else pd.DataFrame(columns=["time","price","side","qty","label4"])

                    # メインチャート（サイズ統一・時間固定）
                    title_text = f"{sel_name} [{sel_key}]" if sel_name else sel_key
                    fig = make_candle_with_indicators(view, title=title_text, height=ht, x_range=x_range)

                    # 4分類マーカー
                    marker_styles = {
                        "買建": dict(symbol="triangle-up",        size=11, color="#2ca02c", line=dict(width=1.2)),
                        "売建": dict(symbol="triangle-down",      size=11, color="#d62728", line=dict(width=1.2)),
                        "売埋": dict(symbol="triangle-down-open", size=12, color="#9467bd", line=dict(width=1.4)),
                        "買埋": dict(symbol="triangle-up-open",   size=12, color="#1f77b4", line=dict(width=1.4)),
                    }
                    if not trades.empty:
                        for label, mk in marker_styles.items():
                            sub = trades[trades["label4"] == label]
                            if not sub.empty:
                                fig.add_trace(go.Scatter(
                                    x=sub["time"], y=sub["price"], mode="markers",
                                    name=label, marker=mk,
                                    hovertemplate=f"{label}<br>%{{x|%H:%M}}<br>¥%{{y:.0f}}<extra></extra>"
                                ))

                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

                # 同日の下に：日経先物 / 日経平均（時間レンジ固定・サイズ統一）
                # 候補キー（当日データありに限定）
                fut_keys = [k for k in keys_that_day if ("NK2251" in k or "OSE_NK2251" in k)]
                idx_keys = [k for k in keys_that_day if ("NI225" in k or "TVC_NI225" in k)]

                st.markdown("#### 日経先物（NK225mini等）")
                if fut_keys:
                    k = fut_keys[0]
                    df = ohlc_map.get(k)
                    vw = df[(df["time"]>=t0) & (df["time"]<=t1)].copy()
                    if not vw.empty:
                        nm = guess_name_for_ohlc_key(k, CODE_TO_NAME)
                        ttl = f"{nm} [{k}]" if nm else k
                        figx = make_candle_with_indicators(vw, title=ttl, height=ht, x_range=x_range)
                        st.plotly_chart(figx, use_container_width=True, config={"displayModeBar": True})
                    else:
                        st.info(f"{k}：{sel_date} のデータなし。")
                else:
                    st.info("日経先物（`OSE_NK2251!` など）が見つかりません。")

                st.markdown("#### 日経平均")
                if idx_keys:
                    k = idx_keys[0]
                    df = ohlc_map.get(k)
                    vw = df[(df["time"]>=t0) & (df["time"]<=t1)].copy()
                    if not vw.empty:
                        nm = guess_name_for_ohlc_key(k, CODE_TO_NAME)
                        ttl = f"{nm} [{k}]" if nm else k
                        figx = make_candle_with_indicators(vw, title=ttl, height=ht, x_range=x_range)
                        st.plotly_chart(figx, use_container_width=True, config={"displayModeBar": True})
                    else:
                        st.info(f"{k}：{sel_date} のデータなし。")
                else:
                    st.info("日経平均（`TVC_NI225` など）が見つかりません。")
