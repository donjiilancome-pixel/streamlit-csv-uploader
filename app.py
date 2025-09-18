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

# =========================================================
# 基本設定
# =========================================================
st.set_page_config(page_title="デイトレ結果ダッシュボード", page_icon="📈", layout="wide")
st.title("📈 デイトレ結果ダッシュボード（VWAP/MA対応・3分足＋IN/OUT）")
st.caption("複数ファイルアップロード対応・3分足: Asia/Tokyo / 9:00–15:30・信用区分フィルタ（全体/一般=デイ/制度=スイング）")

TZ = ZoneInfo("Asia/Tokyo")
MAIN_CHART_HEIGHT = 600   # 標準の高さ
LARGE_CHART_HEIGHT = 860  # 拡大表示時の高さ

# 時間帯（前場/後場）
MORNING_START_SEC = 9*3600
MORNING_END_SEC   = 11*3600 + 30*60
AFTERNOON_START_SEC = 12*3600 + 30*60
AFTERNOON_END_SEC   = 15*3600 + 30*60

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
    """日本語CSVでよくある表記を数値化。 (123)→-123, 全角マイナス, 桁区切り, 円/株/％など除去"""
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

# ---- アップロード用の堅牢リーダー（CSV/TXT/XLSX）
@st.cache_data(show_spinner=False)
def read_table_from_upload(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    # Excel
    if file_name.lower().endswith(".xlsx"):
        try:
            return clean_columns(pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl"))
        except Exception:
            return pd.DataFrame()

    # テキスト（CSV/TSV/その他区切り）
    text = None
    for enc in ["utf-8-sig","utf-8","cp932","shift_jis","euc_jp"]:
        try:
            text = file_bytes.decode(enc)
            break
        except Exception:
            continue
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
    """アップロード複数ファイルのキャッシュキー生成"""
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
    for col in ["コード4","銘柄コード","コード"]:
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
    cands = preferred or ["約定日","約定日時","約定日付","日時","日付"]
    for c in cands:
        if c in df.columns: return c
    for c in df.columns:
        if re.search(r"(約定)?(日付|日時)", str(c)): return c
    return None

def pick_time_col(df: pd.DataFrame, preferred=None) -> str | None:
    if df is None or df.empty: return None
    cands = preferred or ["約定時刻","約定時間","時刻","時間","約定時刻(JST)","約定時間(日本)","時間(JST)"]
    for c in cands:
        if c in df.columns: return c
    for c in df.columns:
        if re.search(r"(約定)?(時刻|時間)", str(c)): return c
    return None

def _contains_time_string(s: pd.Series) -> pd.Series:
    ss = s.astype(str)
    has_hms = ss.str.contains(r"\d{1,2}[:：]\d{1,2}")
    has_num = ss.str.contains(r"\b\d{3,6}\b")
    has_jp  = ss.str.contains(r"\d{1,2}時\d{1,2}分")
    return has_hms | has_num | has_jp

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
    mask_kanji = ss.str.match(r"^\d{1,2}時\d{1,2}分(\d{1,2}秒)?$")
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
    # 日付が文字列末尾に時刻数値を含むケースへの救済
    dt_str = df[date_col].astype(str)
    tail_num = dt_str.str.extract(r"(\d{3,6})\s*$")[0]
    mask_fill = td.isna() & tail_num.notna()
    if mask_fill.any():
        td.loc[mask_fill] = parse_time_only_to_timedelta(tail_num.loc[mask_fill])
    ts = d.dt.floor("D") + td
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        ts = ts.dt.tz_localize(TZ)
    except Exception:
        ts = ts.dt.tz_convert(TZ)
    return ts

def parse_datetime_from_dtcol(df: pd.DataFrame, dtcol: str) -> pd.Series:
    s = df[dtcol].astype(str).str.strip().str.replace("：",":", regex=False)
    date_part = s.str.extract(r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})")[0]
    d = pd.to_datetime(date_part, errors="coerce")
    t_hms  = s.str.extract(r"\b(\d{1,2}:\d{1,2}(?::\d{1,2})?)\b")[0]
    t_kan  = s.str.extract(r"\b(\d{1,2}時\d{1,2}分(?:\d{1,2}秒)?)\b")[0]
    t_tail = s.str.extract(r"\s(\d{3,6})\s*$")[0]
    t_str = t_hms.fillna(t_kan).fillna(t_tail).fillna("")
    td = parse_time_only_to_timedelta(t_str)
    ts = d.dt.floor("D") + td
    ts = pd.to_datetime(ts, errors="coerce")
    try:
        ts = ts.dt.tz_localize(TZ)
    except Exception:
        ts = ts.dt.tz_convert(TZ)
    return ts

def normalize_realized(df: pd.DataFrame) -> pd.DataFrame:
    """
    列名ゆる検出＋数値化。'約定日時'(TZ付き) と '約定日'(date) と '実現損益[円]' を作る。
    さらに '約定時刻あり'(bool) を付与し、時刻情報を持たない行を除外できるようにする。
    """
    if df is None or df.empty:
        return df
    d = clean_columns(df.copy())

    # 候補検出
    date_col = pick_dt_col(d)
    time_col = pick_time_col(d)

    # --- 約定日時の生成 ---
    if date_col and time_col:
        ts = combine_date_time_cols(d, date_col, time_col)
        has_time_raw = d[time_col].astype(str).str.strip().ne("")
    elif date_col and _contains_time_string(d[date_col]).any():
        ts = parse_datetime_from_dtcol(d, date_col)
        has_time_raw = _contains_time_string(d[date_col])
    elif date_col:
        ts = pd.to_datetime(d[date_col], errors="coerce", infer_datetime_format=True)
        try:
            ts = ts.dt.tz_localize(TZ)
        except Exception:
            ts = ts.dt.tz_convert(TZ)
        has_time_raw = pd.Series(False, index=d.index)
    else:
        ts = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
        has_time_raw = pd.Series(False, index=d.index)

    # TZ付与（再保証）
    if getattr(ts.dtype, "tz", None) is None:
        try:
            ts = pd.to_datetime(ts, errors="coerce").dt.tz_localize(TZ)
        except Exception:
            ts = pd.to_datetime(ts, errors="coerce").dt.tz_convert(TZ)

    # "約定時刻あり": 実際に時分秒が 00:00:00 以外
    time_nonzero = ts.notna() & ((ts.dt.hour + ts.dt.minute + ts.dt.second) > 0)
    d["約定日時"] = ts
    d["約定日"] = pd.to_datetime(ts, errors="coerce").dt.date
    d["約定時刻あり"] = (has_time_raw | time_nonzero).fillna(False)

    # 実現損益列を検出 → "実現損益[円]" に正規化
    pl_col = None
    for c in ["実現損益[円]","実現損益（円）","実現損益","損益[円]","損益額","損益"]:
        if c in d.columns:
            pl_col = c; break
    if pl_col is None:
        candidates = [c for c in d.columns if any(t in str(c).lower() for t in ["損益","pnl","profit","realized","pl"])]
        best, best_ratio = None, 0.0
        for c in candidates:
            s = to_numeric_jp(d[c])
            ratio = s.notna().mean()
            if ratio > best_ratio:
                best_ratio, best = ratio, c
        pl_col = best
    d["実現損益[円]"] = to_numeric_jp(d[pl_col]) if pl_col else pd.Series(dtype="float64")

    return normalize_symbol_cols(d)

def normalize_yakujyou(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    d = normalize_symbol_cols(df.copy())
    return d

def lr_from_realized_trade(val: str) -> str | None:
    if val is None or (isinstance(val, float) and pd.isna(val)): return None
    s = str(val).replace("　","").replace(" ","")
    if "売埋" in s or ("売" in s and "返済" in s): return "LONG"
    if "買埋" in s or ("買" in s and "返済" in s): return "SHORT"
    if "買建" in s or "売建" in s: return None
    sl = s.lower()
    if "sell" in sl and "close" in sl: return "LONG"
    if "buy" in sl and "close" in sl:  return "SHORT"
    return None

def side_to_io(val: str) -> str | None:
    if val is None or (isinstance(val,float) and np.isnan(val)): return None
    s = str(val).replace("　","").replace(" ","").lower()
    if "買建" in s or ("buy" in s and "close" not in s):  return "IN"
    if "売建" in s or ("sell" in s and "close" not in s): return "IN"
    if "売埋" in s or "売返済" in s or ("sell" in s and "close" in s): return "OUT"
    if "買埋" in s or "買返済" in s or ("buy" in s and "close" in s):  return "OUT"
    return None

def side_to_action(val: str) -> str | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).replace("　","").replace(" ","").lower()
    if "買建" in s: return "買建"
    if "売建" in s: return "売建"
    if ("売埋" in s) or ("売返済" in s): return "売埋"
    if ("買埋" in s) or ("買返済" in s): return "買埋"
    if "buy" in s and "close" not in s:  return "買建"
    if "sell" in s and "close" not in s: return "売建"
    if "sell" in s and "close" in s:     return "売埋"
    if "buy" in s and "close" in s:      return "買埋"
    return None

# ---- 約定（約定履歴）→ IN/OUT マーカー生成の補助
def build_exec_table_allperiod(yj_all: pd.DataFrame) -> pd.DataFrame:
    if yj_all is None or yj_all.empty:
        return pd.DataFrame(columns=["code_key","name_key","exec_time","price","io","action"])
    d = yj_all.copy()

    dtcol = pick_dt_col(d); tmcol = pick_time_col(d)
    if dtcol is None:
        return pd.DataFrame(columns=["code_key","name_key","exec_time","price","io","action"])

    def has_time_rowwise(row):
        val_dt = str(row.get(dtcol, "")).strip()
        val_tm = str(row.get(tmcol, "")).strip() if tmcol else ""
        if val_tm:
            return True
        if re.search(r"[:：]", val_dt): return True
        if re.search(r"\b\d{3,6}\b", val_dt): return True
        if re.search(r"時\d{1,2}分", val_dt): return True
        return False

    mask_time = d.apply(has_time_rowwise, axis=1)
    d = d.loc[mask_time].copy()
    if d.empty:
        return pd.DataFrame(columns=["code_key","name_key","exec_time","price","io","action"])

    if tmcol:
        exec_ts = combine_date_time_cols(d, dtcol, tmcol)
    else:
        exec_ts = parse_datetime_from_dtcol(d, dtcol)
    d["exec_time"] = exec_ts

    # 価格
    PRICE_CANDS = ["約定単価(円)","約定単価（円）","約定価格","価格","約定単価"]
    def select_price_series(df: pd.DataFrame) -> pd.Series | None:
        for c in PRICE_CANDS:
            if c in df.columns: return to_numeric_jp(df[c])
        pat = re.compile(r"(約定)?.*(単価|価格)")
        best,nn = None,-1
        for c in df.columns:
            if pat.search(str(c)):
                s = to_numeric_jp(df[c]); k = s.notna().sum()
                if k>nn: best,nn = c,k
        return to_numeric_jp(df[best]) if best else None

    price_series = select_price_series(d)
    d["price"] = price_series if price_series is not None else np.nan

    d = normalize_symbol_cols(d)

    side_col = None
    for c in ["売買","売買区分","売買種別","Side","取引"]:
        if c in d.columns: side_col = c; break
    if side_col is None:
        for c in d.columns:
            if "売買" in c or "side" in c.lower() or "取引" in c:
                side_col = c; break
    d["io"] = d[side_col].map(side_to_io) if side_col else None
    d["action"] = d[side_col].map(side_to_action) if side_col else None

    cols = ["code_key","name_key","exec_time","price","io","action"]
    out = d[cols].dropna(subset=["exec_time"]).copy()
    out["code_key"] = out["code_key"].astype("string").str.upper().str.strip().replace({"":"<NA>","NAN":"<NA>"}).replace("<NA>", pd.NA)
    return out

def build_trade_table_for_display(yakujyou_all: pd.DataFrame, sel_date: date, code4: str) -> pd.DataFrame:
    if yakujyou_all is None or yakujyou_all.empty:
        return pd.DataFrame(columns=["約定時間","売買","約定数","約定単価"])

    d = normalize_symbol_cols(yakujyou_all.copy())
    dtcol = pick_dt_col(d)
    tmcol = pick_time_col(d)
    if dtcol is None:
        return pd.DataFrame(columns=["約定時間","売買","約定数","約定単価"])

    if tmcol:
        ts = combine_date_time_cols(d, dtcol, tmcol)
    else:
        ts = parse_datetime_from_dtcol(d, dtcol)
    d["__exec_time__"] = ts

    d["code_key"] = d["code_key"].astype("string").str.upper().str.strip()
    mask = (d["code_key"] == str(code4).upper()) & (d["__exec_time__"].dt.tz_convert(TZ).dt.date == sel_date)
    d = d.loc[mask].copy()
    if d.empty:
        return pd.DataFrame(columns=["約定時間","売買","約定数","約定単価"])

    side_col = None
    for c in ["売買","売買区分","売買種別","Side","取引"]:
        if c in d.columns: side_col = c; break
    if side_col is None:
        for c in d.columns:
            if "売買" in c or "side" in c.lower() or "取引" in c:
                side_col = c; break

    def _side_to_action(val: str) -> str | None:
        if val is None or (isinstance(val, float) and np.isnan(val)): return None
        s = str(val).replace("　","").replace(" ","").lower()
        if "買建" in s: return "買建"
        if "売建" in s: return "売建"
        if ("売埋" in s) or ("売返済" in s): return "売埋"
        if ("買埋" in s) or ("買返済" in s): return "買埋"
        if "buy" in s and "close" not in s:  return "買建"
        if "sell" in s and "close" not in s: return "売建"
        if "sell" in s and "close" in s:     return "売埋"
        if "buy" in s and "close" in s:      return "買埋"
        return None

    if side_col:
        side_series = d[side_col].astype(str)
        action_series = d[side_col].map(_side_to_action)
        side_series = side_series.where(side_series.str.strip().ne(""), action_series)
    else:
        side_series = d.get("action", None)

    def select_qty_series(df: pd.DataFrame) -> pd.Series | None:
        cand_exact = [
            "約定数量", "約定数量(株)", "約定数量（株）", "約定株数",
            "出来数量", "数量", "株数", "出来高",
            "約定数量(口)", "口数"
        ]
        for c in cand_exact:
            if c in df.columns:
                return to_numeric_jp(df[c])
        best, nn = None, -1
        for c in df.columns:
            cname = str(c).replace(" ", "")
            if any(k in cname for k in ["数量", "株数", "口数", "出来高"]):
                s = to_numeric_jp(df[c])
                k = s.notna().sum()
                if k > nn:
                    best, nn = c, k
        return to_numeric_jp(df[best]) if best else None

    def select_price_series(df: pd.DataFrame) -> pd.Series | None:
        PRICE_CANDS = ["約定単価(円)","約定単価（円）","約定価格","価格","約定単価"]
        for c in PRICE_CANDS:
            if c in df.columns: return to_numeric_jp(df[c])
        pat = re.compile(r"(約定)?.*(単価|価格)")
        best,nn = None,-1
        for c in df.columns:
            if pat.search(str(c)):
                s = to_numeric_jp(df[c]); k = s.notna().sum()
                if k>nn: best,nn = c,k
        return to_numeric_jp(df[best]) if best else None

    qty_series   = select_qty_series(d)
    price_series = select_price_series(d)

    out = pd.DataFrame({
        "約定時間": d["__exec_time__"].dt.tz_convert(TZ).dt.strftime("%H:%M:%S"),
        "売買": side_series,
        "約定数": qty_series,
        "約定単価": price_series,
    })
    out = out.sort_values("約定時間").reset_index(drop=True)
    return out

# =========================================================
# 3分足ロード（アップロード群から作る）
# =========================================================
@st.cache_data(show_spinner=False)
def load_ohlc_map_from_uploads(files, sig: str):
    """
    time, open, high, low, close を必須とし、任意で VWAP, MA1, MA2, MA3 を取り込む。
    キーはアップロード時のファイル名（拡張子除く）。
    """
    ohlc_map = {}
    if not files: return ohlc_map
    for f in files:
        df = read_table_from_upload(f.name, f.getvalue())
        if df.empty: continue

        col_rename_map, found_cols = {}, {}
        CANDIDATES = {
            "time": ["time", "日時"],
            "open": ["open", "始値"],
            "high": ["high", "高値"],
            "low":  ["low",  "安値"],
            "close":["close","終値"],
        }
        original_cols = {c.lower(): c for c in df.columns}
        for std, cands in CANDIDATES.items():
            for cand in cands:
                if cand in original_cols:
                    col_rename_map[original_cols[cand]] = std
                    found_cols[std] = True
                    break
        if len(found_cols) < len(CANDIDATES):
            continue

        df = df.rename(columns=col_rename_map)

        t = pd.to_datetime(df["time"], errors="coerce", infer_datetime_format=True)
        if getattr(t.dtype, "tz", None) is None: t = t.dt.tz_localize(TZ)
        else: t = t.dt.tz_convert(TZ)
        df = df.copy()
        df["time"] = t

        def pick_one(df, names):
            for n in names:
                if n in df.columns: return df[n]
            return None
        vwap = pick_one(df, ["VWAP","vwap","Vwap"])
        ma1  = pick_one(df, ["MA1","ma1","Ma1"])
        ma2  = pick_one(df, ["MA2","ma2","Ma2"])
        ma3  = pick_one(df, ["MA3","ma3","Ma3"])
        if vwap is not None: df["VWAP"] = pd.to_numeric(vwap, errors="coerce")
        if ma1  is not None: df["MA1"]  = pd.to_numeric(ma1,  errors="coerce")
        if ma2  is not None: df["MA2"]  = pd.to_numeric(ma2,  errors="coerce")
        if ma3  is not None: df["MA3"]  = pd.to_numeric(ma3,  errors="coerce")

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
        t = df["time"]
        t = t.dt.tz_localize(TZ) if getattr(t.dtype,"tz",None) is None else t.dt.tz_convert(TZ)
        if t.notna().any():
            mins.append(t.min().date()); maxs.append(t.max().date())
    if not mins or not maxs: return None, None
    return min(mins), max(maxs)

def market_time(sel_date: date, start_str="09:00", end_str="15:30"):
    return pd.Timestamp(f"{sel_date} {start_str}", tz=TZ), pd.Timestamp(f"{sel_date} {end_str}", tz=TZ)

def pick_best_ohlc_key_for_date(code4: str, ohlc_code_index: dict, ohlc_map: dict, sel_date) -> tuple[str|None, pd.DataFrame|None, str]:
    keys = ohlc_code_index.get(str(code4).upper(), [])
    if not keys: return None, None, "該当キーなし"
    start_dt, end_dt = market_time(sel_date)
    candidates, dbg = [], []
    for k in keys:
        df = ohlc_map.get(k)
        if df is None or df.empty or "time" not in df.columns:
            dbg.append(f"{k}: データなし/列不備"); continue
        t = df["time"]
        t = t.dt.tz_localize(TZ) if getattr(t.dtype,"tz",None) is None else t.dt.tz_convert(TZ)
        tmin, tmax = t.min(), t.max()
        dbg.append(f"{k}: [{tmin} .. {tmax}]")
        in_win = (t>=start_dt)&(t<=end_dt)
        same_day = (t.dt.date==sel_date)
        day_dist = float("inf")
        if t.notna().any():
            sel_start = pd.Timestamp(sel_date, tz=TZ)
            if (t.dt.date==sel_date).any(): day_dist = 0.0
            elif tmax < sel_start: day_dist = (sel_start - tmax).total_seconds()
            elif tmin > sel_start: day_dist = (tmin - sel_start).total_seconds()
            else: day_dist = 0.0
        score = (0 if in_win.any() else (1 if same_day.any() else 2), 0 if same_day.any() else 1, day_dist)
        candidates.append((score,k,df))
    if not candidates: return None, None, "候補なし"
    candidates.sort(key=lambda x:x[0])
    best = candidates[0]
    return best[1], best[2], " / ".join(dbg)+f" => PICK: {best[1]}"

def download_button_df(df, label, filename):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

def compute_max_drawdown(series: pd.Series) -> float:
    if series is None:
        return np.nan
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return np.nan
    arr = s.to_numpy(dtype=float)
    peak, max_dd = -np.inf, 0.0
    for x in arr:
        peak = max(peak, x)
        max_dd = max(max_dd, peak - x)
    return float(max_dd)

# ======== タイムフレーム検出＆リサンプリング ========
def _detect_base_interval_minutes(ts: pd.Series) -> int | None:
    if ts is None or ts.empty: return None
    t = pd.to_datetime(ts, errors="coerce")
    if getattr(t.dtype, "tz", None) is None:
        t = t.dt.tz_localize(TZ)
    diffs = t.sort_values().diff().dropna().dt.total_seconds() / 60
    if diffs.empty: return None
    return int(round(diffs.median()))

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    need_cols = {"time","open","high","low","close"}
    if df.empty or not need_cols.issubset(df.columns): 
        return df.copy()

    d = df.copy()
    t = pd.to_datetime(d["time"], errors="coerce")
    if getattr(t.dtype, "tz", None) is None:
        t = t.dt.tz_localize(TZ)
    d["time"] = t
    d = d.set_index("time")

    agg = {"open":"first","high":"max","low":"min","close":"last"}
    if "volume" in d.columns:
        agg["volume"] = "sum"

    out = d.resample(rule, origin="start_day").agg(agg).dropna(subset=["open","high","low","close"])

    if "VWAP" in d.columns:
        if "volume" in d.columns:
            wnum = (d["VWAP"] * d["volume"]).resample(rule, origin="start_day").sum(min_count=1)
            wden = d["volume"].resample(rule, origin="start_day").sum(min_count=1)
            out["VWAP"] = (wnum / wden)
        else:
            out["VWAP"] = d["VWAP"].resample(rule, origin="start_day").mean()

    for ma in ["MA1","MA2","MA3"]:
        if ma in d.columns:
            out[ma] = pd.to_numeric(d[ma], errors="coerce").resample(rule, origin="start_day").mean()

    out = out.reset_index()
    return out

# ======== 時間系ヘルパ（“秒”で比較する方式に統一） ========
BASE_ANCHOR_DATE = datetime(2000,1,1, tzinfo=TZ)

def to_time_of_day_ts(dt_series: pd.Series) -> pd.Series:
    dt_local = dt_series.dt.tz_convert(TZ)
    return pd.to_datetime([datetime(2000,1,1,h, m, s, tzinfo=TZ)
                           for h,m,s in zip(dt_local.dt.hour, dt_local.dt.minute, dt_local.dt.second)])

def session_of(dt_series: pd.Series) -> pd.Series:
    """前場/後場/その他（Na）を判定（秒で比較）"""
    dt_local = dt_series.dt.tz_convert(TZ)
    sec = dt_local.dt.hour*3600 + dt_local.dt.minute*60 + dt_local.dt.second
    out = pd.Series(pd.NA, index=dt_series.index, dtype="object")
    out[(sec >= MORNING_START_SEC) & (sec <= MORNING_END_SEC)]  = "前場"
    out[(sec >= AFTERNOON_START_SEC) & (sec <= AFTERNOON_END_SEC)] = "後場"
    return out

def clip_market_time(dt_series: pd.Series) -> pd.Series:
    """9:00～15:30 に入っている行だけ True（秒で比較）"""
    dt_local = dt_series.dt.tz_convert(TZ)
    sec = dt_local.dt.hour*3600 + dt_local.dt.minute*60 + dt_local.dt.second
    return (sec >= MORNING_START_SEC) & (sec <= AFTERNOON_END_SEC)

# =========================================================
# サイドバー：データアップロード & フィルタ
# =========================================================
st.sidebar.header("① データアップロード（複数ファイルまとめてOK）")
realized_files = st.sidebar.file_uploader("実現損益 CSV/Excel", type=["csv","txt","xlsx"], accept_multiple_files=True)
yakujyou_files = st.sidebar.file_uploader("約定履歴 CSV/Excel", type=["csv","txt","xlsx"], accept_multiple_files=True)
ohlc_files     = st.sidebar.file_uploader("3分足 OHLC CSV/Excel（ファイル名末尾に _7974 などのコードを含める）", type=["csv","txt","xlsx"], accept_multiple_files=True)

sig_realized = files_signature(realized_files)
sig_yakujyou = files_signature(yakujyou_files)
sig_ohlc     = files_signature(ohlc_files)

yakujyou_all = concat_uploaded_tables(yakujyou_files, sig_yakujyou)
yakujyou_all = normalize_yakujyou(clean_columns(yakujyou_all))

realized = concat_uploaded_tables(realized_files, sig_realized)
realized = normalize_realized(clean_columns(realized))

ohlc_map = load_ohlc_map_from_uploads(ohlc_files, sig_ohlc)
CODE_TO_NAME = build_code_to_name_map(realized, yakujyou_all)

# 信用区分フィルタ
st.sidebar.header("② トレード種別フィルタ")
trade_type = st.sidebar.radio("対象", ["全体","デイトレード（一般）","スイングトレード（制度）"], index=0)

def apply_trade_type_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "信用区分" not in df.columns: return df
    if trade_type.startswith("全体"): return df
    if "デイトレ" in trade_type: return df[df["信用区分"]=="一般"]
    if "スイング" in trade_type: return df[df["信用区分"]=="制度"]
    return df

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
    if df.empty or dt_col not in df.columns: return df
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
    mask = (dt.dt.date>=s)&(dt.dt.date<=e)
    return df.loc[mask]

realized_f = apply_trade_type_filter(filter_by_span(realized, "約定日"))

# =========================================================
# KPI
# =========================================================
st.subheader("KPI")

if not realized_f.empty and "実現損益[円]" in realized_f.columns:
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

c4,c5,c6 = st.columns(3)
with c4:
    win_rate = (pl>0).mean()*100 if pl is not None and not pl.empty else np.nan
    st.metric("勝率（全体）", f"{win_rate:.1f}%" if pd.notna(win_rate) else "—")

wr_long = wr_short = np.nan
if not realized_f.empty and ("取引" in realized_f.columns):
    rj = realized_f.copy()
    rj["PL"] = to_numeric_jp(rj["実現損益[円]"])
    rj["LR"] = rj["取引"].map(lr_from_realized_trade)
    pl_long  = rj.loc[rj["LR"]=="LONG","PL"]
    pl_short = rj.loc[rj["LR"]=="SHORT","PL"]
    wr_long  = (pl_long>0).mean()*100  if pl_long.notna().any()  else np.nan
    wr_short = (pl_short>0).mean()*100 if pl_short.notna().any() else np.nan

with c5: st.metric("勝率（ロング）", f"{wr_long:.1f}%" if pd.notna(wr_long) else "—")
with c6: st.metric("勝率（ショート）", f"{wr_short:.1f}%" if pd.notna(wr_short) else "—")

# 最大DD
c7,_c8,_c9 = st.columns(3)
with c7:
    if not realized_f.empty and "約定日" in realized_f.columns and "実現損益[円]" in realized_f.columns:
        tmp = realized_f.copy()
        tmp["日"] = pd.to_datetime(tmp["約定日"], errors="coerce").dt.date
        tmp["実現損益[円]"] = to_numeric_jp(tmp["実現損益[円]"])
        seq = tmp.groupby("日", as_index=False)["実現損益[円]"].sum().sort_values("日")
        seq["累計"] = pd.to_numeric(seq["実現損益[円]"], errors="coerce").cumsum()
        dd = compute_max_drawdown(seq["累計"])
        st.metric("最大ドローダウン", f"{int(round(dd)):,} 円" if pd.notna(dd) else "—")
    else:
        st.metric("最大ドローダウン", "—")

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
    if realized_f.empty:
        st.info("実現損益データが必要です。")
    else:
        r = realized_f.copy()
        dts = pd.to_datetime(r["約定日"], errors="coerce")
        r["実現損益[円]"] = to_numeric_jp(r["実現損益[円]"])
        r["日"] = dts.dt.date
        r["週"] = (dts - pd.to_timedelta(dts.dt.weekday, unit="D")).dt.date
        r["月"] = dts.dt.to_period("M").dt.to_timestamp().dt.date
        r["年"] = dts.dt.to_period("Y").dt.to_timestamp().dt.date
        for label,col in [("日別","日"),("週別","週"),("月別","月"),("年別","年")]:
            g = r.groupby(col, as_index=False)["実現損益[円]"].sum().sort_values(col)
            st.write(f"**{label}**")
            st.dataframe(g, use_container_width=True, hide_index=True)
            download_button_df(g, f"⬇ CSVダウンロード（{label}）", f"{col}_pl.csv")
            fig_bar = go.Figure([go.Bar(x=g[col], y=g["実現損益[円]"], name=f"{label} 実現損益")])
            fig_bar.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=300, xaxis_title=label, yaxis_title="実現損益[円]")
            st.plotly_chart(fig_bar, use_container_width=True)

# ---- 1b) 時間別
with tab1b:
    st.markdown("### 実現損益（時間別・1時間ごと）")
    if realized_f.empty:
        st.info("実現損益データが必要です。")
    else:
        d = realized_f.copy()

        # --- 時刻がある行だけ採用 ---
        if "約定日時" not in d.columns:
            st.warning("実現損益データに '約定日時' 列がありません。サイドバーのエンコーディングや列検出をご確認ください。")
        else:
            dt = pd.to_datetime(d["約定日時"], errors="coerce")
            try:
                dt = dt.dt.tz_convert(TZ)
            except Exception:
                dt = dt.dt.tz_localize(TZ)

            if "約定時刻あり" in d.columns:
                d = d.loc[d["約定時刻あり"]].copy()
                dt = dt.loc[d.index]
            else:
                mask_has_time = dt.notna() & ((dt.dt.hour + dt.dt.minute + dt.dt.second) > 0)
                d = d.loc[mask_has_time].copy()
                dt = dt.loc[d.index]

            # 市場時間でクリップ（9:00〜15:30）— 秒で比較
            dt_local = dt.dt.tz_convert(TZ)
            sec = dt_local.dt.hour*3600 + dt_local.dt.minute*60 + dt_local.dt.second
            mask_mkt = (sec >= MORNING_START_SEC) & (sec <= AFTERNOON_END_SEC)
            d = d.loc[mask_mkt].copy()
            dt = dt.loc[d.index]

            if d.empty:
                st.info("時刻付きのレコードがありません（または市場時間外のみでした）。")
            else:
                # 指標列
                d["PL"] = to_numeric_jp(d["実現損益[円]"])
                d["win"] = d["PL"] > 0
                d["LR"]  = d["取引"].map(lr_from_realized_trade) if "取引" in d.columns else pd.NA

                # 1時間ビン（x軸は固定9:00〜15:30の擬似日付 2000-01-01 を使用）
                hour_floor = dt.dt.floor("H")
                hour_x = pd.to_datetime([datetime(2000,1,1,h.hour,0,0, tzinfo=TZ) for h in hour_floor])
                d["hour_x"] = hour_x

                x_range = [datetime(2000,1,1,9,0, tzinfo=TZ), datetime(2000,1,1,15,30, tzinfo=TZ)]
                x_ticks = pd.date_range(x_range[0], x_range[1], freq="60min", inclusive="both")
                base = pd.DataFrame({"hour_x": x_ticks})

                by = d.groupby("hour_x", as_index=False).agg(
                    収支=("PL","sum"),
                    取引回数=("PL","count"),
                    勝率=("win","mean"),
                    平均損益=("PL","mean")
                )
                if d["LR"].notna().any():
                    gL = d[d["LR"]=="LONG"].groupby("hour_x")["win"].mean().rename("勝率_ロング")
                    gS = d[d["LR"]=="SHORT"].groupby("hour_x")["win"].mean().rename("勝率_ショート")
                    by = base.merge(by, on="hour_x", how="left").merge(gL, how="left", on="hour_x").merge(gS, how="left", on="hour_x")
                else:
                    by = base.merge(by, on="hour_x", how="left")
                    by["勝率_ロング"] = np.nan
                    by["勝率_ショート"] = np.nan

                disp = by.copy()
                disp["時間"] = disp["hour_x"].dt.strftime("%H:%M")
                disp["勝率"] = (disp["勝率"]*100).round(1)
                disp["勝率_ロング"] = (disp["勝率_ロング"]*100).round(1)
                disp["勝率_ショート"] = (disp["勝率_ショート"]*100).round(1)
                st.dataframe(disp[["時間","収支","取引回数","勝率","勝率_ロング","勝率_ショート","平均損益"]],
                             use_container_width=True, hide_index=True)
                download_button_df(disp, "⬇ CSVダウンロード（時間別）", "hourly_stats.csv")

                # グラフ
                fig_h_pl = go.Figure([go.Bar(x=by["hour_x"], y=by["収支"], name="収支（合計）")])
                fig_h_pl.update_layout(title="時間別 収支（合計）", xaxis_title="時間", yaxis_title="円",
                                       margin=dict(l=10,r=10,t=30,b=10), height=300,
                                       xaxis=dict(tickformat="%H:%M", range=x_range))
                st.plotly_chart(fig_h_pl, use_container_width=True)

                fig_h_wr = go.Figure([go.Scatter(x=by["hour_x"], y=by["勝率"]*100, mode="lines+markers", name="勝率")])
                fig_h_wr.update_layout(title="時間別 勝率（全体）", xaxis_title="時間", yaxis_title="勝率（%）",
                                       margin=dict(l=10,r=10,t=30,b=10), height=300,
                                       yaxis=dict(range=[0,100]),
                                       xaxis=dict(tickformat="%H:%M", range=x_range))
                st.plotly_chart(fig_h_wr, use_container_width=True)

                fig_h_lr = go.Figure()
                fig_h_lr.add_trace(go.Scatter(x=by["hour_x"], y=by["勝率_ロング"]*100, mode="lines+markers", name="勝率（ロング）"))
                fig_h_lr.add_trace(go.Scatter(x=by["hour_x"], y=by["勝率_ショート"]*100, mode="lines+markers", name="勝率（ショート）"))
                fig_h_lr.update_layout(title="時間別 勝率（ロング/ショート）", xaxis_title="時間", yaxis_title="勝率（%）",
                                       margin=dict(l=10,r=10,t=30,b=10), height=300,
                                       yaxis=dict(range=[0,100]),
                                       xaxis=dict(tickformat="%H:%M", range=x_range))
                st.plotly_chart(fig_h_lr, use_container_width=True)

                fig_h_cnt = go.Figure([go.Bar(x=by["hour_x"], y=by["取引回数"], name="取引回数")])
                fig_h_cnt.update_layout(title="時間別 取引回数", xaxis_title="時間", yaxis_title="回",
                                        margin=dict(l=10,r=10,t=30,b=10), height=300,
                                        xaxis=dict(tickformat="%H:%M", range=x_range))
                st.plotly_chart(fig_h_cnt, use_container_width=True)

                fig_h_avg = go.Figure([go.Bar(x=by["hour_x"], y=by["平均損益"], name="平均損益（/回）")])
                fig_h_avg.update_layout(title="時間別 平均損益（/回）", xaxis_title="時間", yaxis_title="円/回",
                                        margin=dict(l=10,r=10,t=30,b=10), height=300,
                                        xaxis=dict(tickformat="%H:%M", range=x_range))
                st.plotly_chart(fig_h_avg, use_container_width=True)

                # 前場 / 後場 比較
                st.markdown("### 前場 / 後場 比較")
                ses = session_of(dt.loc[d.index])
                d["セッション"] = ses
                cmp = d.dropna(subset=["セッション"]).groupby("セッション").agg(
                    収支=("PL","sum"),
                    取引回数=("PL","count"),
                    勝率=("win","mean"),
                    平均損益=("PL","mean")
                ).reset_index()
                cmp["勝率"] = (cmp["勝率"]*100).round(1)
                st.dataframe(cmp, use_container_width=True, hide_index=True)
                download_button_df(cmp, "⬇ CSVダウンロード（前場後場比較）", "am_pm_compare.csv")

                cc1, cc2 = st.columns(2)
                with cc1:
                    fig_cmp_pl = go.Figure([go.Bar(x=cmp["セッション"], y=cmp["収支"], name="収支")])
                    fig_cmp_pl.update_layout(title="前場/後場 収支", xaxis_title="", yaxis_title="円",
                                             margin=dict(l=10,r=10,t=30,b=10), height=300)
                    st.plotly_chart(fig_cmp_pl, use_container_width=True)
                with cc2:
                    fig_cmp_wr = go.Figure([go.Bar(x=cmp["セッション"], y=cmp["勝率"], name="勝率")])
                    fig_cmp_wr.update_layout(title="前場/後場 勝率", xaxis_title="", yaxis_title="勝率（%）",
                                             margin=dict(l=10,r=10,t=30,b=10), height=300,
                                             yaxis=dict(range=[0,100]))
                    st.plotly_chart(fig_cmp_wr, use_container_width=True)

                # 累積勝率の時間推移（5分ビン）
                st.markdown("### 累積勝率の時間推移（全期間・5分ビン）")
                five_bin = dt.loc[d.index].dt.floor("5min")
                x_five = pd.to_datetime([datetime(2000,1,1,t.hour,t.minute,0, tzinfo=TZ) for t in five_bin.dt.time])
                tmp = pd.DataFrame({"x": x_five, "win": d["win"].astype(float), "cnt": 1.0})
                grid = pd.DataFrame({"x": pd.date_range(datetime(2000,1,1,9,0, tzinfo=TZ),
                                                        datetime(2000,1,1,15,30, tzinfo=TZ),
                                                        freq="5min", inclusive="both")})
                agg5 = tmp.groupby("x").agg(win_sum=("win","sum"), cnt=("cnt","sum")).reset_index()
                grid = grid.merge(agg5, on="x", how="left").fillna(0.0)
                grid["cum_wr"] = np.where(grid["cnt"].cumsum()>0,
                                          grid["win_sum"].cumsum()/grid["cnt"].cumsum()*100.0, np.nan)

                fig_cum = go.Figure([go.Scatter(x=grid["x"], y=grid["cum_wr"], mode="lines", name="累積勝率")])
                fig_cum.update_layout(title="累積勝率（時間の経過とともに）", xaxis_title="時間", yaxis_title="勝率（%）",
                                      margin=dict(l=10,r=10,t=30,b=10), height=320,
                                      yaxis=dict(range=[0,100]),
                                      xaxis=dict(tickformat="%H:%M",
                                                 range=[datetime(2000,1,1,9,0, tzinfo=TZ),
                                                        datetime(2000,1,1,15,30, tzinfo=TZ)]))
                st.plotly_chart(fig_cum, use_container_width=True)

# ---- 2) 累計
with tab2:
    st.markdown("### 累計実現損益（選択期間内、日次ベース）")
    if realized_f.empty:
        st.info("実現損益データが必要です。")
    else:
        d = realized_f.copy()
        d["日"] = pd.to_datetime(d["約定日"]).dt.date
        d["実現損益[円]"] = to_numeric_jp(d["実現損益[円]"])
        seq = d.groupby("日", as_index=False)["実現損益[円]"].sum().sort_values("日")
        seq["累計"] = pd.to_numeric(seq["実現損益[円]"], errors="coerce").cumsum()
        seq_disp = seq.copy()
        seq_disp["実現損益[円]"] = seq_disp["実現損益[円]"].round(0).astype("Int64")
        seq_disp["累計"] = seq_disp["累計"].round(0).astype("Int64")
        st.dataframe(seq_disp, use_container_width=True, hide_index=True)
        download_button_df(seq_disp, "⬇ CSVダウンロード（累計・日次）", "cumulative_daily_pl.csv")
        left,right = st.columns(2)
        with left:
            fig_bar = go.Figure([go.Bar(x=seq["日"], y=seq["実現損益[円]"], name="日次 実現損益")])
            fig_bar.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=350, xaxis_title="日付", yaxis_title="実現損益[円]")
            st.plotly_chart(fig_bar, use_container_width=True)
        with right:
            fig_line = go.Figure([go.Scatter(x=seq["日"], y=seq["累計"], mode="lines", name="累計")])
            fig_line.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=350, xaxis_title="日付", yaxis_title="累計実現損益[円]")
            st.plotly_chart(fig_line, use_container_width=True)

# ---- 3) 個別銘柄
def per_symbol_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["銘柄コード","銘柄名","実現損益合計","取引回数","1回平均損益","勝率"])
    d = normalize_symbol_cols(df.copy())
    if "実現損益[円]" in d.columns: d["実現損益"] = to_numeric_jp(d["実現損益[円]"])
    else:
        cand = next((c for c in d.columns if ("実現" in str(c) and "損益" in str(c))), None)
        if cand is None: 
            return pd.DataFrame(columns=["銘柄コード","銘柄名","実現損益合計","取引回数","1回平均損益","勝率"])
        d["実現損益"] = to_numeric_jp(d[cand])
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
    if realized_f.empty:
        st.info("実現損益データが必要です。")
    else:
        sym = per_symbol_stats(realized_f)
        if not sym.empty:
            disp = sym.copy()
            if "1回平均損益" in disp.columns: disp["1回平均損益"] = disp["1回平均損益"].round(0).astype("Int64")
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
    if realized_f.empty:
        st.info("実現損益データが必要です。")
    else:
        d = normalize_symbol_cols(realized_f.copy())
        d["実現損益"] = to_numeric_jp(d["実現損益[円]"]) if "実現損益[円]" in d.columns else np.nan
        d["group_key"] = np.where(d["code_key"].notna()&(d["code_key"]!=""), d["code_key"], "NAMEONLY::"+d["name_key"].astype(str))
        by_symbol = d.groupby("group_key").agg({"実現損益":["count","sum","mean"]})
        by_symbol.columns = ["取引回数","実現損益合計","1回平均損益"]
        rep_name = d.groupby("group_key").apply(representative_name).rename("銘柄名")
        code_col = d.groupby("group_key")["code_key"].first().rename("銘柄コード")
        out = by_symbol.join(rep_name).join(code_col).reset_index(drop=True)
        out["銘柄コード"] = out["銘柄コード"].fillna("—")
        if "1回平均損益" in out.columns: out["1回平均損益"] = out["1回平均損益"].round(0).astype("Int64")
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

# ---- 5) 3分足 IN/OUT + VWAP/MA（＋日経先物・日経平均）
with tab5:
    st.markdown("### 個別銘柄の3分足 + IN/OUT（当日指定｜時刻付き約定のみ）＋ 指標（VWAP/MA）")

    exec_all = build_exec_table_allperiod(yakujyou_all) if not yakujyou_all.empty else pd.DataFrame()

    ohlc_min_d, ohlc_max_d = ohlc_global_date_range(ohlc_map)
    allow_ohlc_only = st.checkbox("約定が無い日でも3分足を表示する", value=True)
    if allow_ohlc_only and (ohlc_min_d is not None) and (ohlc_max_d is not None):
        default_d = ohlc_max_d
        sel_date = st.date_input("日付を選択", value=default_d, min_value=ohlc_min_d, max_value=ohlc_max_d)
    else:
        if not exec_all.empty:
            dates = sorted(exec_all["exec_time"].dt.date.dropna().unique().tolist())
            if dates:
                sel_date = st.date_input("日付を選択（IN/OUTがある日）", value=dates[-1], min_value=dates[0], max_value=dates[-1])
            else:
                sel_date = st.date_input("日付を選択", value=date.today())
        else:
            sel_date = st.date_input("日付を選択", value=date.today())

    ohlc_code_index = build_ohlc_code_index(ohlc_map)
    all_codes_in_ohlc = sorted(ohlc_code_index.keys())
    sel_options = []
    if not exec_all.empty:
        day_exec = exec_all[exec_all["exec_time"].dt.date == sel_date]
        sel_options = sorted(day_exec["code_key"].dropna().unique().tolist())
    if not sel_options and all_codes_in_ohlc:
        sel_options = all_codes_in_ohlc
    if not sel_options:
        st.warning("表示できる銘柄データ（約定履歴 or 3分足）がありません。")
        st.stop()

    CODE_TO_NAME = build_code_to_name_map(realized, yakujyou_all)
    selected_code = st.selectbox(
        "銘柄を選択",
        options=sel_options,
        index=0,
        format_func=lambda c: f"{c}｜{CODE_TO_NAME.get(str(c).upper(), '名称不明')}"
    )
    if not selected_code: st.stop()
    code4 = str(selected_code).upper()
    disp_nm = CODE_TO_NAME.get(code4, "名称不明")

    st.markdown("#### 約定履歴（当日）")
    trades_tbl = build_trade_table_for_display(yakujyou_all, sel_date, code4)
    if trades_tbl.empty:
        st.info("当日の約定履歴がありません。")
    else:
        st.dataframe(trades_tbl, use_container_width=True, hide_index=True)
        download_button_df(trades_tbl, "⬇ CSVダウンロード（約定履歴・当日）", f"trades_{code4}_{sel_date}.csv")

    best_key, ohlc, dbg = pick_best_ohlc_key_for_date(code4, ohlc_code_index, ohlc_map, sel_date)
    if best_key is None or ohlc is None or ohlc.empty:
        st.error(f"銘柄コード {code4} の3分足データが見つかりません。"
                 "\nヒント：3分足ファイル名の末尾に `_7974` のようなコードを含めてください。")
        st.stop()
    auto_key = best_key

    # タイムゾーン/日付範囲
    if getattr(ohlc["time"].dtype,"tz",None) is None:
        ohlc["time"] = pd.to_datetime(ohlc["time"], errors="coerce").dt.tz_localize(TZ)
    else:
        ohlc["time"] = ohlc["time"].dt.tz_convert(TZ)
    start_dt = pd.Timestamp(f"{sel_date} 09:00", tz=TZ)
    end_dt   = pd.Timestamp(f"{sel_date} 15:30", tz=TZ)
    ohlc_day = ohlc.loc[(ohlc["time"]>=start_dt) & (ohlc["time"]<=end_dt)].copy()
    if ohlc_day.empty:
        st.warning(f"{sel_date} の {code4} の3分足データがありません。")
        st.stop()

    # マーカー
    marker_groups = {}
    skipped_price = 0
    if not exec_all.empty:
        marks = exec_all[(exec_all["exec_time"].dt.date == sel_date) & (exec_all["code_key"]==code4)].copy()
        in_window = (marks["exec_time"]>=start_dt) & (marks["exec_time"]<=end_dt)
        marks = marks.loc[in_window].copy()
        if not marks.empty:
            ohlc_idx = (ohlc_day[["time","close"]]
                        .rename(columns={"time":"ohlc_time","close":"ohlc_close"})
                        .sort_values("ohlc_time"))
            mk = marks.sort_values("exec_time").copy()
            mk["exec_time_floor"] = mk["exec_time"]
            merged = pd.merge_asof(
                mk.sort_values("exec_time_floor"),
                ohlc_idx,
                left_on="exec_time_floor",
                right_on="ohlc_time",
                direction="nearest",
                tolerance=pd.Timedelta("6min")
            )
            merged["price"] = merged["price"].fillna(merged["ohlc_close"])
            skipped_price = merged["price"].isna().sum()
            merged = merged.dropna(subset=["price"])

            for act in ["買建","売建","売埋","買埋"]:
                df_act = merged[merged["action"]==act][["exec_time","price"]].copy()
                if not df_act.empty:
                    marker_groups[act] = df_act

    st.caption(
        f"自動選択: **{code4}｜{disp_nm}**（ファイル: {auto_key}）｜"
        "IN/OUTは時刻付き約定のみ。価格欠損はOHLC近傍で補完（±6分）"
    )

    # === 表示オプション ===
    left, mid, right = st.columns([2,2,3])
    with left:
        tf_label = st.selectbox(
            "タイムフレーム",
            ["そのまま","6分","9分","15分"],
            index=0,
            help="元データより細かい足は作れません（3分→6/9/15分などに集約）。"
        )
    with mid:
        show_breaks = st.checkbox(
            "取引時間外と昼休みを隠す（レンジブレイク）",
            value=True,
            help="土日・夜間（15:30〜翌9:00）・昼休み（11:30〜12:30）を非表示にします。"
        )
    with right:
        show_lines = st.multiselect(
            "ラインの表示",
            options=["VWAP","MA1","MA2","MA3"],
            default=[x for x in ["VWAP","MA1","MA2"] if x in ohlc_day.columns],
        )

    enlarge = st.checkbox("🔍 チャートを拡大表示", value=False)
    chart_h = LARGE_CHART_HEIGHT if enlarge else MAIN_CHART_HEIGHT

    # タイムフレーム適用
    ohlc_disp = ohlc_day.copy()
    base_min = _detect_base_interval_minutes(ohlc_disp["time"])
    tf_map = {"そのまま": None, "6分":"6min", "9分":"9min", "15分":"15min"}
    target_rule = tf_map[tf_label]
    if target_rule:
        tgt_min = int(target_rule.replace("min",""))
        if base_min is not None and tgt_min >= base_min:
            ohlc_disp = _resample_ohlc(ohlc_disp, target_rule)
        else:
            st.caption("⚠ 元データより細かい間隔は作れないため、そのまま表示しています。")

    # ==== 色設定（MA3=青）
    COLOR_VWAP = "#808080"    # グレー
    COLOR_MA1  = "#2ca02c"    # 緑
    COLOR_MA2  = "#ff7f0e"    # オレンジ
    COLOR_MA3  = "#1f77b4"    # 青

    # ==== 個別銘柄チャート（マーカー付き）
    fig = go.Figure()
    fig.add_candlestick(
        x=ohlc_disp["time"], open=ohlc_disp["open"], high=ohlc_disp["high"],
        low=ohlc_disp["low"], close=ohlc_disp["close"], name="3分足"
    )
    if "VWAP" in show_lines and "VWAP" in ohlc_disp.columns and ohlc_disp["VWAP"].notna().any():
        fig.add_trace(go.Scatter(x=ohlc_disp["time"], y=ohlc_disp["VWAP"], mode="lines",
                                 line=dict(color=COLOR_VWAP, width=2), name="VWAP"))
    if "MA1" in show_lines and "MA1" in ohlc_disp.columns and ohlc_disp["MA1"].notna().any():
        fig.add_trace(go.Scatter(x=ohlc_disp["time"], y=ohlc_disp["MA1"], mode="lines",
                                 line=dict(color=COLOR_MA1, width=1.8), name="MA1"))
    if "MA2" in show_lines and "MA2" in ohlc_disp.columns and ohlc_disp["MA2"].notna().any():
        fig.add_trace(go.Scatter(x=ohlc_disp["time"], y=ohlc_disp["MA2"], mode="lines",
                                 line=dict(color=COLOR_MA2, width=1.8), name="MA2"))
    if "MA3" in show_lines and "MA3" in ohlc_disp.columns and ohlc_disp["MA3"].notna().any():
        fig.add_trace(go.Scatter(x=ohlc_disp["time"], y=ohlc_disp["MA3"], mode="lines",
                                 line=dict(color=COLOR_MA3, width=1.8), name="MA3"))

    # IN/OUT／建埋マーカー
    COLOR_MAP = {"買建":"#ff69b4","売建":"#1f77b4","売埋":"#2ca02c","買埋":"#ff7f0e"}
    SYMBOL_MAP = {"買建":"triangle-up","売建":"triangle-up","売埋":"triangle-down","買埋":"triangle-down"}
    TEXT_POS   = {"買建":"top center","売建":"top center","売埋":"bottom center","買埋":"bottom center"}
    for act, df_act in marker_groups.items():
        fig.add_trace(go.Scatter(
            x=df_act["exec_time"], y=df_act["price"],
            mode="markers+text",
            text=[act]*len(df_act), textposition=TEXT_POS.get(act, "top center"),
            marker_symbol=SYMBOL_MAP.get(act, "circle"),
            marker_size=10,
            marker_color=COLOR_MAP.get(act, "#444"),
            name=act,
            hovertemplate="時刻=%{x|%H:%M:%S}<br>価格=%{y:.2f}<extra>"+act+"</extra>",
        ))

    fig.update_layout(
        title=f"{sel_date} - {disp_nm} ({code4}) 3分足",
        height=chart_h, xaxis_title="時刻", yaxis_title="価格",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(l=10,r=10,t=30,b=10),
        xaxis=dict(tickformat="%H:%M")
    )
    # 範囲＆ブレイク
    fig.add_vline(x=pd.Timestamp(f"{sel_date} 09:00", tz=TZ), line=dict(width=1, dash="dot", color="#999"))
    fig.add_vline(x=pd.Timestamp(f"{sel_date} 15:30", tz=TZ), line=dict(width=1, dash="dot", color="#999"))
    if show_breaks:
        fig.update_xaxes(rangebreaks=[
            dict(bounds=["sat","mon"]),
            dict(bounds=[15.5,9], pattern="hour"),
            dict(bounds=[11.5,12.5], pattern="hour"),
        ])
    fig.update_xaxes(range=[start_dt, end_dt])
    st.plotly_chart(fig, use_container_width=True)

    if skipped_price > 0:
        st.warning(f"価格を補完できずマーカーを表示できなかった約定: {skipped_price} 件（±6分に足が無い 等）")

    # ===== 日経先物 =====
    st.markdown("#### 日経先物（OSE_NK2251!）")
    def find_first_key_by_prefix(ohlc_map: dict, prefix: str, sel_date: date | None = None) -> str | None:
        if not ohlc_map: return None
        pfx = prefix.lower()
        cands = [k for k in ohlc_map.keys() if k.lower().startswith(pfx)]
        if not cands: return None
        def norm_t(s: pd.Series):
            return s.dt.tz_localize(TZ) if getattr(s.dtype,"tz",None) is None else s.dt.tz_convert(TZ)
        if sel_date is None:
            best = max(cands, key=lambda k: norm_t(ohlc_map[k]["time"]).max())
            return best
        s_dt, e_dt = market_time(sel_date)
        def score(k):
            t = norm_t(ohlc_map[k]["time"])
            if t.empty:
                return (3, float("inf"), float("-inf"))
            in_win   = ((t>=s_dt)&(t<=e_dt)).any()
            same_day = (t.dt.date==sel_date).any()
            sel_ts   = pd.Timestamp(sel_date, tz=TZ)
            tmin, tmax = t.min(), t.max()
            if tmax < sel_ts:
                dd = (sel_ts - tmax).total_seconds()
            elif tmin > sel_ts:
                dd = (tmin - sel_ts).total_seconds()
            else:
                dd = 0.0
            return (0 if in_win else (1 if same_day else 2), dd, -tmax.value)
        cands.sort(key=score)
        return cands[0]

    key_fut = find_first_key_by_prefix(ohlc_map, "OSE_NK2251!", sel_date)
    if key_fut:
        o = ohlc_map[key_fut].copy()
        if getattr(o["time"].dtype,"tz",None) is None: o["time"] = pd.to_datetime(o["time"], errors="coerce").dt.tz_localize(TZ)
        else: o["time"] = o["time"].dt.tz_convert(TZ)
        o_day = o.loc[(o["time"]>=start_dt)&(o["time"]<=end_dt)].copy()
        if not o_day.empty:
            o_disp = o_day.copy()
            base_min2 = _detect_base_interval_minutes(o_disp["time"])
            if target_rule:
                tgt_min = int(target_rule.replace("min",""))
                if base_min2 is not None and tgt_min >= base_min2:
                    o_disp = _resample_ohlc(o_disp, target_rule)

            fig2 = go.Figure()
            fig2.add_candlestick(x=o_disp["time"], open=o_disp["open"], high=o_disp["high"], low=o_disp["low"], close=o_disp["close"], name="3分足")
            if "VWAP" in show_lines and "VWAP" in o_disp.columns and o_disp["VWAP"].notna().any():
                fig2.add_trace(go.Scatter(x=o_disp["time"], y=o_disp["VWAP"], mode="lines", line=dict(color=COLOR_VWAP, width=2), name="VWAP"))
            if "MA1" in show_lines and "MA1" in o_disp.columns and o_disp["MA1"].notna().any():
                fig2.add_trace(go.Scatter(x=o_disp["time"], y=o_disp["MA1"], mode="lines", line=dict(color=COLOR_MA1, width=1.8), name="MA1"))
            if "MA2" in show_lines and "MA2" in o_disp.columns and o_disp["MA2"].notna().any():
                fig2.add_trace(go.Scatter(x=o_disp["time"], y=o_disp["MA2"], mode="lines", line=dict(color=COLOR_MA2, width=1.8), name="MA2"))
            if "MA3" in show_lines and "MA3" in o_disp.columns and o_disp["MA3"].notna().any():
                fig2.add_trace(go.Scatter(x=o_disp["time"], y=o_disp["MA3"], mode="lines", line=dict(color=COLOR_MA3, width=1.8), name="MA3"))
            fig2.update_layout(
                height=chart_h, xaxis_title="時刻", yaxis_title="価格",
                xaxis_rangeslider_visible=False, hovermode="x unified",
                margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(tickformat="%H:%M"),
                title=f"{sel_date} - OSE_NK2251!（ファイル: {key_fut}）"
            )
            if show_breaks:
                fig2.update_xaxes(rangebreaks=[
                    dict(bounds=["sat","mon"]),
                    dict(bounds=[15.5,9], pattern="hour"),
                    dict(bounds=[11.5,12.5], pattern="hour"),
                ])
            fig2.update_xaxes(range=[start_dt, end_dt])
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("当日の先物データが見つかりませんでした。")
    else:
        st.info("先物ファイル（プレフィックス 'OSE_NK2251!'）が見つかりません。")

    # ===== 日経平均 =====
    st.markdown("#### 日経平均（TVC_NI225）")
    key_cfd = find_first_key_by_prefix(ohlc_map, "TVC_NI225", sel_date)
    if key_cfd:
        o = ohlc_map[key_cfd].copy()
        if getattr(o["time"].dtype,"tz",None) is None: o["time"] = pd.to_datetime(o["time"], errors="coerce").dt.tz_localize(TZ)
        else: o["time"] = o["time"].dt.tz_convert(TZ)
        o_day = o.loc[(o["time"]>=start_dt)&(o["time"]<=end_dt)].copy()
        if not o_day.empty:
            o_disp = o_day.copy()
            base_min3 = _detect_base_interval_minutes(o_disp["time"])
            if target_rule:
                tgt_min = int(target_rule.replace("min",""))
                if base_min3 is not None and tgt_min >= base_min3:
                    o_disp = _resample_ohlc(o_disp, target_rule)

            fig3 = go.Figure()
            fig3.add_candlestick(x=o_disp["time"], open=o_disp["open"], high=o_disp["high"], low=o_disp["low"], close=o_disp["close"], name="3分足")
            if "VWAP" in show_lines and "VWAP" in o_disp.columns and o_disp["VWAP"].notna().any():
                fig3.add_trace(go.Scatter(x=o_disp["time"], y=o_disp["VWAP"], mode="lines", line=dict(color=COLOR_VWAP, width=2), name="VWAP"))
            if "MA1" in show_lines and "MA1" in o_disp.columns and o_disp["MA1"].notna().any():
                fig3.add_trace(go.Scatter(x=o_disp["time"], y=o_disp["MA1"], mode="lines", line=dict(color=COLOR_MA1, width=1.8), name="MA1"))
            if "MA2" in show_lines and "MA2" in o_disp.columns and o_disp["MA2"].notna().any():
                fig3.add_trace(go.Scatter(x=o_disp["time"], y=o_disp["MA2"], mode="lines", line=dict(color=COLOR_MA2, width=1.8), name="MA2"))
            if "MA3" in show_lines and "MA3" in o_disp.columns and o_disp["MA3"].notna().any():
                fig3.add_trace(go.Scatter(x=o_disp["time"], y=o_disp["MA3"], mode="lines", line=dict(color=COLOR_MA3, width=1.8), name="MA3"))
            fig3.update_layout(
                height=chart_h, xaxis_title="時刻", yaxis_title="価格",
                xaxis_rangeslider_visible=False, hovermode="x unified",
                margin=dict(l=10,r=10,t=10,b=10), xaxis=dict(tickformat="%H:%M"),
                title=f"{sel_date} - TVC_NI225（ファイル: {key_cfd}）"
            )
            if show_breaks:
                fig3.update_xaxes(rangebreaks=[
                    dict(bounds=["sat","mon"]),
                    dict(bounds=[15.5,9], pattern="hour"),
                    dict(bounds=[11.5,12.5], pattern="hour"),
                ])
            fig3.update_xaxes(range=[start_dt, end_dt])
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("当日の日経平均データが見つかりませんでした。")
    else:
        st.info("日経平均ファイル（プレフィックス 'TVC_NI225'）が見つかりません。")
