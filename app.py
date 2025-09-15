# -*- coding: utf-8 -*-
import io, re, hashlib
from io import StringIO
from collections import Counter
from datetime import date, timedelta

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
MAIN_CHART_HEIGHT = 600
LARGE_CHART_HEIGHT = 860

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
    """日本語CSVでよくある表記を数値化"""
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
# 日付/時刻列の検出
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
    cands = preferred or ["約定時刻","約定時間","時刻","時間"]
    for c in cands:
        if c in df.columns: return c
    for c in df.columns:
        if re.search(r"(約定)?(時刻|時間)", str(c)): return c
    return None

def _row_has_explicit_time(row_val: str) -> bool:
    if row_val is None: return False
    s = str(row_val).strip()
    if re.search(r"\d{1,2}:\d{1,2}(:\d{1,2})?", s): return True
    if re.search(r"\d{3,6}$", s): return True
    if re.search(r"\d{1,2}時\d{1,2}分", s): return True
    return False

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

# =========================================================
# 実現損益・約定の正規化
# =========================================================
def normalize_realized(df: pd.DataFrame) -> pd.DataFrame:
    """
    列名ゆる検出＋数値化。'約定日時'・'約定日'・'実現損益[円]' を作成。
    '__has_time__' で「時刻が明示されている行」を保持（時間別集計で使用）。
    """
    if df is None or df.empty:
        return df
    d = clean_columns(df.copy())

    date_col = pick_dt_col(d)
    time_col = pick_time_col(d)

    # 時刻が明示されているかを行単位で推定
    has_time = pd.Series(False, index=d.index)
    if time_col:
        has_time = has_time | d[time_col].astype(str).map(_row_has_explicit_time)
    if date_col:
        has_time = has_time | d[date_col].astype(str).map(_row_has_explicit_time)

    # 約定日時
    if date_col and time_col:
        d["約定日時"] = combine_date_time_cols(d, date_col, time_col)
    elif date_col:
        ts = pd.to_datetime(d[date_col], errors="coerce", infer_datetime_format=True)
        try:
            ts = ts.dt.tz_localize(TZ)
        except Exception:
            ts = ts.dt.tz_convert(TZ)
        d["約定日時"] = ts
    else:
        d["約定日時"] = pd.NaT

    d["約定日"] = pd.to_datetime(d["約定日時"], errors="coerce").dt.date

    # 実現損益列の正規化
    pl_col = None
    for c in ["実現損益[円]","実現損益（円）","実現損益","損益[円]","損益額","損益"]:
        if c in d.columns:
            pl_col = c; break
    if pl_col is None:
        candidates = [c for c in d.columns
                      if any(t in str(c).lower() for t in ["損益","pnl","profit","realized","pl"])]
        best, best_ratio = None, 0.0
        for c in candidates:
            s = to_numeric_jp(d[c])
            ratio = s.notna().mean()
            if ratio > best_ratio:
                best_ratio, best = ratio, c
        pl_col = best
    if pl_col:
        d["実現損益[円]"] = to_numeric_jp(d[pl_col])
    else:
        d["実現損益[円]"] = pd.Series(dtype="float64")

    d["__has_time__"] = has_time
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

# ---- 約定（約定履歴）→ IN/OUT マーカー生成（チャート用）
def pick_dt_col_yj(df: pd.DataFrame) -> str | None:
    return pick_dt_col(df)

def pick_time_col_yj(df: pd.DataFrame) -> str | None:
    return pick_time_col(df)

def build_exec_table_allperiod(yj_all: pd.DataFrame) -> pd.DataFrame:
    if yj_all is None or yj_all.empty:
        return pd.DataFrame(columns=["code_key","name_key","exec_time","price","io","action"])
    d = yj_all.copy()

    dtcol = pick_dt_col_yj(d); tmcol = pick_time_col_yj(d)
    if dtcol is None:
        return pd.DataFrame(columns=["code_key","name_key","exec_time","price","io","action"])

    def has_time_rowwise(row):
        val_dt = str(row.get(dtcol, "")).strip()
        val_tm = str(row.get(tmcol, "")).strip() if tmcol else ""
        if val_tm: return True
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
    dtcol = pick_dt_col_yj(d)
    tmcol = pick_time_col_yj(d)
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

    # 売買
    side_col = None
    for c in ["売買","売買区分","売買種別","Side","取引"]:
        if c in d.columns: side_col = c; break
    if side_col is None:
        for c in d.columns:
            if "売買" in c or "side" in c.lower() or "取引" in c:
                side_col = c; break

    def _side_to_action(val: str) -> str | None:
        return side_to_action(val)

    if side_col:
        side_series = d[side_col].astype(str)
        action_series = d[side_col].map(_side_to_action)
        side_series = side_series.where(side_series.str.strip().ne(""), action_series)
    else:
        side_series = d.get("action", None)

    # 数量/価格
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
                s = to_numeric_jp(df[c]); k = s.notna().sum()
                if k > nn: best, nn = c, k
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
    time, open, high, low, close を必須。任意で VWAP, MA1, MA2, MA3。
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
        df = df.copy(); df["time"] = t

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

def slice_window_with_fallback(df: pd.DataFrame, sel_date: date) -> tuple[pd.DataFrame, str]:
    note = ""
    if df.empty or "time" not in df.columns:
        return df.head(0), "該当日のデータが見つかりませんでした。"

    t = df["time"]
    if getattr(t.dtype,"tz",None) is None:
        t_local = pd.to_datetime(t, errors="coerce").dt.tz_localize(TZ)
    else:
        t_local = pd.to_datetime(t, errors="coerce").dt.tz_convert(TZ)
    base = df.copy(); base["time"] = t_local

    s,e = market_time(sel_date, "09:00", "15:30")
    out = base.loc[(base["time"]>=s)&(base["time"]<=e)]
    if not out.empty:
        return out.copy(), note

    if getattr(t.dtype,"tz",None) is None:
        t_utc = pd.to_datetime(t, errors="coerce").dt.tz_localize("UTC").dt.tz_convert(TZ)
        base2 = df.copy(); base2["time"] = t_utc
        out2 = base2.loc[(base2["time"]>=s)&(base2["time"]<=e)]
        if not out2.empty:
            return out2.copy(), "指数/先物CSVの time を UTC として再解釈しました。"

    s2,e2 = market_time(sel_date, "08:45", "15:30")
    out = base.loc[(base["time"]>=s2)&(base["time"]<=e2)]
    if not out.empty:
        return out.copy(), "8:45–15:30 に拡大して抽出しました。"

    out = base.loc[base["time"].dt.date == sel_date]
    if not out.empty:
        return out.copy(), "当日の終日データで表示しました。"

    df_dates = base["time"].dt.date
    if df_dates.empty:
        return base.head(0), "該当日の近傍にもデータが見つかりませんでした。"
    diffs = (pd.to_datetime(df_dates) - pd.Timestamp(sel_date)).abs()
    idx = diffs.values.argmin()
    nearest_day = df_dates.iloc[idx]
    out = base.loc[df_dates==nearest_day]
    if not out.empty:
        return out.copy(), f"{sel_date} にデータが無いため {nearest_day} のデータを表示しています。"

    return base.head(0), "該当日のデータが見つかりませんでした。"

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

if not realized_f.empty and "実現損益[円
