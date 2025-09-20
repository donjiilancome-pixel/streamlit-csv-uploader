# app.py — Part 1/2
# -*- coding: utf-8 -*-
import io
import re
import sys
import math
import pytz
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
import streamlit as st

# =========================
# 基本設定・カラー
# =========================
TZ = pytz.timezone("Asia/Tokyo")
st.set_page_config(page_title="トレード可視化アプリ", layout="wide")

# 線色（ご要望どおり）
COLOR_VWAP = "#888888"   # グレー
COLOR_MA1  = "#2ca02c"   # 緑
COLOR_MA2  = "#ff7f0e"   # オレンジ
COLOR_MA3  = "#1f77b4"   # 青

MAIN_CHART_HEIGHT  = 420
LARGE_CHART_HEIGHT = 620

# =========================
# ユーティリティ
# =========================
@st.cache_data(show_spinner=False)
def try_read_csv(file, encodings=("utf-8-sig", "cp932", "shift_jis", "euc_jp")):
    last_err = None
    if hasattr(file, "read"):
        raw = file.read()
    else:
        raw = file
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception as e:
            last_err = e
            continue
    raise last_err

def to_numeric_jp(s):
    if s is None:
        return pd.Series(dtype=float)
    s = s.astype(str).str.replace(",", "", regex=False)\
                     .str.replace("\u3000", "", regex=False)\
                     .str.replace(" ", "", regex=False)\
                     .str.replace("−", "-", regex=False)\
                     .str.replace("ー", "-", regex=False)
    s = s.str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False)
    return pd.to_numeric(s, errors="coerce")

def _to_jst_series(s, index):
    if s is None:
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns, Asia/Tokyo]")
    out = pd.to_datetime(s, errors="coerce", utc=False)
    if getattr(out.dt, "tz", None) is None:
        out = out.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT")
    else:
        out = out.dt.tz_convert(TZ)
    return out

def pick_dt_col(df: pd.DataFrame):
    """日付/日時っぽい列候補から最初を返す"""
    if df is None or df.empty:
        return None
    cand = [c for c in df.columns if any(k in str(c) for k in ["約定日", "約定日時", "日時", "日付", "約定時間", "time", "date", "Date"])]
    return cand[0] if cand else None

def pick_dt_with_optional_time(df: pd.DataFrame):
    """
    df 内の「約定日 + (約定時間|時刻)」 or 「約定日時」から JST の Timestamp を作る。
    """
    if df is None or df.empty:
        return pd.Series(pd.NaT, index=df.index)
    cols = list(df.columns)
    # 先に「約定日時」系を試す
    dtcol = None
    for k in ["約定日時", "日時", "time", "Time", "約定時刻", "ExecutionTime", "取引時間"]:
        if k in cols:
            dtcol = k; break
    if dtcol:
        s = pd.to_datetime(df[dtcol], errors="coerce")
        if getattr(s.dt, "tz", None) is None:
            s = s.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT")
        else:
            s = s.dt.tz_convert(TZ)
        return s

    # 「約定日 + 約定時間/時刻」 を結合
    dcol = None
    for k in ["約定日", "日付", "Date"]:
        if k in cols:
            dcol = k; break
    tcol = None
    for k in ["約定時間", "時刻", "Time", "約定時刻"]:
        if k in cols:
            tcol = k; break
    if dcol and tcol:
        ds = pd.to_datetime(df[dcol], errors="coerce").dt.date.astype(str)
        ts = df[tcol].astype(str).str.replace(r"[^\d:]", "", regex=True)
        s = pd.to_datetime(ds + " " + ts, errors="coerce")
        s = s.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT")
        return s

    # 最後の砦：日付だけ
    if dcol:
        s = pd.to_datetime(df[dcol], errors="coerce")
        s = s.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT")
        return s
    return pd.Series(pd.NaT, index=df.index)

def extract_code_from_ohlc_key(key: str):
    """'TSE_9984, 3_xxxx.csv' などから 4桁コードを推定"""
    if not key:
        return None
    m = re.search(r"(\d{4})", key)
    return m.group(1) if m else None

def ohlc_global_date_range(ohlc_map: dict):
    tmin, tmax = None, None
    for _, df in ohlc_map.items():
        if df is None or df.empty or "time" not in df.columns:
            continue
        tt = _to_jst_series(df["time"], df.index)
        mn, mx = tt.min(), tt.max()
        if pd.isna(mn) or pd.isna(mx):
            continue
        if tmin is None or mn < tmin: tmin = mn
        if tmax is None or mx > tmax: tmax = mx
    if tmin is None: return None, None
    return tmin.date(), tmax.date()

def download_button_df(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# ===== 数値ソートを確実にする共通ヘルパー（新規 ①②③） =====
def _coerce_numeric_jp(val):
    """
    '90,077' や '−5,310'(U+2212) / '(1,234)' / 空白 などを数値へ。失敗時 NaN。
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "":
        return np.nan
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    s = s.replace(",", "").replace("\u3000", "").replace(" ", "")
    trans = str.maketrans("０１２３４５６７８９－", "0123456789-")
    s = s.translate(trans).replace("−", "-").replace("ー", "-")
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    if not m:
        return np.nan
    x = float(m.group(0))
    if neg:
        x = -x
    return x

def _numify_cols(df: pd.DataFrame, cols: list[str], round0=False) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(_coerce_numeric_jp)
            if round0:
                df[c] = pd.to_numeric(df[c], errors="coerce").round(0)
    return df

def show_numeric_table(df: pd.DataFrame, num_formats: dict[str, str], index=False, key=None):
    cfg = {}
    for col, fmt in num_formats.items():
        if col in df.columns:
            cfg[col] = st.column_config.NumberColumn(col, format=fmt)
    st.dataframe(df, use_container_width=True, hide_index=not index, column_config=cfg, key=key)

# =========================
# データ入力（アップロード）
# =========================
st.sidebar.header("📤 データアップロード")
up_realized = st.sidebar.file_uploader("実現損益（複数可）", type=["csv"], accept_multiple_files=True)
up_yakujyou = st.sidebar.file_uploader("約定履歴（複数可）", type=["csv"], accept_multiple_files=True)
up_ohlc     = st.sidebar.file_uploader("3分足OHLC（複数可）", type=["csv"], accept_multiple_files=True)

# =========================
# 実現損益・約定履歴の正規化
# =========================
@st.cache_data(show_spinner=False)
def normalize_realized(files):
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = try_read_csv(f)
            df["__src__"] = getattr(f, "name", "uploaded.csv")
            dfs.append(df)
        except Exception as e:
            st.warning(f"実現損益の読込失敗: {getattr(f,'name','(unknown)')} / {e}")
    if not dfs:
        return pd.DataFrame()
    d = pd.concat(dfs, ignore_index=True)

    # 金額列の推定
    pl_col = None
    cand = ["実現損益", "実現損益[円]", "損益", "損益（円）", "RealizedPnL", "PL", "profit", "金額"]
    for c in d.columns:
        if any(k in str(c) for k in cand):
            pl_col = c; break
    if pl_col is None:
        # どうしても無い場合はゼロ
        d["pl"] = 0.0
    else:
        d["pl"] = to_numeric_jp(d[pl_col])

    # 銘柄名・コード候補
    name_col = next((c for c in d.columns if any(k in str(c) for k in ["銘柄名", "名称", "Name"])), None)
    code_col = next((c for c in d.columns if any(k in str(c) for k in ["銘柄コード", "コード", "Symbol", "Code"])), None)
    if code_col: d["code_key"] = d[code_col].astype(str).str.extract(r"(\d{4})", expand=False)
    if name_col: d["name_key"] = d[name_col].astype(str)

    # 日時
    d["約定日時_final"] = pick_dt_with_optional_time(d)
    d["約定日_final"]   = pd.to_datetime(d["約定日時_final"], errors="coerce").dt.tz_convert(TZ).dt.date

    return d

@st.cache_data(show_spinner=False)
def normalize_yakujyou(files):
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = try_read_csv(f)
            df["__src__"] = getattr(f, "name", "uploaded.csv")
            dfs.append(df)
        except Exception as e:
            st.warning(f"約定履歴の読込失敗: {getattr(f,'name','(unknown)')} / {e}")
    if not dfs:
        return pd.DataFrame()
    d = pd.concat(dfs, ignore_index=True)

    code_col = next((c for c in d.columns if any(k in str(c) for k in ["銘柄コード", "コード", "Symbol", "Code"])), None)
    name_col = next((c for c in d.columns if any(k in str(c) for k in ["銘柄名", "名称", "Name"])), None)
    if code_col: d["code_key"] = d[code_col].astype(str).str.extract(r"(\d{4})", expand=False)
    if name_col: d["name_key"] = d[name_col].astype(str)

    d["約定日時"] = pick_dt_with_optional_time(d)
    d["約定日"]   = pd.to_datetime(d["約定日時"], errors="coerce").dt.tz_convert(TZ).dt.date
    return d

@st.cache_data(show_spinner=False)
def normalize_ohlc(files):
    """
    期待列: time, open, high, low, close[, volume][, VWAP][, MA1][, MA2][, MA3]
    """
    if not files:
        return {}
    out = {}
    for f in files:
        try:
            df = try_read_csv(f)
            # 列名ゆらぎ対応
            cols = {c.lower(): c for c in df.columns}
            def pick(*cand):
                for k in cand:
                    if k in cols: return cols[k]
                return None
            tcol = pick("time", "日時", "date", "datetime")
            ocol = pick("open", "始値")
            hcol = pick("high", "高値")
            lcol = pick("low",  "安値")
            ccol = pick("close","終値","close_price")

            if not all([tcol, ocol, hcol, lcol, ccol]):
                st.warning(f"OHLC列が不足: {getattr(f,'name','file')}")
                continue
            d = pd.DataFrame({
                "time": pd.to_datetime(df[tcol], errors="coerce"),
                "open": to_numeric_jp(df[ocol]),
                "high": to_numeric_jp(df[hcol]),
                "low":  to_numeric_jp(df[lcol]),
                "close":to_numeric_jp(df[ccol]),
            })
            # 既に tz 付きでなければ JST を付与
            if getattr(d["time"].dt, "tz", None) is None:
                d["time"] = d["time"].dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT")
            else:
                d["time"] = d["time"].dt.tz_convert(TZ)

            # 追加列（任意）
            for k in ["volume", "VWAP", "MA1", "MA2", "MA3"]:
                col = pick(k.lower(), k)
                if col and col in df.columns:
                    d[k] = to_numeric_jp(df[col])
            key = getattr(f, "name", f"ohlc_{len(out)+1}.csv")
            out[key] = d.dropna(subset=["time"]).sort_values("time")
        except Exception as e:
            st.warning(f"OHLCの読込失敗: {getattr(f,'name','(unknown)')} / {e}")
    return out

realized_all = normalize_realized(up_realized)
yakujyou_all = normalize_yakujyou(up_yakujyou)
ohlc_map     = normalize_ohlc(up_ohlc)

# 銘柄名辞書（3分足の想定名にも使用）
CODE_TO_NAME = {}
if not realized_all.empty and "code_key" in realized_all.columns and "name_key" in realized_all.columns:
    s = realized_all.dropna(subset=["code_key","name_key"]).drop_duplicates("code_key")[["code_key","name_key"]]
    CODE_TO_NAME.update(dict(zip(s["code_key"].str.upper(), s["name_key"])))
if not yakujyou_all.empty and "code_key" in yakujyou_all.columns and "name_key" in yakujyou_all.columns:
    s = yakujyou_all.dropna(subset=["code_key","name_key"]).drop_duplicates("code_key")[["code_key","name_key"]]
    CODE_TO_NAME.update(dict(zip(s["code_key"].str.upper(), s["name_key"])))

st.title("📈 投資管理アプリ（Streamlit）")

# =========================
# 1) 集計（期間別）
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["集計（期間別）","集計（時間別）","累計損益","個別/ランキング","3分足 IN/OUT + 指標"])

with tab1:
    st.subheader("実現損益（期間別集計）")
    if realized_all.empty:
        st.info("実現損益ファイルをアップロードしてください。")
    else:
        d = realized_all.copy()
        # 日別
        day = d.groupby("約定日_final", dropna=True)["pl"].sum().reset_index().rename(columns={"約定日_final":"日付", "pl":"実現損益[円]"})
        day = day.sort_values("日付")
        day = _numify_cols(day, ["実現損益[円]"], round0=True)
        st.markdown("**日別**")
        show_numeric_table(day, {"実現損益[円]":"%,.0f"}, key="period_day")
        # 週別
        d["week"] = pd.to_datetime(d["約定日_final"]).astype("datetime64[ns]")
        wk = d.dropna(subset=["week"]).copy()
        wk["week"] = wk["week"].dt.to_period("W").apply(lambda p: p.start_time.date())
        week = wk.groupby("week")["pl"].sum().reset_index().rename(columns={"week":"週", "pl":"実現損益[円]"})
        week = week.sort_values("週")
        week = _numify_cols(week, ["実現損益[円]"], round0=True)
        st.markdown("**週別**")
        show_numeric_table(week, {"実現損益[円]":"%,.0f"}, key="period_week")
        # 月別
        mo = d.copy()
        mo["month"] = pd.to_datetime(mo["約定日_final"]).astype("datetime64[ns]")
        mo = mo.dropna(subset=["month"])
        mo["month"] = mo["month"].dt.to_period("M").apply(lambda p: p.start_time.date())
        month = mo.groupby("month")["pl"].sum().reset_index().rename(columns={"month":"月", "pl":"実現損益[円]"})
        month = month.sort_values("月")
        month = _numify_cols(month, ["実現損益[円]"], round0=True)
        st.markdown("**月別**")
        show_numeric_table(month, {"実現損益[円]":"%,.0f"}, key="period_month")

        # 線グラフ（日別）
        if not day.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pd.to_datetime(day["日付"]), y=day["実現損益[円]"], mode="lines+markers", name="日別PL"))
            fig.update_layout(
                height=360, xaxis_rangeslider_visible=False,
                xaxis=dict(type="date", tickformat="%Y-%m-%d"),
                yaxis=dict(tickformat=",.0f"),
                margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================
# 2) 集計（時間別）
# =========================
with tab2:
    st.subheader("実現損益（時間別・1時間ごと）")
    if realized_all.empty and yakujyou_all.empty:
        st.info("実現損益または約定履歴が必要です。")
    else:
        # 時間情報は realized にあれば最優先、無ければ yakujyou
        if not realized_all.empty and realized_all["約定日時_final"].notna().any():
            dt = realized_all[["約定日時_final","pl"]].dropna()
            dt_col = "約定日時_final"
            pl_col = "pl"
        elif not yakujyou_all.empty and yakujyou_all["約定日時"].notna().any():
            # yak には損益が無いことが多いので金額はカウントのみになる場合あり
            dt = yakujyou_all[["約定日時"]].dropna().copy()
            dt["pl"] = 0.0
            dt_col = "約定日時"
            pl_col = "pl"
        else:
            st.info("市場時間内に“時刻付き”レコードがありません。")
            dt = pd.DataFrame()

        if not dt.empty:
            tt = pd.to_datetime(dt[dt_col], errors="coerce")
            if getattr(tt.dt, "tz", None) is None:
                tt = tt.dt.tz_localize(TZ, nonexistent="NaT", ambiguous="NaT")
            else:
                tt = tt.dt.tz_convert(TZ)
            dt = dt.assign(__t__=tt).dropna(subset=["__t__"])
            tl = dt["__t__"].dt.time
            mask_mkt = (tl >= time(9,0)) & (tl <= time(15,30))
            dt = dt[mask_mkt]
            if dt.empty:
                st.info("市場時間内の時刻付きレコードがありません。")
            else:
                dt["hour"] = dt["__t__"].dt.floor("h").dt.strftime("%H:00")
                g = dt.groupby("hour").agg(収支=(pl_col,"sum"),
                                          取引回数=(pl_col,"count"),
                                          平均損益=(pl_col,"mean")).reset_index()
                # 勝率（pl>0の比率）
                win = dt.assign(win=(dt[pl_col] > 0).astype(int)).groupby(dt["__t__"].dt.floor("h")).agg(win=("win","mean")).reset_index()
                win["hour"] = win["__t__"].dt.strftime("%H:00")
                g = g.merge(win[["hour","win"]], on="hour", how="left").rename(columns={"win":"勝率"})
                g = g.sort_values("hour")

                # 数値化＆表示（カンマ書式・数値ソート可）
                g = _numify_cols(g, ["収支","平均損益","勝率","取引回数"], round0=False)
                # 勝率(0-1)→%
                if g["勝率"].notna().any(): g["勝率"] = g["勝率"]*100.0
                show_numeric_table(g, {"収支":"%,.0f", "平均損益":"%,.0f", "勝率":"%.1f%%", "取引回数":"%,d"}, key="by_hour")

                # 収支/勝率の線グラフ
                if not g.empty:
                    x = g["hour"]
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=x, y=g["取引回数"], name="取引回数", yaxis="y2", opacity=0.3))
                    fig.add_trace(go.Scatter(x=x, y=g["収支"], name="収支", mode="lines+markers"))
                    fig.add_trace(go.Scatter(x=x, y=g["勝率"], name="勝率(%)", mode="lines+markers", yaxis="y3"))
                    fig.update_layout(
                        height=360, xaxis=dict(type="category"),
                        yaxis=dict(title="収支", tickformat=",.0f"),
                        yaxis2=dict(title="回数", overlaying="y", side="right", showgrid=False),
                        yaxis3=dict(title="勝率(%)", overlaying="y", anchor="free", position=1.08, showgrid=False),
                        legend=dict(orientation="h"), margin=dict(l=10,r=60,t=10,b=10)
                    )
                    st.plotly_chart(fig, use_container_width=True)

# app.py — Part 2/2

# =========================
# 3) 累計損益
# =========================
with tab3:
    st.subheader("累計損益")
    if realized_all.empty:
        st.info("実現損益ファイルをアップロードしてください。")
    else:
        d = realized_all.copy()
        day = d.groupby("約定日_final", dropna=True)["pl"].sum().reset_index().rename(columns={"約定日_final":"日付", "pl":"実現損益[円]"})
        day = day.sort_values("日付")
        day["累計"] = day["実現損益[円]"].cumsum()
        day = _numify_cols(day, ["実現損益[円]","累計"], round0=True)

        show_numeric_table(day, {"実現損益[円]":"%,.0f", "累計":"%,.0f"}, key="cum_table")

        if not day.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=pd.to_datetime(day["日付"]), y=day["実現損益[円]"], name="日次PL"))
            fig.add_trace(go.Scatter(x=pd.to_datetime(day["日付"]), y=day["累計"], name="累計", mode="lines"))
            fig.update_layout(
                height=380, xaxis_rangeslider_visible=False,
                xaxis=dict(type="date", tickformat="%Y-%m-%d"),
                yaxis=dict(tickformat=",.0f"),
                margin=dict(l=10,r=10,t=10,b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

# =========================
# 4) 個別/ランキング
# =========================
with tab4:
    st.subheader("個別銘柄・ランキング")
    if realized_all.empty:
        st.info("実現損益ファイルをアップロードしてください。")
    else:
        base = realized_all.copy()
        # 銘柄キー（コードがあれば優先）
        sym = None
        if "code_key" in base.columns:
            sym = base["code_key"].fillna(base.get("name_key"))
        elif "name_key" in base.columns:
            sym = base["name_key"]
        else:
            sym = pd.Series(["N/A"]*len(base), index=base.index)
        base = base.assign(symbol=sym)

        named_aggs = {
            "実現損益合計": ("pl", "sum"),
            "取引回数": ("pl", "count"),
            "1回平均損益": ("pl", "mean"),  # ← 先頭が数字でもOK（辞書展開）
        }
        by_symbol = base.groupby("symbol").agg(**named_aggs).reset_index().sort_values("実現損益合計", ascending=False)

        by_symbol = _numify_cols(by_symbol, ["実現損益合計","1回平均損益","取引回数"], round0=True)

        st.markdown("**個別銘柄**")
        show_numeric_table(by_symbol, {"実現損益合計":"%,.0f","1回平均損益":"%,.0f","取引回数":"%,d"}, key="per_symbol")

        # ランキング（上位/下位）
        st.markdown("**ランキング（上位10 / 下位10）**")
        top = by_symbol.head(10).copy()
        worst = by_symbol.tail(10).copy()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("上位10")
            show_numeric_table(top, {"実現損益合計":"%,.0f","1回平均損益":"%,.0f","取引回数":"%,d"}, key="rank_top")
        with c2:
            st.markdown("下位10")
            show_numeric_table(worst, {"実現損益合計":"%,.0f","1回平均損益":"%,.0f","取引回数":"%,d"}, key="rank_worst")

# =========================
# 5) 3分足 IN/OUT + 指標（先に日付を選び、その日にデータがある銘柄だけ選択）
# =========================

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

def _nearest_pos_by_ns(idx: pd.DatetimeIndex, t0: pd.Timestamp) -> int:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(TZ)
    if t0.tzinfo is None:
        t0 = t0.tz_localize(TZ)
    idx_ns = idx.asi8
    t0_ns  = int(pd.Timestamp(t0).value)
    dist = np.abs(idx_ns - t0_ns)
    return int(dist.argmin())

def align_trades_to_ohlc(ohlc: pd.DataFrame, trades: pd.DataFrame, max_gap_min=6):
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
        if ("買" in s and ("新規" in s or "建" in s)) or re.search(r"\bBUY\b.*\b(OPEN|NEW)\b", s, re.I): return "買建"
        if ("売" in s and ("新規" in s or "建" in s)) or re.search(r"\bSELL\b.*\b(OPEN|NEW)\b", s, re.I): return "売建"
        if ("売" in s and ("返済" in s or "決済" in s)) or re.search(r"\bSELL\b.*\b(CLOSE)\b", s, re.I): return "売埋"
        if ("買" in s and ("返済" in s or "決済" in s)) or re.search(r"\bBUY\b.*\b(CLOSE|COVER)\b", s, re.I): return "買埋"
        return None

    tdf["label4"] = tdf["side"].map(classify_side4)

    odf = ohlc.copy()
    tt = _to_jst_series(odf["time"], odf.index)
    odf = odf.set_index(tt).sort_index()

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
        pos = _nearest_pos_by_ns(window.index, t0)
        near_time = window.index[pos]
        price_on_bar = window.loc[near_time, "close"]
        out_rows.append({
            "time": near_time, "price": price_on_bar,
            "side": row["side"], "qty": row["qty"], "label4": row["label4"]
        })
    return pd.DataFrame(out_rows)

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
        dmin, dmax = ohlc_global_date_range(ohlc_map)
        if dmin is None or dmax is None:
            st.info("有効な日時列が見つかりませんでした。")
        else:
            # 初期値は当日（範囲外ならクランプ）
            today_jst = datetime.now(TZ).date()
            default_day = min(max(today_jst, dmin), dmax)

            c1, c2, c3 = st.columns([2,2,1])
            with c1:
                sel_date = st.date_input("表示日を選択", value=default_day, min_value=dmin, max_value=dmax)
            with c2:
                enlarge = st.toggle("🔍 拡大表示", value=False, help="チェックでチャートを大きくします")
            with c3:
                ht = LARGE_CHART_HEIGHT if enlarge else MAIN_CHART_HEIGHT

            # 9:00〜15:30 に固定
            t0 = pd.Timestamp(f"{sel_date} 09:00", tz=TZ)
            t1 = pd.Timestamp(f"{sel_date} 15:30", tz=TZ)
            x_range = [t0, t1]

            # 選択日の約定表（全銘柄）
            st.markdown("#### 約定表（選択日・全銘柄）")
            if yakujyou_all is None or yakujyou_all.empty:
                st.info("約定履歴が未アップロードです。")
            else:
                yak_day_all = yakujyou_all.copy()
                y_dtcol = pick_dt_col(yak_day_all) or "約定日"
                yak_day_all["約定日時"] = pick_dt_with_optional_time(yak_day_all) if y_dtcol in yak_day_all.columns else _to_jst_series(pd.Series(pd.NaT, index=yak_day_all.index), yak_day_all.index)
                yak_day_all = yak_day_all[yak_day_all["約定日時"].notna()]
                yak_day_all = yak_day_all[(yak_day_all["約定日時"]>=t0) & (yak_day_all["約定日時"]<=t1)].copy()

                if yak_day_all.empty:
                    st.info(f"{sel_date} の市場時間内に約定はありません。")
                else:
                    price_col = next((c for c in ["約定単価(円)","約定単価（円）","約定価格","価格","約定単価"] if c in yak_day_all.columns), None)
                    if price_col is None:
                        for c in yak_day_all.columns:
                            if re.search(r"(約定)?.*(単価|価格)", str(c)): price_col = c; break
                    qty_col   = next((c for c in ["約定数量(株/口)","約定数量","出来数量","数量","株数","出来高","口数"] if c in yak_day_all.columns), None)
                    if qty_col is None:
                        for c in yak_day_all.columns:
                            if any(k in str(c) for k in ["数量","株数","口数","出来高"]): qty_col = c; break
                    side_col  = next((c for c in ["売買","売買区分","売買種別","Side","取引"] if c in yak_day_all.columns), None)

                    disp = pd.DataFrame({
                        "時刻": yak_day_all["約定日時"].dt.strftime("%H:%M:%S"),
                        "銘柄名": yak_day_all.get("name_key", pd.Series([""]*len(yak_day_all), index=yak_day_all.index)),
                        "銘柄コード": yak_day_all.get("code_key", pd.Series([""]*len(yak_day_all), index=yak_day_all.index)),
                        "売買": yak_day_all[side_col] if side_col else "",
                        "価格": to_numeric_jp(yak_day_all[price_col]) if price_col else np.nan,
                        "数量": to_numeric_jp(yak_day_all[qty_col]) if qty_col else np.nan,
                    }).sort_values("時刻")

                    # 数値 dtype を維持（クリックで数値ソート可）＋見た目はカンマ
                    disp["価格"] = pd.to_numeric(disp["価格"], errors="coerce").round(0)
                    disp["数量"] = pd.to_numeric(disp["数量"], errors="coerce")
                    show_numeric_table(disp, {"価格":"%,.0f","数量":"%,d"}, key="fills_table")
                    download_button_df(disp, f"⬇ CSVダウンロード（約定表 {sel_date}）", f"fills_{sel_date}.csv")

            # 当日データがある銘柄のみ選択肢に
            keys_that_day = []
            for k, df in ohlc_map.items():
                if df is None or df.empty: continue
                vw = df[(df["time"]>=t0) & (df["time"]<=t1)]
                if not vw.empty:
                    keys_that_day.append(k)

            if not keys_that_day:
                st.info("選択日のデータがある銘柄が見つかりません。別の日付を選んでください。")
            else:
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
                    # 同コードの約定を抽出して近傍バーへスナップ
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

                    trades = align_trades_to_ohlc(view, yak, max_gap_min=6) if not yak.empty else pd.DataFrame(columns=["time","price","side","qty","label4"])
                    title_text = f"{sel_name} [{sel_key}]" if sel_name else sel_key
                    fig = make_candle_with_indicators(view, title=title_text, height=ht, x_range=[t0, t1])

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

            # 下段：日経先物・日経平均（あれば）
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
                    figx = make_candle_with_indicators(vw, title=ttl, height=ht, x_range=[t0, t1])
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
                    figx = make_candle_with_indicators(vw, title=ttl, height=ht, x_range=[t0, t1])
                    st.plotly_chart(figx, use_container_width=True, config={"displayModeBar": True})
                else:
                    st.info(f"{k}：{sel_date} のデータなし。")
            else:
                st.info("日経平均（`TVC_NI225` など）が見つかりません。")
