# -*- coding: utf-8 -*-
import io
import re
import os
from datetime import datetime, date, time, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =============================
# 基本設定
# =============================
st.set_page_config(page_title="トレードダッシュボード", layout="wide")

TZ = "Asia/Tokyo"
COLOR_VWAP = "#888888"   # グレー
COLOR_MA1  = "#2ca02c"   # 緑
COLOR_MA2  = "#ff7f0e"   # オレンジ
COLOR_MA3  = "#1f77b4"   # 青
MAIN_CHART_HEIGHT  = 420
LARGE_CHART_HEIGHT = 620

# =============================
# ユーティリティ
# =============================
@st.cache_data(show_spinner=False)
def read_csv_safely(file, encoding="utf-8-sig", **kwargs):
    errors = []
    for enc in [encoding, "cp932", "utf-8", "ISO-2022-JP", "latin-1"]:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc, **kwargs)
        except Exception as e:
            errors.append(f"{enc}: {e}")
    raise ValueError("文字コードの解釈に失敗しました。サイドバーの文字コードを変更して再試行してください。\n" + "\n".join(errors))


def to_numeric_jp(s):
    if s is None:
        return pd.Series(dtype=float)
    return (
        pd.to_numeric(
            pd.Series(s, dtype="object")
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("円", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace("\u2212", "-", regex=False)  # 全角マイナス
            .str.replace("−", "-", regex=False)       # 別の全角マイナス
            .str.strip(),
            errors="coerce",
        )
    )


def _to_jst_series(s, index_like):
    if s is None:
        return pd.Series(pd.NaT, index=index_like)
    dt = pd.to_datetime(s, errors="coerce")
    if isinstance(dt, pd.Series):
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(TZ)
        else:
            dt = dt.dt.tz_convert(TZ)
    else:
        if dt.tzinfo is None:
            dt = dt.tz_localize(TZ)
        else:
            dt = dt.tz_convert(TZ)
    return dt


def ensure_jst(dt_like):
    dt = pd.to_datetime(dt_like, errors="coerce")
    if isinstance(dt, pd.Series):
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(TZ)
        else:
            dt = dt.dt.tz_convert(TZ)
    else:
        if dt.tzinfo is None:
            dt = dt.tz_localize(TZ)
        else:
            dt = dt.tz_convert(TZ)
    return dt


def pick_dt_col(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    for c in ["約定日時_final", "約定日時_推定", "約定日時", "日時", "日付", "約定日"]:
        if c in df.columns:
            return c
    # あれば先頭の datetime っぽい列
    for c in df.columns:
        if re.search(r"(time|date|日時|日付)", str(c), flags=re.I):
            return c
    return None


def pick_dt_with_optional_time(df: pd.DataFrame):
    col = pick_dt_col(df)
    if col is None:
        return pd.Series(pd.NaT, index=df.index)
    dt = pd.to_datetime(df[col], errors="coerce")
    # 日付のみ（時間成分が欠落）を判定 -> NaTのまま扱い（時間別集計では除外）
    # ただし3分足タブの約定表などでは日次フィルタのために 00:00 が混ざることがあるので注意
    # ここでは一律にJST付与のみ（0時は表示側で除外可）
    if isinstance(dt, pd.Series):
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(TZ)
        else:
            dt = dt.dt.tz_convert(TZ)
    else:
        if dt.tzinfo is None:
            dt = dt.tz_localize(TZ)
        else:
            dt = dt.tz_convert(TZ)
    return dt


def extract_code_from_ohlc_key(key: str) -> str | None:
    # 例: "TSE_9984, 3_xxxx.csv" -> "TSE_9984"
    base = os.path.basename(str(key))
    head = base.split(",")[0]
    head = head.replace(".csv", "")
    m = re.search(r"([A-Z]{2,}_[0-9A-Z!]+)", head, flags=re.I)
    if m:
        return m.group(1).upper()
    # それ以外（指数など）はそのまま返す
    return head.strip() if head else None


def ohlc_global_date_range(ohlc_map: dict):
    tmins, tmaxs = [], []
    for _, df in ohlc_map.items():
        if df is None or df.empty or "time" not in df.columns:
            continue
        tt = ensure_jst(df["time"])  # tz-aware
        tmins.append(tt.min().date())
        tmaxs.append(tt.max().date())
    if not tmins:
        return None, None
    return min(tmins), max(tmaxs)


def compute_vwap_by_day(df: pd.DataFrame):
    # volume 列の検出
    vol_col = None
    for c in ["volume", "出来高", "出来高(株)", "出来高(口)", "Volume", "VOL"]:
        if c in df.columns:
            vol_col = c
            break
    if vol_col is None:
        return df  # VWAP不可
    out = df.copy()
    out["_pv"] = out["close"] * to_numeric_jp(out[vol_col])
    tt = ensure_jst(out["time"])
    out["_date"] = tt.dt.date
    out["_cum_pv"] = out.groupby("_date")["_pv"].cumsum()
    out["_cum_v"]  = out.groupby("_date")[vol_col].cumsum()
    out["VWAP"] = out["_cum_pv"] / out["_cum_v"].replace(0, np.nan)
    return out.drop(columns=["_pv", "_date", "_cum_pv", "_cum_v"])


def add_mas(df: pd.DataFrame, w1=5, w2=20, w3=60):
    out = df.copy()
    if "close" in out.columns:
        out["MA1"] = out["close"].rolling(w1, min_periods=1).mean()
        out["MA2"] = out["close"].rolling(w2, min_periods=1).mean()
        out["MA3"] = out["close"].rolling(w3, min_periods=1).mean()
    return out


def guess_name_for_ohlc_key(key: str, code_to_name: dict) -> str | None:
    code = extract_code_from_ohlc_key(key)
    name = None
    if code:
        name = code_to_name.get(str(code).upper())
    if not name:
        ku = str(key).upper()
        if "NK2251" in ku or "OSE_NK2251" in ku:
            name = "日経225先物"
        elif "NI225" in ku or "TVC_NI225" in ku:
            name = "日経平均"
    return name


def number_cfg(label=None, fmt="%,d"):
    return st.column_config.NumberColumn(label=label, format=fmt)


def percent_cfg(label=None, fmt="%.1f%%"):
    return st.column_config.NumberColumn(label=label, format=fmt)


def download_button_df(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# DatetimeIndex 最近傍位置（堅牢版）
def _nearest_pos_by_ns(idx: pd.DatetimeIndex, t0: pd.Timestamp) -> int:
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.DatetimeIndex(idx)
    if idx.tz is None:
        idx = idx.tz_localize(TZ)
    if t0.tzinfo is None:
        t0 = t0.tz_localize(TZ)
    idx_ns = idx.view("int64")
    t0_ns = int(pd.Timestamp(t0).value)
    dist = np.abs(idx_ns - t0_ns)
    return int(dist.argmin())

# 近傍バーへスナップ（買建/売建/売埋/買埋）
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
            if re.search(r"(約定)?.*(単価|価格)", str(c)):
                price_col = c; break
    if qty_col is None:
        for c in tdf.columns:
            if any(k in str(c) for k in ["数量","株数","口数","出来高"]):
                qty_col = c; break

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

# =============================
# サイドバー（アップロード）
# =============================
st.sidebar.header("📤 ファイルアップロード")
enc = st.sidebar.selectbox("文字コード", ["utf-8-sig", "cp932", "utf-8", "ISO-2022-JP", "latin-1"], index=0)

ohlc_files = st.sidebar.file_uploader("3分足 OHLC CSV（複数可）", type=["csv"], accept_multiple_files=True)
fill_files = st.sidebar.file_uploader("約定履歴 CSV（複数可）", type=["csv"], accept_multiple_files=True)
realized_files = st.sidebar.file_uploader("実現損益 CSV（複数可）", type=["csv"], accept_multiple_files=True)

ma1 = st.sidebar.number_input("MA1 ウィンドウ", min_value=1, max_value=200, value=5)
ma2 = st.sidebar.number_input("MA2 ウィンドウ", min_value=1, max_value=400, value=20)
ma3 = st.sidebar.number_input("MA3 ウィンドウ", min_value=1, max_value=800, value=60)

# =============================
# データ読み込み
# =============================
# 3分足
ohlc_map: dict[str, pd.DataFrame] = {}
if ohlc_files:
    for f in ohlc_files:
        try:
            df = read_csv_safely(f, encoding=enc)
            # 列名標準化
            cols = {c.lower(): c for c in df.columns}
            def pick(*names):
                for n in names:
                    if n in cols:
                        return cols[n]
                return None
            c_time = pick("time", "日時", "date", "datetime")
            c_open = pick("open", "始値")
            c_high = pick("high", "高値")
            c_low  = pick("low",  "安値")
            c_close= pick("close","終値")
            c_vol  = pick("volume","出来高","vol")
            if not all([c_time, c_open, c_high, c_low, c_close]):
                st.warning(f"{f.name}: 必須列(time/open/high/low/close)が見つかりません")
                continue
            out = pd.DataFrame({
                "time": ensure_jst(df[c_time]),
                "open": to_numeric_jp(df[c_open]),
                "high": to_numeric_jp(df[c_high]),
                "low":  to_numeric_jp(df[c_low]),
                "close":to_numeric_jp(df[c_close]),
            })
            if c_vol:
                out[c_vol] = to_numeric_jp(df[c_vol])
            out = compute_vwap_by_day(out)
            out = add_mas(out, w1=ma1, w2=ma2, w3=ma3)
            ohlc_map[f.name] = out
        except Exception as e:
            st.error(f"OHLC読込失敗: {f.name}: {e}")

# 約定履歴
yakujyou_all = pd.DataFrame()
if fill_files:
    frames = []
    for f in fill_files:
        try:
            d = read_csv_safely(f, encoding=enc)
            d["約定日時"] = pick_dt_with_optional_time(d)
            # 銘柄コード/銘柄名の標準化
            code_col = next((c for c in ["銘柄コード","コード","code","symbol"] if c in d.columns), None)
            name_col = next((c for c in ["銘柄名","名称","name"] if c in d.columns), None)
            if code_col: d["code_key"] = d[code_col].astype(str).str.upper()
            if name_col: d["name_key"] = d[name_col].astype(str)
            frames.append(d)
        except Exception as e:
            st.error(f"約定履歴読込失敗: {f.name}: {e}")
    if frames:
        yakujyou_all = pd.concat(frames, ignore_index=True)

# 実現損益
realized_all = pd.DataFrame()
if realized_files:
    frames = []
    for f in realized_files:
        try:
            d = read_csv_safely(f, encoding=enc)
            d["約定日時"] = pick_dt_with_optional_time(d)
            # 損益列を推定
            pl_col = next((c for c in [
                "実現損益（円）","実現損益(円)","実現損益","損益","損益[円]","損益(円)","損益（円）",
                "Realized P&L","P&L","PL","Profit"
            ] if c in d.columns), None)
            if pl_col is None:
                # 正規表現で "損益" を含む列のうち、数値化できそうなもの
                for c in d.columns:
                    if re.search(r"損益|P&L|PL|Profit", str(c), re.I):
                        pl_col = c; break
            if pl_col is not None:
                d["実現損益[円]"] = to_numeric_jp(d[pl_col])
            # 銘柄コード/銘柄名の標準化
            code_col = next((c for c in ["銘柄コード","コード","code","symbol"] if c in d.columns), None)
            name_col = next((c for c in ["銘柄名","名称","name"] if c in d.columns), None)
            if code_col: d["code_key"] = d[code_col].astype(str).str.upper()
            if name_col: d["name_key"] = d[name_col].astype(str)
            frames.append(d)
        except Exception as e:
            st.error(f"実現損益読込失敗: {f.name}: {e}")
    if frames:
        realized_all = pd.concat(frames, ignore_index=True)

# 銘柄コード→名称のマップ
CODE_TO_NAME: dict[str, str] = {}
for df in [yakujyou_all, realized_all]:
    if not df.empty and "code_key" in df.columns and "name_key" in df.columns:
        pairs = df[["code_key","name_key"]].dropna().drop_duplicates()
        for _, r in pairs.iterrows():
            CODE_TO_NAME[str(r["code_key"]).upper()] = str(r["name_key"]) if pd.notna(r["name_key"]) else CODE_TO_NAME.get(str(r["code_key"]).upper(), None)

# =============================
# タブ構成
# =============================
st.title("📊 トレードダッシュボード")

TAB_NAMES = ["集計（期間別）", "集計（時間別）", "累計損益", "個別銘柄", "3分足 IN/OUT + 指標"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_NAMES)

# -----------------------------
# 1) 集計（期間別）
# -----------------------------
with tab1:
    st.subheader("集計（期間別）")
    if realized_all.empty or "実現損益[円]" not in realized_all.columns:
        st.info("実現損益ファイルをアップロードしてください（損益列が必要）")
    else:
        r = realized_all.copy()
        r["dt"] = pick_dt_with_optional_time(r)
        r["日"] = ensure_jst(r["dt"]).dt.date
        r["週"] = ensure_jst(r["dt"]).dt.to_period("W-MON").apply(lambda p: p.start_time.date())
        r["月"] = ensure_jst(r["dt"]).dt.to_period("M").apply(lambda p: p.start_time.date())
        r["年"] = ensure_jst(r["dt"]).dt.to_period("Y").apply(lambda p: p.start_time.date())

        for label, col in [("日別","日"),("週別","週"),("月別","月"),("年別","年")]:
            g = r.groupby(col, as_index=False)["実現損益[円]"].sum().sort_values(col)
            g_disp = g.copy()
            g_disp["日付"] = pd.to_datetime(g_disp[col]).dt.strftime("%Y-%m-%d")

            st.write(f"**{label}**")
            st.dataframe(
                g_disp[["日付","実現損益[円]"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "実現損益[円]": number_cfg("実現損益[円]", fmt="%,d"),
                },
            )
            fig_bar = go.Figure([go.Bar(x=g_disp["日付"], y=g["実現損益[円]"], name=f"{label} 実現損益")])
            fig_bar.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=320, xaxis_title="日付", yaxis_title="実現損益[円]")
            st.plotly_chart(fig_bar, use_container_width=True)
            download_button_df(g[[col, "実現損益[円]"]].rename(columns={col: "date"}), f"⬇ CSVダウンロード（{label}）", f"agg_{col}.csv")

# -----------------------------
# 2) 集計（時間別）
# -----------------------------
with tab2:
    st.subheader("集計（時間別）")
    if realized_all.empty or "実現損益[円]" not in realized_all.columns:
        st.info("実現損益ファイルをアップロードしてください（損益列が必要）")
    else:
        df = realized_all.copy()
        dt = pick_dt_with_optional_time(df)
        df = df.assign(約定日時=dt)
        # 時刻がNaTのものは除外
        df = df[df["約定日時"].notna()].copy()
        tl = ensure_jst(df["約定日時"]).dt.time
        mask_mkt = (tl >= time(9,0)) & (tl <= time(15,30))
        df = df[mask_mkt]
        if df.empty:
            st.info("市場時間内の時刻付きレコードがありませんでした。")
        else:
            df["hour_x"] = ensure_jst(df["約定日時"]).dt.floor("h")
            by = df.groupby("hour_x").agg(
                収支=("実現損益[円]", "sum"),
                取引回数=("実現損益[円]", "count"),
                勝率=("実現損益[円]", lambda s: np.nan if len(s)==0 else (s>0).mean()),
                平均損益=("実現損益[円]", lambda s: s.mean() if len(s)>0 else np.nan)
            ).reset_index()
            # 表示（数値は dtype のまま）
            df_hour = by.copy()
            df_hour["時間"] = ensure_jst(df_hour["hour_x"]).dt.strftime("%H:%M")
            df_hour["勝率(%)"] = (df_hour["勝率"].fillna(0)*100).round(1)

            st.dataframe(
                df_hour[["時間","収支","取引回数","勝率(%)","平均損益"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "収支": number_cfg("収支", fmt="%,d"),
                    "取引回数": number_cfg("取引回数", fmt="%,d"),
                    "平均損益": number_cfg("平均損益", fmt="%,d"),
                    "勝率(%)": percent_cfg("勝率(%)", fmt="%.1f%%"),
                }
            )
            download_button_df(by, "⬇ CSVダウンロード（時間別）", "hourly_stats.csv")

            # グラフ：時間順に
            fig1 = go.Figure([go.Bar(x=df_hour["時間"], y=by["収支"], name="収支")])
            fig1.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=10), xaxis_title="時間", yaxis_title="収支")
            st.plotly_chart(fig1, use_container_width=True)

            fig2 = go.Figure([go.Scatter(x=df_hour["時間"], y=df_hour["勝率(%)"], mode="lines+markers", name="勝率")])
            fig2.update_layout(height=300, margin=dict(l=10,r=10,t=20,b=10), xaxis_title="時間", yaxis_title="勝率(%)")
            st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# 3) 累計損益
# -----------------------------
with tab3:
    st.subheader("累計損益")
    if realized_all.empty or "実現損益[円]" not in realized_all.columns:
        st.info("実現損益ファイルをアップロードしてください（損益列が必要）")
    else:
        r = realized_all.copy()
        r["dt"] = pick_dt_with_optional_time(r)
        r = r[r["dt"].notna()].copy()
        r["日"] = ensure_jst(r["dt"]).dt.date
        seq = r.groupby("日", as_index=False)["実現損益[円]"].sum().sort_values("日")
        seq["累計"] = seq["実現損益[円]"].cumsum()
        seq_disp = seq.copy()
        seq_disp["日付"] = pd.to_datetime(seq_disp["日"]).dt.strftime("%Y-%m-%d")
        st.dataframe(
            seq_disp[["日付","実現損益[円]","累計"]],
            use_container_width=True, hide_index=True,
            column_config={
                "実現損益[円]": number_cfg("実現損益[円]", fmt="%,d"),
                "累計": number_cfg("累計", fmt="%,d"),
            }
        )
        download_button_df(seq.rename(columns={"日":"date"}), "⬇ CSVダウンロード（累計・日次）", "cumulative_daily_pl.csv")

        fig = go.Figure([
            go.Bar(x=seq_disp["日付"], y=seq["実現損益[円]"], name="実現損益[円]"),
            go.Scatter(x=seq_disp["日付"], y=seq["累計"], name="累計", mode="lines")
        ])
        fig.update_layout(height=360, margin=dict(l=10,r=10,t=20,b=10), xaxis_title="日付", yaxis_title="金額[円]")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 4) 個別銘柄
# -----------------------------
with tab4:
    st.subheader("個別銘柄")
    if realized_all.empty or "実現損益[円]" not in realized_all.columns:
        st.info("実現損益ファイルをアップロードしてください（損益列が必要）")
    else:
        r = realized_all.copy()
        # コード・名称の推定
        if "code_key" not in r.columns:
            c = next((c for c in ["銘柄コード","コード","code","symbol"] if c in r.columns), None)
            if c: r["code_key"] = r[c].astype(str).str.upper()
        if "name_key" not in r.columns:
            c = next((c for c in ["銘柄名","名称","name"] if c in r.columns), None)
            if c: r["name_key"] = r[c].astype(str)

        if "code_key" not in r.columns:
            st.info("銘柄コード列が見つかりません。")
        else:
            agg = r.groupby("code_key").agg(**{
                "実現損益合計": ("実現損益[円]", "sum"),
                "取引回数": ("実現損益[円]", "count"),
                "1回平均損益": ("実現損益[円]", "mean"),
                "銘柄名": ("name_key", lambda s: s.dropna().iloc[0] if s.dropna().size>0 else ""),
            }).reset_index().rename(columns={"code_key":"銘柄コード"})

            st.dataframe(
                agg[["銘柄コード","銘柄名","実現損益合計","取引回数","1回平均損益"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "実現損益合計": number_cfg("実現損益合計", fmt="%,d"),
                    "取引回数": number_cfg("取引回数", fmt="%,d"),
                    "1回平均損益": number_cfg("1回平均損益", fmt="%,d"),
                }
            )
            download_button_df(agg, "⬇ CSVダウンロード（個別銘柄）", "per_symbol_stats.csv")

# -----------------------------
# 5) 3分足 IN/OUT + 指標（先に日付選択 → その日にデータがある銘柄のみ）
# -----------------------------

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
    st.subheader("3分足 IN/OUT + 指標（VWAP/MA1/MA2/MA3）")
    if not ohlc_map:
        st.info("3分足OHLCファイルをアップロードしてください。")
    else:
        dmin, dmax = ohlc_global_date_range(ohlc_map)
        if dmin is None or dmax is None:
            st.info("有効な日時列が見つかりませんでした。")
        else:
            today_jst = datetime.now(pd.Timestamp.now(tz=TZ).tz).date()
            default_day = today_jst
            if dmin and default_day < dmin: default_day = dmin
            if dmax and default_day > dmax: default_day = dmax

            c1, c2, c3 = st.columns([2,2,1])
            with c1:
                sel_date = st.date_input("表示日を選択", value=default_day, min_value=dmin, max_value=dmax)
            with c2:
                enlarge = st.toggle("🔍 拡大表示", value=False, help="チェックでチャートを大きくします")
            with c3:
                ht = LARGE_CHART_HEIGHT if enlarge else MAIN_CHART_HEIGHT

            # 時間レンジ固定 9:00〜15:30
            t0 = pd.Timestamp(f"{sel_date} 09:00", tz=TZ)
            t1 = pd.Timestamp(f"{sel_date} 15:30", tz=TZ)
            x_range = [t0, t1]

            # 選択日の約定表（全銘柄）
            st.markdown("#### 約定表（選択日・全銘柄）")
            if yakujyou_all is None or yakujyou_all.empty:
                st.info("約定履歴が未アップロードです。")
            else:
                yak_day_all = yakujyou_all.copy()
                yak_day_all["約定日時"] = pick_dt_with_optional_time(yak_day_all)
                yak_day_all = yak_day_all[yak_day_all["約定日時"].notna()]
                yak_day_all = yak_day_all[(yak_day_all["約定日時"]>=t0) & (yak_day_all["約定日時"]<=t1)].copy()
                if yak_day_all.empty:
                    st.info(f"{sel_date} の市場時間内に約定はありません。")
                else:
                    price_col = next((c for c in ["約定単価(円)","約定単価（円）","約定価格","価格","約定単価"] if c in yak_day_all.columns), None)
                    if price_col is None:
                        for c in yak_day_all.columns:
                            if re.search(r"(約定)?.*(単価|価格)", str(c)):
                                price_col = c; break
                    qty_col = next((c for c in ["約定数量(株/口)","約定数量","出来数量","数量","株数","出来高","口数"] if c in yak_day_all.columns), None)
                    if qty_col is None:
                        for c in yak_day_all.columns:
                            if any(k in str(c) for k in ["数量","株数","口数","出来高"]):
                                qty_col = c; break
                    side_col  = next((c for c in ["売買","売買区分","売買種別","Side","取引"] if c in yak_day_all.columns), None)

                    disp = pd.DataFrame({
                        "時刻": ensure_jst(yak_day_all["約定日時"]).dt.strftime("%H:%M:%S"),
                        "銘柄名": yak_day_all.get("name_key", pd.Series([""]*len(yak_day_all))),
                        "銘柄コード": yak_day_all.get("code_key", pd.Series([""]*len(yak_day_all))),
                        "売買": yak_day_all[side_col] if side_col else "",
                        "価格": to_numeric_jp(yak_day_all[price_col]) if price_col else np.nan,
                        "数量": to_numeric_jp(yak_day_all[qty_col]) if qty_col else np.nan,
                    }).sort_values("時刻")

                    st.dataframe(
                        disp,
                        use_container_width=True, hide_index=True,
                        column_config={
                            "価格": number_cfg("価格", fmt="%,d"),
                            "数量": number_cfg("数量", fmt="%,d"),
                        }
                    )
                    download_button_df(disp, f"⬇ CSVダウンロード（約定表 {sel_date}）", f"fills_{sel_date}.csv")

            # 当日データがある銘柄のみ選択可能に
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
                    yak = yak.copy()
                    yak["約定日時"] = pick_dt_with_optional_time(yak)
                    yak = yak[yak["約定日時"].notna()]
                    yak = yak[(yak["約定日時"]>=t0) & (yak["約定日時"]<=t1)]

                    trades = align_trades_to_ohlc(view, yak, max_gap_min=6) if not yak.empty else pd.DataFrame(columns=["time","price","side","qty","label4"])

                    title_text = f"{sel_name} [{sel_key}]" if sel_name else sel_key
                    fig = make_candle_with_indicators(view, title=title_text, height=ht, x_range=x_range)

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

                # 下段：日経先物 / 日経平均（その日データがあるもののみ・時間固定）
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
