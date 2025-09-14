import io
import csv
from dataclasses import dataclass
from typing import Optional, Tuple, List

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# 文字コード推定（無くても動くようにtry）
try:
    from charset_normalizer import from_bytes as cn_from_bytes
except Exception:
    cn_from_bytes = None

st.set_page_config(page_title="3分足＋約定オーバーレイ（実現損益付き）", layout="wide")
st.title("CSV/Excelアップロード：3分足（OHLC）＋約定オーバーレイ＋実現損益")
st.caption("タブごとにCSV/Excelをアップロード。文字コード・区切りは自動判別にも対応。")

# ================= サイドバー（共通設定） =================
with st.sidebar:
    st.header("読み込み設定")
    encoding = st.selectbox(
        "文字コード",
        options=[
            "auto (自動判別)", "utf-8", "utf-8-sig",
            "cp932 (Shift_JIS)", "utf-16", "utf-16-le", "utf-16-be",
            "euc_jp", "iso2022_jp"
        ],
        index=0,
        help="不明なときは『auto』でOK。Excel/TSV/セミコロン区切りも自動で推定します。",
    )
    decimal = st.selectbox("小数点記号", options=[".", ","], index=0)
    thousands = st.selectbox("桁区切り", options=[None, ",", "_", " "], index=0)

    st.divider()
    st.header("列名の候補（カンマ区切りで追記OK）")
    time_candidates = st.text_input("時刻/日付 列候補", value="time,Time,日時,日付,約定時間,datetime,Datetime")
    o_col = st.text_input("始値 列候補", value="open,Open,始値")
    h_col = st.text_input("高値 列候補", value="high,High,高値")
    l_col = st.text_input("安値 列候補", value="low,Low,安値")
    c_col = st.text_input("終値 列候補", value="close,Close,終値")
    v_col = st.text_input("出来高 列候補", value="volume,出来高")
    vwap_col = st.text_input("VWAP 列候補", value="VWAP,vwap")

# ================= ユーティリティ =================
def _split_candidates(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def _find_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

@st.cache_data(show_spinner=True)
def load_text_table(file_bytes: bytes, encoding_choice: str, decimal: str, thousands):
    """
    CSV/TSV/セミコロン/パイプ区切りのテキスト表を読み込む。
    - 文字コード：自動判別 + BOMヒューリスティクス + フォールバック候補
    - 区切り文字：csv.Sniffer + ヒューリスティクス
    """
    # 文字コード候補を組み立て
    candidates = []
    if encoding_choice.startswith("auto"):
        b = file_bytes
        if b.startswith(b"\xff\xfe") or b.startswith(b"\xfe\xff"):
            candidates.append("utf-16")
        elif b.startswith(b"\xef\xbb\xbf"):
            candidates.append("utf-8-sig")
        if cn_from_bytes is not None:
            try:
                best = cn_from_bytes(b).best()
                if best and best.encoding:
                    candidates.insert(0, best.encoding)
            except Exception:
                pass
        candidates += ["utf-8", "cp932", "utf-16", "utf-16-le", "utf-16-be", "euc_jp", "iso2022_jp"]
    else:
        candidates = ["cp932" if "cp932" in encoding_choice else encoding_choice]

    last_err = None
    for enc in dict.fromkeys(candidates):  # 重複除去しつつ順番維持
        try:
            text = file_bytes.decode(enc, errors="strict")
            sample = text[:20000]
            sep = None
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
                sep = dialect.delimiter
            except Exception:
                # ざっくりヒューリスティクス
                if sample.count("\t") > sample.count(",") and sample.count("\t") >= 2:
                    sep = "\t"
                elif sample.count(";") > sample.count(",") and sample.count(";") >= 2:
                    sep = ";"
                elif sample.count("|") > sample.count(",") and sample.count("|") >= 2:
                    sep = "|"

            df = pd.read_csv(
                io.StringIO(text),
                sep=sep,
                decimal=decimal,
                thousands=None if thousands in (None, "None", "") else thousands,
                engine="python",
            )
            df.attrs["used_encoding"] = enc
            df.attrs["used_sep"] = sep or ","
            return df
        except Exception as e:
            last_err = e
            continue
    # すべて失敗した場合は最後の例外を投げる
    raise last_err

@st.cache_data(show_spinner=True)
def load_any_table(file_name: str, file_bytes: bytes, encoding_choice: str, decimal: str, thousands):
    """
    拡張子で分岐：.xlsx は Excel、その他はテキスト表とみなす
    """
    if file_name.lower().endswith(".xlsx"):
        # ExcelはBytesIOから読む
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        return load_text_table(file_bytes, encoding_choice, decimal, thousands)

def parse_datetime_index(df: pd.DataFrame, time_cols: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    col = _find_first(df, time_cols)
    if col is None:
        return df, None
    s = pd.to_datetime(df[col], errors="coerce")
    df[col] = s
    df = df.sort_values(col)
    try:
        df = df.set_index(col)
    except Exception:
        pass
    return df, col

@dataclass
class OHLCCols:
    open: str
    high: str
    low: str
    close: str
    volume: Optional[str] = None
    vwap: Optional[str] = None

def detect_ohlc_cols(df: pd.DataFrame,
                     o_cands: List[str], h_cands: List[str],
                     l_cands: List[str], c_cands: List[str],
                     v_cands: List[str], vwap_cands: List[str]) -> Optional[OHLCCols]:
    o = _find_first(df, o_cands)
    h = _find_first(df, h_cands)
    l = _find_first(df, l_cands)
    c = _find_first(df, c_cands)
    if not all([o, h, l, c]):
        return None
    v = _find_first(df, v_cands)
    vwap = _find_first(df, vwap_cands)
    return OHLCCols(open=o, high=h, low=l, close=c, volume=v, vwap=vwap)

def cast_numeric(df: pd.DataFrame, cols: List[Optional[str]]):
    for col in cols:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

# ================= セッション（約定をタブ1で使う） =================
if "trades_df" not in st.session_state:
    st.session_state["trades_df"] = None
if "trades_time_col" not in st.session_state:
    st.session_state["trades_time_col"] = None
if "trades_price_col" not in st.session_state:
    st.session_state["trades_price_col"] = None
if "trades_side_col" not in st.session_state:
    st.session_state["trades_side_col"] = None

# ================= タブ =================
tab1, tab2, tab3 = st.tabs(["① 3分足（OHLC）", "② 約定履歴", "③ 実現損益"])

# ---------- ① 3分足（OHLC） ----------
with tab1:
    st.subheader("3分足（OHLC）をアップロード（CSV/TSV/Excel）")
    ohlc_file = st.file_uploader("time, open, high, low, close, volume, VWAP などを含む表", type=["csv", "txt", "xlsx"], key="ohlc_upl")

    if ohlc_file is None:
        st.info("📄 ファイルを選んでください。")
    else:
        try:
            df_ohlc = load_any_table(ohlc_file.name, ohlc_file.getvalue(), encoding, decimal, thousands)
        except Exception as e:
            st.error("読み込みに失敗しました。サイドバーの文字コードや区切りを見直すか、Excel/CSVの形式をご確認ください。")
            st.exception(e)
            st.stop()

        # 参考情報（何で読めたか）
        used_enc = df_ohlc.attrs.get("used_encoding")
        used_sep = df_ohlc.attrs.get("used_sep")
        if used_enc or used_sep:
            st.caption(f"🔎 encoding={used_enc or 'Excel'}, sep={used_sep or '(Excel)'}")

        # 時刻→インデックス
        df_ohlc, ohlc_time_col = parse_datetime_index(df_ohlc, _split_candidates(time_candidates))
        # 列検出
        cols = detect_ohlc_cols(
            df_ohlc,
            _split_candidates(o_col), _split_candidates(h_col),
            _split_candidates(l_col), _split_candidates(c_col),
            _split_candidates(v_col), _split_candidates(vwap_col),
        )
        if cols is None:
            st.error("OHLC列（始値/高値/安値/終値）が見つかりません。サイドバーの候補に実際の列名を追記して再読込してください。")
            st.stop()

        # 数値化
        cast_numeric(df_ohlc, [cols.open, cols.high, cols.low, cols.close, cols.volume, cols.vwap])

        st.write("#### プレビュー")
        st.dataframe(df_ohlc.head(100))

        # 表示範囲
        st.write("#### 表示範囲（任意）")
        if isinstance(df_ohlc.index, pd.DatetimeIndex) and len(df_ohlc) > 1:
            min_d, max_d = df_ohlc.index.min(), df_ohlc.index.max()
            rng = st.slider("期間を指定", min_value=min_d.to_pydatetime(), max_value=max_d.to_pydatetime(),
                            value=(min_d.to_pydatetime(), max_d.to_pydatetime()))
            view = df_ohlc.loc[(df_ohlc.index >= rng[0]) & (df_ohlc.index <= rng[1])].copy()
        else:
            view = df_ohlc.copy()

        # ローソク足
        fig = go.Figure()
        x = view.index if isinstance(view.index, pd.DatetimeIndex) else np.arange(len(view))
        fig.add_trace(go.Candlestick(
            x=x, open=view[cols.open], high=view[cols.high],
            low=view[cols.low], close=view[cols.close], name="OHLC",
        ))

        # VWAP
        if cols.vwap and cols.vwap in view.columns:
            fig.add_trace(go.Scatter(x=x, y=view[cols.vwap], mode="lines", name="VWAP", opacity=0.85))

        # 約定オーバーレイ
        overlay_ok = st.checkbox("約定履歴を重ねる（タブ②で読み込むと有効）",
                                 value=True, disabled=st.session_state["trades_df"] is None)
        if overlay_ok and st.session_state["trades_df"] is not None:
            tdf = st.session_state["trades_df"].copy()
            t_time = st.session_state["trades_time_col"]
            t_price = st.session_state["trades_price_col"]
            t_side = st.session_state["trades_side_col"]

            # 表示期間内に絞る
            if isinstance(view.index, pd.DatetimeIndex) and t_time:
                tdf = tdf[(tdf[t_time] >= view.index.min()) & (tdf[t_time] <= view.index.max())]

            # 買い/売りに分けて描画
            if t_side and t_side in tdf.columns:
                buys = tdf[tdf[t_side] == "BUY"]
                sells = tdf[tdf[t_side] == "SELL"]
            else:
                buys, sells = tdf, pd.DataFrame(columns=tdf.columns)

            if t_time and t_price and (t_time in tdf.columns) and (t_price in tdf.columns):
                if len(buys) > 0:
                    fig.add_trace(go.Scatter(
                        x=buys[t_time], y=buys[t_price], mode="markers",
                        name="買", marker_symbol="triangle-up", marker_size=10, opacity=0.9,
                    ))
                if len(sells) > 0:
                    fig.add_trace(go.Scatter(
                        x=sells[t_time], y=sells[t_price], mode="markers",
                        name="売", marker_symbol="triangle-down", marker_size=10, opacity=0.9,
                    ))

        fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), xaxis_title=ohlc_time_col or "index")
        st.plotly_chart(fig, use_container_width=True)

# ---------- ② 約定履歴 ----------
with tab2:
    st.subheader("約定履歴（CSV/TSV/Excel）")
    st.caption("想定列： 約定時間 / 売買 / 約定数 / 約定単価（列名は任意。下の候補で指定）")
    trades_file = st.file_uploader("約定履歴ファイル", type=["csv", "txt", "xlsx"], key="trades_upl")

    # 列候補
    t_time_c = st.text_input("（約定）時刻 列候補", value="約定時間,日時,日付,time,Time")
    t_side_c = st.text_input("売買 列候補", value="売買,side,Side,区分,取引")
    t_qty_c  = st.text_input("数量（約定数） 列候補", value="約定数,数量,株数,約定数量,Qty,qty,サ_
