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
    time_candidates = st.text_input("時刻/日付 列候補", value="time,Time,日時,日付,約定時間,約定日,datetime,Datetime")
    o_col = st.text_input("始値 列候補", value="open,Open,始値")
    h_col = st.text_input("高値 列候補", value="high,High,高値")
    l_col = st.text_input("安値 列候補", value="low,Low,安値")
    c_col = st.text_input("終値 列候補", value="close,Close,終値")
    v_col = st.text_input("出来高 列候補", value="volume,出来高")
    vwap_col = st.text_input("VWAP 列候補", value="VWAP,vwap")

# ================= ユーティリティ =================
def _drop_tz_index(idx):
    """DatetimeIndex の tz を外して naive に統一"""
    if isinstance(idx, pd.DatetimeIndex):
        try:
            return idx.tz_localize(None)
        except Exception:
            try:
                return idx.tz_convert(None)
            except Exception:
                return idx
    return idx

def _drop_tz_series(s: pd.Series) -> pd.Series:
    """Series[datetime] の tz を外して naive に統一"""
    if not isinstance(s, pd.Series):
        return s
    # 一度 datetime 型へ
    s = pd.to_datetime(s, errors="coerce")
    # tz を外す
    try:
        return s.dt.tz_localize(None)
    except Exception:
        try:
            return s.dt.tz_convert(None)
        except Exception:
            return s

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
        return pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")
    else:
        return load_text_table(file_bytes, encoding_choice, decimal, thousands)

def parse_datetime_index(df: pd.DataFrame, time_cols: List[str]) -> Tuple[pd.DataFrame, Optional[str]]:
    col = _find_first(df, time_cols)
    if col is None:
        return df, None
    s = _drop_tz_series(df[col])                 # ← tz を除去してから
    df[col] = s
    df = df.sort_values(col)
    try:
        df = df.set_index(col)
        df.index = _drop_tz_index(df.index)      # ← 念のため index 側も tz 除去
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

# ====== 売買/INOUT 正規化 & 数値クレンジング ======
def normalize_side(val: object) -> str | float:
    """BUY/SELL を決める（買建/売建/買埋/売埋 に対応。IN/OUTは返さない）"""
    s = str(val).strip()
    sl = s.lower()
    # 英語・略称
    if sl in ["buy", "b", "long", "現買", "新規買"]:
        return "BUY"
    if sl in ["sell", "s", "short", "現売", "新規売"]:
        return "SELL"
    # 日本語（建/埋 含む）
    if "買" in s:   # 買建/買埋/買 など
        return "BUY"
    if "売" in s:   # 売建/売埋/売 など
        return "SELL"
    return np.nan

def normalize_inout(val: object) -> str | float:
    """IN/OUT を決める（建=IN / 埋=OUT を判定）"""
    s = str(val).strip()
    sl = s.lower()
    if sl in ["in", "entry", "エントリー", "新規", "新規建"] or ("建" in s):
        return "IN"
    if sl in ["out", "exit", "決済", "返済", "手仕舞い", "クローズ"] or ("埋" in s):
        return "OUT"
    return np.nan

def clean_numeric_series(s: pd.Series) -> pd.Series:
    """¥, 円, カンマ, 全角マイナス, (123) → -123 などを吸収して数値化"""
    t = s.astype(str)
    t = t.str.replace(r"\((\s*[\d,\.]+)\)", r"-\1", regex=True)  # (123) -> -123
    t = t.str.replace("−", "-", regex=False)                    # 全角マイナス
    t = t.str.replace(",", "", regex=False)                      # 桁区切り
    t = t.str.replace("¥", "", regex=False).str.replace("円", "", regex=False)
    t = t.str.replace("%", "", regex=False)                      # %除去（必要なら）
    t = t.str.replace(r"[^\d\.\-\+eE]", "", regex=True)          # 残りの記号を除去
    return pd.to_numeric(t, errors="coerce")

# ================= セッション（約定をタブ1で使う） =================
if "trades_df" not in st.session_state:
    st.session_state["trades_df"] = None
if "trades_time_col" not in st.session_state:
    st.session_state["trades_time_col"] = None
if "trades_price_col" not in st.session_state:
    st.session_state["trades_price_col"] = None
if "trades_side_col" not in st.session_state:
    st.session_state["trades_side_col"] = None
if "trades_inout_col" not in st.session_state:
    st.session_state["trades_inout_col"] = None

# ================= タブ =================
tab1, tab2, tab3 = st.tabs(["① 3分足（OHLC）", "② 約定履歴", "③ 実現損益"])

# ---------- ① 3分足（OHLC） ----------
with tab1:
    st.subheader("3分足（OHLC）をアップロード（CSV/TSV/Excel）")
    ohlc_file = st.file_uploader("time, open, high, low, close, volume, VWAP などを含む表",
                                 type=["csv", "txt", "xlsx"], key="ohlc_upl")

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

        # 約定オーバーレイ（安全版）
        overlay_ok = st.checkbox(
            "約定履歴を重ねる（タブ②で読み込むと有効）",
            value=True,
            disabled=st.session_state["trades_df"] is None,
        )
        
        if overlay_ok and st.session_state["trades_df"] is not None:
            tdf = st.session_state["trades_df"].copy()
        
            # セッションから列名を取得（None 安全）
            t_time_col  = st.session_state.get("trades_time_col")
            t_price_col = st.session_state.get("trades_price_col")
            t_side_norm = "SIDE_NORM" if "SIDE_NORM" in tdf.columns else None
            t_inout_norm= st.session_state.get("trades_inout_col")
        
            # 列の存在チェック（無ければオーバーレイせず注意書き）
            if not t_time_col or t_time_col not in tdf.columns:
                st.caption("⚠ 約定の時刻列が見つからないため、オーバーレイをスキップしました。")
            else:
                # 価格列クレンジング（存在すれば）
                if t_price_col and t_price_col in tdf.columns and not pd.api.types.is_numeric_dtype(tdf[t_price_col]):
                    tdf[t_price_col] = clean_numeric_series(tdf[t_price_col])
        
                # 表示期間内に絞る（両者とも tz なしに統一して比較）
                if isinstance(view.index, pd.DatetimeIndex):
                    idx_naive = _drop_tz_index(view.index)
                    t_series  = _drop_tz_series(tdf[t_time_col])
                    if len(idx_naive) > 0:
                        min_ts, max_ts = idx_naive.min(), idx_naive.max()
                        mask = (t_series >= min_ts) & (t_series <= max_ts)
                        tdf = tdf[mask]
        
                # 1) BUY/SELL があれば描画
                if t_side_norm and t_side_norm in tdf.columns and t_price_col and t_price_col in tdf.columns:
                    buys  = tdf[tdf[t_side_norm] == "BUY"]
                    sells = tdf[tdf[t_side_norm] == "SELL"]
                    if len(buys) > 0:
                        fig.add_trace(go.Scatter(
                            x=buys[t_time_col], y=buys[t_price_col], mode="markers",
                            name="買", marker_symbol="triangle-up", marker_size=10, opacity=0.9,
                        ))
                    if len(sells) > 0:
                        fig.add_trace(go.Scatter(
                            x=sells[t_time_col], y=sells[t_price_col], mode="markers",
                            name="売", marker_symbol="triangle-down", marker_size=10, opacity=0.9,
                        ))
        
                # 2) IN/OUT があれば描画（SIDE が無くても表示）
                if t_inout_norm and t_inout_norm in tdf.columns and t_price_col and t_price_col in tdf.columns:
                    inns = tdf[tdf[t_inout_norm] == "IN"]
                    outs = tdf[tdf[t_inout_norm] == "OUT"]
                    if len(inns) > 0:
                        fig.add_trace(go.Scatter(
                            x=inns[t_time_col], y=inns[t_price_col], mode="markers",
                            name="IN", marker_symbol="x", marker_size=9, opacity=0.9,
                        ))
                    if len(outs) > 0:
                        fig.add_trace(go.Scatter(
                            x=outs[t_time_col], y=outs[t_price_col], mode="markers",
                            name="OUT", marker_symbol="diamond-open", marker_size=11, opacity=0.9,
                        ))

        st.plotly_chart(fig, use_container_width=True)

# ---------- ② 約定履歴 ----------
with tab2:
    st.subheader("約定履歴（CSV/TSV/Excel）")
    st.caption("想定列： 約定日/約定時間 / 売買 / 約定数 / 約定単価（列名は任意。下の候補で指定）")
    trades_file = st.file_uploader("約定履歴ファイル", type=["csv", "txt", "xlsx"], key="trades_upl")

    # 列候補（既定に「約定日」「約定単価(円)」なども含める）
    t_time_c = st.text_input("（約定）時刻 列候補", value="約定日,約定時間,日時,日付,time,Time")
    t_side_c = st.text_input("売買 列候補", value="売買,side,Side,区分,取引")
    t_qty_c  = st.text_input("数量（約定数） 列候補", value="約定数,数量,株数,約定数量,Qty,qty,サイズ,約定数量(株/口)")
    t_price_c= st.text_input("価格（約定単価） 列候補", value="約定単価,約定単価(円),単価,価格,Price,price")
    t_inout_c= st.text_input("IN/OUT 列候補（新規/返済・エントリー/決済 等）",
                             value="IN/OUT,INOUT,新規返済,新規/返済,entry_exit,EntryExit,区分2,取引種別")

    if trades_file is None:
        st.info("📄 ファイルを選ぶとタブ①に“買/売/IN/OUTマーカー”を重ねられます。")
    else:
        try:
            df_tr = load_any_table(trades_file.name, trades_file.getvalue(), encoding, decimal, thousands)
        except Exception as e:
            st.error("読み込みに失敗しました。")
            st.exception(e)
            st.stop()

        used_enc = df_tr.attrs.get("used_encoding")
        used_sep = df_tr.attrs.get("used_sep")
        if used_enc or used_sep:
            st.caption(f"🔎 encoding={used_enc or 'Excel'}, sep={used_sep or '(Excel)'}")

        # ---- 列検出（依存関数なしの安全版）----
        def pick_col(df: pd.DataFrame, cand_str: str) -> Optional[str]:
            cands = [s.strip() for s in str(cand_str).split(",") if s.strip()]
            for c in cands:
                if c in df.columns:
                    return c
            return None

        t_time  = pick_col(df_tr, t_time_c)
        t_side  = pick_col(df_tr, t_side_c)
        t_qty   = pick_col(df_tr, t_qty_c)
        t_price = pick_col(df_tr, t_price_c)
        t_inout = pick_col(df_tr, t_inout_c)

        # 型変換
        if t_time:
            df_tr[t_time] = pd.to_datetime(df_tr[t_time], errors="coerce")
            df_tr[t_time] = _drop_tz_series(df_tr[t_time])   # ← 追加：tz を外す

        for col in [t_qty, t_price]:
            if col and col in df_tr.columns:
                df_tr[col] = clean_numeric_series(df_tr[col])

        # 正規化カラムを追加（IN/OUT列が無くても 売買 から導出）
        if t_side and t_side in df_tr.columns:
            df_tr["SIDE_NORM"] = df_tr[t_side].apply(normalize_side)

        if t_inout and t_inout in df_tr.columns:
            df_tr["INOUT_NORM"] = df_tr[t_inout].apply(normalize_inout)
        elif t_side and t_side in df_tr.columns:
            df_tr["INOUT_NORM"] = df_tr[t_side].apply(normalize_inout)

        st.write("#### プレビュー")
        st.dataframe(df_tr.head(200))

        with st.expander("検出状況（デバッグ）"):
            st.write({
                "time_col": t_time, "price_col": t_price, "side_col": t_side, "inout_col": t_inout
            })
            if "SIDE_NORM" in df_tr.columns:
                st.write("SIDE_NORM counts:", df_tr["SIDE_NORM"].value_counts(dropna=False))
            if "INOUT_NORM" in df_tr.columns:
                st.write("INOUT_NORM counts:", df_tr["INOUT_NORM"].value_counts(dropna=False))

        with st.expander("簡易サマリ"):
            total_rows = len(df_tr)
            buy_n = int(df_tr["SIDE_NORM"].eq("BUY").sum()) if "SIDE_NORM" in df_tr.columns else 0
            sell_n = int(df_tr["SIDE_NORM"].eq("SELL").sum()) if "SIDE_NORM" in df_tr.columns else 0
            st.write(f"- 行数: {total_rows} / 買: {buy_n} / 売: {sell_n}")
            if t_qty:
                st.write(f"- 総数量: {pd.to_numeric(df_tr[t_qty], errors='coerce').sum():,.0f}")
            if t_price:
                st.write(f"- 価格（約定単価）min/median/max: {df_tr[t_price].min()} / {df_tr[t_price].median()} / {df_tr[t_price].max()}")

        # タブ①で使うためセッション保存
        st.session_state["trades_df"] = df_tr
        st.session_state["trades_time_col"] = t_time
        st.session_state["trades_price_col"] = t_price
        st.session_state["trades_side_col"] = t_side
        st.session_state["trades_inout_col"] = "INOUT_NORM" if "INOUT_NORM" in df_tr.columns else None

# ---------- ③ 実現損益 ----------
with tab3:
    st.subheader("実現損益（CSV/TSV/Excel）")
    st.caption("想定列： 日付 / 実現損益（列名自由、下の候補で指定）")
    pnl_file = st.file_uploader("実現損益ファイル", type=["csv", "txt", "xlsx"], key="pnl_upl")
    d_col_cand = st.text_input("日付 列候補", value="日付,日時,Date,date")
    pnl_col_cand = st.text_input("損益 列候補", value="実現損益,損益,PnL,Profit,profit")

    if pnl_file is None:
        st.info("📄 ファイルを選ぶと推移と累積を描画します。")
    else:
        try:
            df_pnl = load_any_table(pnl_file.name, pnl_file.getvalue(), encoding, decimal, thousands)
        except Exception as e:
            st.error("読み込みに失敗しました。")
            st.exception(e)
            st.stop()

        used_enc = df_pnl.attrs.get("used_encoding")
        used_sep = df_pnl.attrs.get("used_sep")
        if used_enc or used_sep:
            st.caption(f"🔎 encoding={used_enc or 'Excel'}, sep={used_sep or '(Excel)'}")

        d_col = _find_first(df_pnl, _split_candidates(d_col_cand))

        # 列候補から見つからない場合に備えて、ゆるめ検出
        p_col = _find_first(df_pnl, _split_candidates(pnl_col_cand))
        if p_col is None:
            tokens = ["損", "益", "損益", "実現", "pl", "p/l", "profit", "pnl", "realized"]
            cand_names = [c for c in df_pnl.columns if any(t in str(c).lower() for t in tokens)]
            best_col, best_ratio = None, 0.0
            for c in cand_names:
                ser = clean_numeric_series(df_pnl[c])
                ratio = ser.notna().mean()
                if ratio > best_ratio:
                    best_ratio, best_col = ratio, c
            if best_col is not None and best_ratio >= 0.5:
                p_col = best_col

        if d_col:
            df_pnl[d_col] = pd.to_datetime(df_pnl[d_col], errors="coerce")
            df_pnl = df_pnl.sort_values(d_col).set_index(d_col)

        if p_col is None:
            st.error("損益列が見つかりません。列候補に実際の列名を追記してください。")
        else:
            # 通貨記号・カンマ・括弧・全角マイナスなどを吸収
            df_pnl[p_col] = clean_numeric_series(df_pnl[p_col])

            st.write("#### プレビュー")
            st.dataframe(df_pnl[[p_col]].head(500))

            st.write("#### 日次（または時系列）推移")
            st.line_chart(df_pnl[[p_col]], height=300)

            st.write("#### 累積損益")
            cum = df_pnl[[p_col]].cumsum().rename(columns={p_col: "累積"})
            st.line_chart(cum, height=300)
