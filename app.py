import io
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CSV Viewer (Upload)", layout="wide")

st.title("CSVアップロード & 可視化（Streamlit）")
st.caption("①CSVをアップロード → ②文字コードなどを選ぶ → ③プレビュー・集計・グラフを見る")

# ========== サイドバー ==========
with st.sidebar:
    st.header("読み込み設定")
    encoding = st.selectbox(
        "文字コード（文字化け対策）",
        options=["utf-8", "utf-8-sig", "cp932 (Shift_JIS)"],
        index=0,
        help="日本のCSVで文字化けする場合は Shift_JIS (=cp932) を試してください。",
    )
    decimal = st.selectbox("小数点記号", options=[".", ","], index=0)
    thousands = st.selectbox("桁区切り", options=[None, ",", "_", " "], index=0)
    preview_rows = st.slider("プレビュー行数", min_value=5, max_value=200, value=50, step=5)
    time_candidates = st.text_input(
        "日時カラム候補（カンマ区切り）",
        value="time,Time,datetime,Datetime,日付,日時,約定時間",
        help="ここに書かれた名前の列が見つかれば時系列として解釈します。",
    )

st.write("## 1) CSVファイルをアップロード")
uploaded = st.file_uploader("CSVを1つ選択", type=["csv"])

@st.cache_data(show_spinner=True)
def load_dataframe(file_bytes: bytes, encoding: str, decimal: str, thousands):
    # pandasに渡す引数を準備
    read_kwargs = {
        "encoding": encoding if "cp932" not in encoding else "cp932",
        "decimal": decimal,
        "thousands": thousands if thousands not in (None, "None", "") else None,
        "engine": "python",   # 区切り文字の自動推定を許可（柔軟）
    }
    with io.BytesIO(file_bytes) as f:
        df = pd.read_csv(f, **read_kwargs)
    return df

def coerce_datetime(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """候補名のいずれかが存在すれば、時刻としてパースしてインデックスに設定"""
    for col in candidates:
        if col in df.columns:
            s = pd.to_datetime(df[col], errors="coerce")
            # タイムゾーン無しなら、そのまま（必要なら .dt.tz_localize で付与）
            df[col] = s
            df = df.sort_values(col)
            try:
                df = df.set_index(col)
            except Exception:
                pass
            return df, col
    return df, None

if uploaded is None:
    st.info("📄 左のサイドバーで設定を調整しつつ、上のボタンからCSVをアップロードしてください。")
    st.stop()

# 読み込み
try:
    df = load_dataframe(uploaded.getvalue(), encoding, decimal, thousands)
except UnicodeDecodeError:
    st.error("文字コードの解釈に失敗しました。サイドバーの文字コードを変更して再試行してください。")
    st.stop()
except Exception as e:
    st.exception(e)
    st.stop()

# 日時解釈（候補はサイドバーで変更可能）
cands = [s.strip() for s in time_candidates.split(",") if s.strip()]
df, time_col = coerce_datetime(df, cands)

# ========== 2) プレビュー ========== 
st.write("## 2) プレビュー")
st.dataframe(df.head(preview_rows))

# ========== 3) 基本統計 ==========
st.write("## 3) 基本統計（数値列）")
num = df.select_dtypes(include="number")
if num.empty:
    st.info("数値列が見つかりませんでした。")
else:
    st.dataframe(num.describe().T)

# ========== 4) グラフ ==========
st.write("## 4) かんたん可視化")
import numpy as np

def pick_numeric_series(df: pd.DataFrame) -> pd.Series | None:
    # よくある終値／価格系の候補を優先
    priority = ["close", "Close", "終値", "約定単価", "price", "Price"]
    for c in priority:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return df[c]
    # なければ最初の数値列
    numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numcols:
        return df[numcols[0]]
    return None

y = pick_numeric_series(df)

if y is None:
    st.info("プロット可能な数値列が見つかりませんでした。")
else:
    if time_col is not None and df.index.is_monotonic_increasing:
        st.line_chart(y, height=320)
        st.caption(f"X軸：{time_col}（時系列） / Y軸：{y.name}")
    else:
        st.line_chart(y.reset_index(drop=True), height=320)
        st.caption(f"X軸：行番号 / Y軸：{y.name}")

# ========== 5) ダウンロード ==========
st.write("## 5) 加工データをダウンロード")
csv_bytes = df.to_csv(index=True).encode("utf-8-sig")
st.download_button(
    "この表をCSVで保存（UTF-8 BOM付き）",
    data=csv_bytes,
    file_name="processed.csv",
    mime="text/csv",
)

st.success("✅ 完了：アップロード → 表示 → 統計 → グラフ → ダウンロード")
