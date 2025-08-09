import os
import streamlit as st
from dotenv import load_dotenv

# 推奨のインポート（非推奨警告が出ない）
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# ローカル実行時に .env を読む（本番は Streamlit Secrets で自動参照）
load_dotenv()

st.title("専門家AIに質問してみよう")
st.write(
    "このアプリは、選んだ専門家の立場であなたの質問に回答します。\n"
    "1) 専門家の種類を選ぶ → 2) 質問を入力 → 3) 送信 を押してください。\n"
    "※ 会話履歴を保持して、前回の回答を踏まえて続きから相談できます。"
)

# --- UI ---
expert = st.radio("専門家の種類を選んでください", ["医師", "弁護士", "教師"])
question = st.text_input("質問を入力してください")

# --- ロール別システムメッセージ ---
SYSTEM_MAP = {
    "医師": "あなたは経験豊富な内科医です。安全第一で、受診の目安やセルフケアも具体的に助言してください。",
    "弁護士": "あなたは弁護士です。一般向けの法的な情報提供として、わかりやすく丁寧に解説してください。最終判断は専門家相談が必要な旨も添えてください。",
    "教師": "あなたはやさしい学校の先生です。要点を段階的に、例を交えてわかりやすく説明してください。",
}

# --- セッション状態（会話履歴 & 現在の専門家） ---
if "history" not in st.session_state:
    st.session_state.history = []
if "expert" not in st.session_state:
    st.session_state.expert = expert

# 専門家を切り替えたら履歴をリセット（文脈混在を防止）
if expert != st.session_state.expert:
    st.session_state.history = []
    st.session_state.expert = expert

def ask_llm(text: str, role: str) -> str:
    """入力テキストと選択ロールからLLMの回答文字列を返す（課題要件の関数）。"""
    if not text:
        return "質問が空です。"

    # OPENAI_API_KEY は .env または Streamlit Secrets（デプロイ時）に設定
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 履歴 + 今回のシステム/ユーザー発話でメッセージを構成
    messages = [SystemMessage(content=SYSTEM_MAP[role])]
    messages.extend(st.session_state.history)
    messages.append(HumanMessage(content=text))

    # 推奨API（invoke）で呼び出し
    resp = llm.invoke(messages)

    # 履歴を更新（次の発話に引き継ぐ）
    st.session_state.history.append(HumanMessage(content=text))
    st.session_state.history.append(AIMessage(content=resp.content))

    return resp.content

# --- ボタン（メイン） ---
left, right = st.columns(2)

with left:
    if st.button("送信"):
        if not question:
            st.error("質問を入力してください。")
        else:
            with st.spinner("考え中…"):
                answer = ask_llm(question, expert)
            st.subheader("回答:")
            st.write(answer)

with right:
    if st.button("会話をリセット"):
        st.session_state.history = []
        st.success("会話履歴をリセットしました。")

# --- サイドバーにも同じ操作を配置 ---
with st.sidebar:
    st.header("設定 / 操作")
    st.write(f"現在の専門家: **{expert}**")
    turns = len(st.session_state.history) // 2
    st.caption(f"現在の会話ターン数: {turns}")

    if st.button("🔁 会話をリセット（サイドバー）"):
        st.session_state.history = []
        st.success("履歴を消去しました。")

    st.divider()
    st.markdown(
        "#### 使い方\n"
        "1. 専門家を選ぶ\n"
        "2. 質問を入力\n"
        "3. 送信\n\n"
        "※ 専門家を切り替えると履歴は自動的にリセットされます。"
    )
