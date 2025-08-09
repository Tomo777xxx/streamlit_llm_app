import streamlit as st
from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# .envからAPIキーを読み込む
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Streamlitのタイトル
st.title("専門家AIに質問してみよう")

# 専門家の選択
expert_type = st.radio("専門家の種類を選んでください", ["医師", "弁護士", "教師"])

# 質問入力
question = st.text_input("質問を入力してください")

# 専門家ごとのプロンプト
system_messages = {
    "医師": "あなたは経験豊富な医師です。健康や病気について丁寧に答えてください。",
    "弁護士": "あなたは優秀な弁護士です。法律的な視点で誠実にアドバイスしてください。",
    "教師": "あなたは親切な学校の先生です。分かりやすく丁寧に説明してください。"
}

# 実行ボタンでChatGPT呼び出し
if st.button("送信"):
    if not question:
        st.error("質問を入力してください。")
    else:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, openai_api_key=api_key)
        messages = [
            SystemMessage(content=system_messages[expert_type]),
            HumanMessage(content=question)
        ]
        response = llm(messages)
        st.success("回答:")
        st.write(response.content)
