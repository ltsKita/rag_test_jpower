import streamlit as st
from app_rag_system import get_qa_response

st.title("RAGチャットボット")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for i, message in enumerate(st.session_state.messages):  # iを使ってユニークなキーを生成
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("参照情報"):
                # 履歴の参照元にも一意のキーを付与
                st.text_area(f"参照元_回答ID:{i // 2 + 1}", message["sources"], height=300)
                # CSSを使ってtext_areaの横幅を調整
                st.markdown(
                    """
                    <style>
                    .stTextArea textarea {
                        width: 100% !important;
                        white-space: pre-wrap !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

# ユーザー入力
if prompt := st.chat_input("質問を入力してください:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ボットの応答
    with st.chat_message("assistant"):
        response = get_qa_response(prompt)
        st.markdown(response["answer"])
        st.session_state.messages.append(
            {"role": "assistant", "content": response["answer"], "sources": response["sources"]}
        )

    # 参照情報を表示
    with st.expander("参照情報"):
        st.text_area("参照元", response["sources"], height=300)
        # CSSを使ってtext_areaの横幅を調整
        st.markdown(
            """
        <style>
        .stTextArea textarea {
            width: 100% !important;
            white-space: pre-wrap !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

# サイドバーにチャット履歴のクリアボタンを追加
if st.sidebar.button("チャット履歴をクリア"):
    st.session_state.messages = []
    st.rerun()
