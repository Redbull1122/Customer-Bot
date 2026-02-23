import streamlit as st
from pathlib import Path
from langchain_core.memory import ConversationBufferMemory


from src.rag_chain import generate_answer


# Page configuration (set once at module load)
st.set_page_config(
    page_title="Hajster – Customer Bot",
    page_icon="favicon.ico",
    layout="wide",
)


def _get_logo_path() -> str | None:
    """Return the logo path if the file exists."""
    candidates = [
        Path("data/pdf/logo.svg"),
        Path("app/hajster_logo.png"),
        Path("hajster_logo.png"),
        Path("data/images/hajster_logo.png"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Chat history for the UI
    
    if "memory" not in st.session_state:
        # Create memory for chat history (k=5 keeps the last 5 exchanges)
        st.session_state.memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="history",
            return_messages=False
        )
    # Custom styling inspired by hajster.com (without wrapping widgets in raw HTML)
    st.markdown(
        """
        <style>
        /* ===== BASE ===== */
        .block-container {
            padding-top: 0rem;
            padding-left: 2rem;
            padding-right: 2rem;
            background-color: #f5f7fb;
            font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", sans-serif;
        }

        html, body, [class*="st-"] {
            color: #0f172a !important;
        }

        h1, h2, h3, h4 {
            color: #0b1f3b !important;
        }

        .stMarkdown p, .stMarkdown li {
            color: #1e293b !important;
            font-size: 0.95rem;
        }

        /* ===== HERO ===== */
        .hajster-hero {
            background: linear-gradient(135deg, #0b1f3b 0%, #13294b 60%, #ff7a00 140%);
            border-radius: 0 0 18px 18px;
            padding: 1.4rem 2.4rem;
            color: #ffffff;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.5rem;
        }

        .hajster-subtitle {
            font-size: 0.95rem;
            opacity: 0.95;
            color: #e5e7eb !important;
        }

        .hajster-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.8rem;
            border-radius: 999px;
            background: #ffefe0;
            color: #b45309;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        /* ===== CARD ===== */
        .hajster-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 1.6rem 1.9rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(15, 23, 42, 0.05);
            color: #0f172a;
        }

        /* ===== INPUT ===== */
        label {
            color: #0b1f3b !important;
            font-weight: 600;
        }

        input[type="text"], textarea {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
            border-radius: 12px !important;
            padding: 0.6rem 0.8rem !important;
            font-size: 0.95rem !important;
        }

        input::placeholder, textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        /* ===== BUTTON ===== */
        button[kind="primary"] {
            background: linear-gradient(135deg, #ff7a00, #ff9f43) !important;
            color: #ffffff !important;
            border-radius: 999px !important;
            font-weight: 700 !important;
            padding: 0.45rem 1.4rem !important;
            box-shadow: 0 8px 20px rgba(255, 122, 0, 0.35) !important;
        }

        button[kind="primary"]:hover {
            filter: brightness(1.05);
        }

        hr {
            border-top: 1px solid #e2e8f0;
        }

        .hajster-footer {
            text-align: center;
            font-size: 0.8rem;
            color: #64748b;
            padding: 1.5rem 0 0.5rem 0;
        }

        /* ===== CHAT MESSAGES ===== */
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .chat-message {
            padding: 1rem 1.2rem;
            border-radius: 16px;
            animation: slideIn 0.4s ease-out;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        }

        .user-message {
            background: linear-gradient(135deg, #ff7a00, #ff9f43);
            color: #ffffff;
            margin-left: 15%;
            border-bottom-right-radius: 4px;
        }

        .bot-message {
            background: #ffffff;
            color: #0f172a;
            margin-right: 15%;
            border: 1px solid #e2e8f0;
            border-bottom-left-radius: 4px;
        }

        .chat-message h4 {
            margin: 0 0 0.5rem 0;
            font-size: 0.85rem;
            opacity: 0.8;
        }

        .chat-message p {
            margin: 0;
            line-height: 1.5;
        }

        /* ===== IMAGE ANIMATIONS ===== */
        .image-container {
            animation: fadeInUp 0.6s ease-out;
            margin: 0.5rem 0;
        }

        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Staggered delay for images */
        .image-container:nth-child(1) { animation-delay: 0.1s; opacity: 0; animation-fill-mode: forwards; }
        .image-container:nth-child(2) { animation-delay: 0.2s; opacity: 0; animation-fill-mode: forwards; }
        .image-container:nth-child(3) { animation-delay: 0.3s; opacity: 0; animation-fill-mode: forwards; }
        .image-container:nth-child(4) { animation-delay: 0.4s; opacity: 0; animation-fill-mode: forwards; }
        .image-container:nth-child(5) { animation-delay: 0.5s; opacity: 0; animation-fill-mode: forwards; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Top block with logo and description (Streamlit columns, no raw widget wrappers)
    hero = st.container()
    with hero:
        st.markdown('<div class="hajster-hero">', unsafe_allow_html=True)
        col_logo, col_title, col_pill = st.columns([1, 3, 1])

        logo_path = _get_logo_path()
        with col_logo:
            if logo_path:
                if logo_path.endswith(".svg"):
                    # Render SVG directly
                    svg_content = Path(logo_path).read_text(encoding="utf-8")
                    st.markdown(svg_content, unsafe_allow_html=True)
                else:
                    st.image(logo_path, use_column_width=True)
            else:
                st.markdown('<div class="hajster-logo-text">hajster</div>', unsafe_allow_html=True)


        with col_pill:
            st.markdown('<div style="text-align: right;"><span class="hajster-pill">Customer Bot</span></div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Main content
    col_left, col_right = st.columns([2, 1])

    # Left column: chat
    with col_left:
        st.markdown("### Chat with Assistant")
        
        # Chat history container
        chat_container = st.container()
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Render full message history
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="chat-message user-message">'
                        f'<h4>You</h4>'
                        f'<p>{msg["content"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:  # bot
                    st.markdown(
                        f'<div class="chat-message bot-message">'
                        f'<h4>Assistant</h4>'
                        f'<p>{msg["content"]}</p>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Show sources
                    if msg.get("sources"):
                        with st.expander("Sources"):
                            for src in msg["sources"]:
                                page = src.get("page", "N/A")
                                source = src.get("source", "")
                                st.markdown(f"- **Page**: {page}, **File**: `{source}`")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Input field at the bottom
        user_query = st.text_input(
            "Your question",
            placeholder="For example: How do I check the system status and operating parameters?",
            key="user_query",
        )
        
        col_send, col_clear = st.columns([3, 1])
        
        with col_send:
            send_button = st.button("Send", type="primary", use_container_width=True)
        
        with col_clear:
            clear_button = st.button("Clear", use_container_width=True)
        
        # Handle message submission
        if send_button and user_query.strip():
            # Append user message
            st.session_state.messages.append({
                "role": "user",
                "content": user_query
            })
            
            with st.spinner("Searching for an answer..."):
                try:
                    # Generate answer with memory
                    result = generate_answer(
                        query_text=user_query, 
                        top_k=8,
                        memory=st.session_state.memory
                    )
                    
                    # Append assistant answer
                    st.session_state.messages.append({
                        "role": "bot",
                        "content": result.get("answer", ""),
                        "sources": result.get("sources", [])
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        # Clear history
        if clear_button:
            st.session_state.messages = []
            # Reset memory
            st.session_state.memory = ConversationBufferWindowMemory(
                k=5,
                memory_key="history",
                return_messages=False
            )
            st.rerun()

    # Right column: info block
    with col_right:
        st.markdown("#### How the Assistant Works")
        st.markdown(
            """
            1. You ask a question related to controller operation.  
            2. The assistant retrieves relevant fragments from the official Hajster manual.  
            3. If related diagrams or images exist in the manual, they can also be shown.  
            """
        )

    st.markdown(
        '<div class="hajster-footer">© 2026 Hajster · Customer Bot demo interface</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
