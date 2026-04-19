# Filename: ui/components.py

import streamlit as st

def display_sidebar():
    """Render the left sidebar and return user input values."""
    sidebar_values = {}

    with st.sidebar:
        # 1) Paper input group
        with st.expander("📝 Input Paper", expanded=True):
            sidebar_values["source_input"] = st.text_input(
                "Enter arXiv / open-access URL (or DOI)",
                key="source_input_key",
                placeholder="e.g., https://arxiv.org/abs/2401.12345",
            )

            sidebar_values["uploaded_file"] = st.file_uploader(
                "OR Upload a PDF",
                type=["pdf"],
                key="uploaded_file_key",
            )

        st.markdown("---")

        # 2) Background group
        with st.expander("👤 Your Background", expanded=True):
            sidebar_values["user_background"] = st.selectbox(
                "Select your background level:",
                (
                    "Bachelor's Student",
                    "Master's Student",
                    "PhD Student/Researcher",
                    "Industry Professional",
                    "Curious Learner",
                ),
                key="user_background_key",
                index=1,  # default to Master's Student
            )

        st.markdown("---")

        with st.expander("💬 Response Style", expanded=True):
            current_persona = st.session_state.get("response_persona", "objective")

            options = (
                "Emotionally aware, empathetic",
                "Strictly objective, fact-focused",
            )
            default_index = 0 if current_persona == "empathetic" else 1

            persona_label = st.radio(
                "How should the assistant respond?",
                options=options,
                index=default_index,
                key="response_persona_radio",
            )

            # Map human-readable label → internal code
            if persona_label.startswith("Emotionally"):
                persona_value = "empathetic"
            else:
                persona_value = "objective"

            st.session_state["response_persona"] = persona_value
            sidebar_values["response_persona"] = persona_value

            st.caption(
                "Mode: **empathetic** for emotionally aware answers, "
                "or **objective** for strictly fact-driven responses."
            )

        st.markdown("---")

        # 3) Model provider group
        with st.expander("⚙️ Model Provider", expanded=True):
            # Current provider comes from session_state (default is set in app.py)
            current_provider = st.session_state.get("llm_provider", "Groq")
            if current_provider not in ("Groq", "Ollama"):
                current_provider = "Groq"

            provider_choice = st.radio(
                "Select LLM provider:",
                options=("Groq", "Ollama"),
                index=0 if current_provider == "Groq" else 1,
                key="llm_provider_radio",
            )

            # Persist selection globally for this Streamlit session
            st.session_state["llm_provider"] = provider_choice
            sidebar_values["llm_provider"] = provider_choice

            st.caption(f"🔌 Active provider: **{provider_choice}**")

        st.markdown("---")

        # 4) Action button
        sidebar_values["process_clicked"] = st.button(
            "Analyze Paper ✨",
            key="process_button_key",
        )

        st.caption(
            "Tip: Direct PDF links (or arXiv URLs) work best for clean parsing."
        )

    return sidebar_values

# You can add more reusable components here later, e.g.:
# def display_chat_interface(...)
# def display_summary(...)
# def display_code(...)