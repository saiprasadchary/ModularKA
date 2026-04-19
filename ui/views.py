# Filename: ui/views.py

import streamlit as st
# import time # Can be used for simulating delays or just part of debugging
from langchain_core.messages import HumanMessage, AIMessage
import logging
from core.utils import get_logger
from ui.branding import APP_NAME  # import app name for UI use

# Initialize logger
logger = get_logger(__name__)

# def render_qa_view(rag_chain):
#     st.header("❓ Ask Questions about the Paper")
#     st.caption("Ask specific questions and get answers grounded in the paper's content.")
#     ...

def render_qa_view(rag_chain):
    """
    Q&A view for the Modular Knowledge Assistant.

    - Prefers the agentic LangGraph chain (agentic_rag_chain) if available.
    - Falls back to the basic RAG chain (rag_chain) otherwise.
    - Adapts tone based on st.session_state['response_persona']:
        - 'empathetic'  -> emotionally aware, supportive tone
        - 'objective'   -> neutral, fact-focused tone
    """

    # If no chain at all, warn and bail early
    agentic_chain = st.session_state.get("agentic_rag_chain")
    if rag_chain is None and agentic_chain is None:
        st.warning("RAG chain not initialized. Please analyze a paper first.")
        return

    # Decide which chain to use
    using_agentic = agentic_chain is not None
    active_chain = agentic_chain if using_agentic else rag_chain

    # Read persona from session state
    persona = (st.session_state.get("response_persona", "objective") or "objective").lower()
    if persona.startswith("empathetic"):
        persona_label = "Emotionally aware, empathetic"
    else:
        persona_label = "Strictly objective, fact-focused"

    engine_label = "Agentic RAG (LangGraph)" if using_agentic else "Basic RAG (standard RAG chain)"

    # Header / badges
    st.markdown(
        f"**Response style:** {persona_label} &nbsp;&nbsp;|&nbsp;&nbsp; "
        f"**Engine:** {engine_label}"
    )
    st.caption(
        "Ask questions about the analyzed paper below. "
        "Answers will follow the selected response style."
    )

    # Initialize chat history if needed
    if "qa_history" not in st.session_state:
        # List of (role, text) tuples, where role in {'user', 'assistant'}
        st.session_state.qa_history = []

    # Display previous messages
    if st.session_state.qa_history:
        st.markdown("---")
        for role, text in st.session_state.qa_history:
            if role == "user":
                st.markdown(f"**You:** {text}")
            else:
                st.markdown(f"**MKA:** {text}")
        st.markdown("---")

    # Input area
    user_question = st.text_input(
        "Ask a question about this paper:",
        key="qa_question_input",
        placeholder="e.g., What problem does this paper try to solve?"
    )

    send_clicked = st.button("Send", key="qa_send_button")

    if send_clicked and user_question.strip():
        # Append user message
        st.session_state.qa_history.append(("user", user_question))

        with st.spinner("Thinking..."):
            try:
                # --- Agentic LangGraph path ---
                if using_agentic:
                    # LangGraph expects: {"messages": [HumanMessage(...), ...]}
                    result = active_chain.invoke(
                        {"messages": [HumanMessage(content=user_question)]}
                    )

                    # result is usually a dict with a "messages" list (MessagesState)
                    messages = result.get("messages", []) if isinstance(result, dict) else result
                    ai_text = None

                    if isinstance(messages, list):
                        # Find the last AI message
                        for msg in reversed(messages):
                            if getattr(msg, "type", "") == "ai":
                                ai_text = msg.content
                                break

                    if not ai_text:
                        ai_text = (
                            "I’m sorry, I couldn’t generate a complete response. "
                            "Please try rephrasing your question."
                        )

                # --- Basic RAG path ---
                else:
                    # create_retrieval_chain typically returns a dict with "answer"
                    result = active_chain.invoke({"input": user_question})
                    if isinstance(result, dict):
                        ai_text = (
                            result.get("answer")
                            or result.get("output_text")
                            or str(result)
                        )
                    else:
                        ai_text = str(result)

            except Exception as e:
                ai_text = f"An error occurred while generating a response: {e}"

        # Append assistant message and refresh to show it above the input
        st.session_state.qa_history.append(("assistant", ai_text))

        # For modern Streamlit versions, use st.rerun()
        st.rerun()

def render_summary_view(summarize_chain, paper_chunks, user_background):
    """
    Renders the Summary generation interface.

    Args:
        summarize_chain: The initialized LangChain summarization runnable.
        paper_chunks (list[str]): List of text chunks from the paper.
        user_background (str): User's selected background.
    """
    st.header("📄 Paper Summary")
    st.caption("Get a concise overview of the paper tailored to your background.")

    if st.button("Generate Summary ✨", key="summarize_button_view"):
        if summarize_chain and paper_chunks:
            with st.spinner("Generating summary... This may take a moment."):
                try:
                    # Join chunks for the 'stuff' summarizer
                    # TODO: Add logic for map_reduce if implemented later
                    full_text = "\n\n".join(paper_chunks)

                    # Simple length check (replace with proper token counting if needed)
                    # Use a reasonable estimate for context window limits (e.g., ~4k tokens for llama3-8b, ~32k for mixtral)
                    # A character count is a very rough proxy. 1 token ~= 4 chars average.
                    # llama3-8b 8192 tokens * 3 chars/token ~= 24k chars
                    # mixtral 32768 tokens * 3 chars/token ~= 98k chars
                    # Let's use a safer limit for the simpler model
                    max_chars = 20000
                    if len(full_text) > max_chars:
                        st.warning(f"Paper text is quite long ({len(full_text)} chars). Summary might be based on the first ~{max_chars} characters for efficiency.")
                        input_text_for_summary = full_text[:max_chars]
                    else:
                        input_text_for_summary = full_text

                    # Invoke the chain. Ensure the input dict keys match the prompt template.
                    summary = summarize_chain.invoke({
                        "text": input_text_for_summary,
                        "user_background": user_background
                    })
                    st.markdown(summary)
                    st.success("Summary generated!")

                except Exception as e:
                    st.error(f"Error generating summary: {e}")
                    logger.error(f"Summarization error: {e}")
        elif not paper_chunks:
             st.warning("Paper text not available. Please process a paper first.")
        else:
            st.error("Summarization chain is not available.")
    else:
        st.info("Click the button above to generate a summary.")


def render_code_view(code_chain, retriever, user_background):
    """
    Renders the Code Generation interface.

    Args:
        code_chain: The initialized LangChain code generation runnable.
        retriever: The retriever to fetch context relevant to the code request.
        user_background (str): User's selected background.
    """
    st.header("💻 Code Generation")
    st.caption("Generate code snippets based on the paper's content (experimental).")

    code_request = st.text_area(
        "Describe the algorithm, method, or concept you want code for:",
        key="code_request_input_view",
        height=100,
        placeholder="e.g., Implement the data preprocessing steps mentioned in Section 2.1"
    )

    col1, col2 = st.columns(2)
    with col1:
        language = st.selectbox(
            "Language",
            ("Python"),
            key="code_lang_view"
        )
    with col2:
        framework = st.text_input(
            "Framework/Library (if applicable)",
            key="code_framework_view",
            placeholder="e.g., PyTorch, TensorFlow, Scikit-learn, R"
        )

    if st.button("Generate Code ✨", key="code_gen_button_view"):
        if not code_request:
            st.warning("Please describe the code you want to generate.")
        elif code_chain and retriever:
            with st.spinner("Searching context and generating code..."):
                try:
                    # 1. Retrieve context relevant to the request
                    st.write("Finding relevant context...")
                    relevant_docs = retriever.invoke(code_request)
                    if not relevant_docs:
                         st.warning("Could not find specific context in the paper for your request. Proceeding with general knowledge (results may be less accurate).")
                         context_text = "No specific context found in the paper for this request."
                    else:
                         context_text = "\n\n---\n\n".join([f"Source Chunk {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
                         # Optionally show retrieved context
                         # with st.expander("Retrieved Context for Code Generation"):
                         #     st.text(context_text)

                    # 2. Invoke the code chain
                    st.write(f"Generating {language} code...")
                    generated_code = code_chain.invoke({
                        "language": language,
                        "framework": framework if framework else "Not specified",
                        "context": context_text,
                        "description": code_request,
                        "user_background": user_background # Pass user background to agent prompt
                    })

                    # Display code with language formatting
                    st.code(generated_code, language=language.lower() if language != "Other" else "plaintext")
                    st.success("Code generated! (Review carefully before use)")
                    st.caption("⚠️ AI-generated code may contain errors or inaccuracies. Always test thoroughly.")

                except Exception as e:
                    st.error(f"Error generating code: {e}")
                    logger.error(f"Code generation error: {e}")

        elif not retriever:
            st.error("Retriever not available. Please process a paper first.")
        else: # code_chain is None
            st.error("Code generation chain not available.")
    else:
         st.info("Describe the desired code, select language/framework, and click generate.")