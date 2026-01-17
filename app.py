import streamlit as st
from agent.agent import agent_answer
from utils.logger import setup_logger

logger = setup_logger("app")


st.title("📚 Agentic RAG Assistant")

# -----------------------------
# Session state initialization
# -----------------------------
for key in ["question", "answer", "evaluation", "actions"]:
    if key not in st.session_state:
        st.session_state[key] = None

# -----------------------------
# User input
# -----------------------------
question = st.text_input(
    "Ask a question about the project documentation",
    value=st.session_state.get("question", ""),
    key="question_input"
)

try:
    # -----------------------------
    # Buttons
    # -----------------------------
    ask_button = st.button("Ask")

    if "app_started" not in st.session_state:
        logger.info("App started")
        st.session_state["app_started"] = True

    if ask_button and question.strip():
        # ----------------------
        # Clear previous state
        # ----------------------
        logger.info("clear")
        st.session_state.question = question # save new question
        st.session_state.answer = None
        st.session_state.evaluation = None
        st.session_state.actions = None

        if question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("Agent is reasoning..."):
                try:
                    logger.info(f"User question submitted: {question}")
                    answer, evaluation, actions = agent_answer(question)
                    st.session_state.answer = answer
                    st.session_state.evaluation = evaluation
                    st.session_state.actions = actions

                except Exception as e:
                    logger.exception("Unhandled error in app")
                    st.error("An internal error occurred.")
                    st.exception(e)

    # -----------------------------
    # Display answer
    # -----------------------------
    if st.session_state.answer is not None:
        st.subheader("🧠 Answer")
        st.write(st.session_state.answer)

    # -----------------------------
    # Display agent reflection
    # -----------------------------
    if st.session_state.evaluation is not None:
        st.subheader("🔍 Agent Self-Reflection")
        st.json({
            "evaluation": st.session_state.evaluation,
            "actions_taken": st.session_state.actions
        })

except Exception as e:
    st.error("Something went wrong while answering your question.")
    st.exception(e)
