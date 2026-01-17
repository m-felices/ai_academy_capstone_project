from llm.client import generate_answer, evaluate_answer, rewrite_answer
from rag.retriever import retrieve_docs
from utils.logger import setup_logger

logger = setup_logger("agent")

def agent_answer(question: str):
    actions = []
    try:
        logger.info("Agent invoked")
        logger.info(f"Question: {question}")

        # 1. Retrieve
        docs = retrieve_docs(question, k=3)
        context = "\n".join(d.page_content for d in docs)

        if not docs:
            actions.append("NO_CONTEXT")
            logger.warning("No context found")
            return "I don't know.", None, actions

        # 2️. Answer
        answer = generate_answer(question, context)
        logger.info("Answer generated")

        # 3️. Self-evaluate
        evaluation = evaluate_answer(question, context, answer)
        logger.info(f"Evaluation result: {evaluation}")
        evaluation = {k: int(v) for k, v in evaluation.items()}

        # 4️. Reflect & act
        if evaluation["faithfulness"] <= 3:
            actions.append("ABSTAIN")
            logger.warning("Low faithfulness, abstention response")
            return "I don't know.", evaluation, actions

        if evaluation["relevance"] <= 2:
            actions.append("RETRIEVE_MORE")
            logger.warning("Low relevance, retrieve more info")
            docs = retrieve_docs(question, k=6)
            context = "\n".join(d.page_content for d in docs)
            answer = generate_answer(question, context)

        if evaluation["clarity"] <= 2:
            actions.append("REWRITE")
            logger.warning("Low faithfulness, rewriting answer")
            answer = rewrite_answer(answer)

        if not actions:
            actions = ["ANSWER_ACCEPTED"]
            logger.info("Answer accepted")

        return answer, evaluation, actions
    except Exception as e:
        actions.append("AGENT_ERROR")
        return f"An internal error occurred: {e}", None, actions
