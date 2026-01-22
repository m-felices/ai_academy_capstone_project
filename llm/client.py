import json
import re
from agent.prompts import ANSWER_PROMPT, EVAL_PROMPT, REWRITE_PROMPT
from openai import OpenAI

from dotenv import load_dotenv
from utils.logger import setup_logger
logger = setup_logger("client")


load_dotenv()
client = OpenAI()
def generate_answer(question: str, context: str) -> str:
    logger.info("Generating answer")
    prompt = ANSWER_PROMPT.format(
        context=context,
        question=question
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


def evaluate_answer(question: str, context: str, answer: str)-> dict[str, int]:
    logger.info("Evaluating answer")
    prompt = EVAL_PROMPT.format(
        question=question,
        context=context,
        answer=answer
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=150
    )

    raw = response.choices[0].message.content.strip()

    # Strip ```json ... ``` or ``` ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    logger.debug(f"Raw evaluator output: {raw}")

    # Parse as JSON
    evaluation = json.loads(raw)
    logger.info(f"Parsed evaluation: {evaluation}")
    return evaluation


def rewrite_answer(answer: str) -> str:
    prompt = REWRITE_PROMPT.format(
        answer=answer
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()
