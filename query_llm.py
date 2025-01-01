from config import GPT_MODEL_NAME
from parameters import PERSONA
from typing import Any

def _query_llm(client: Any, persona: str, question: str, example: str =""):
  completion = client.chat.completions.create(
    model=GPT_MODEL_NAME,
    messages=[
      {"role": "system", "content": f"{persona}"},
      {"role": "user", "content": f"{question}"},
      {"role": "assistant", "content": f" You can use the following information as an example: {example}"},
    ],
  )
  return completion.choices[0].message.content


def answer_question(openai_client: Any, best_result: Any, question: str):
    persona=PERSONA+best_result
    return _query_llm(client=openai_client, question=question, persona=persona)
 