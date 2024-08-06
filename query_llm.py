from config import PERSONA, AZURE_MODEL_NAME



def _query_llm(client,persona, question, example=""):
  completion = client.chat.completions.create(
    model=AZURE_MODEL_NAME,
    messages=[
      {"role": "system", "content": f"{persona}"},
      {"role": "user", "content": f"{question}"},
      {"role": "assistant", "content": f" You can use the following information as an example: {example}"},
    ],
  )
  return completion.choices[0].message.content



def answer_question(azure_client, best_result, question):
    persona=PERSONA+best_result
    return _query_llm(client=azure_client, question=question, persona=persona)
 