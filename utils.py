import os
from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.load import dumps, loads
from langsmith import Client


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results

def create_query_generation_chain(langsmith_prompt_name: str):
    client = Client()
    prompt = client.pull_prompt(langsmith_prompt_name)

    generate_queries = (
        prompt | ChatOpenAI(temperature=0) | StrOutputParser() | (lambda x: x.split("\n"))
    )
    return generate_queries