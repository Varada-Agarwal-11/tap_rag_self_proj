from langchain_core.runnables import RunnableLambda

from src.query_parser import parse_query


class LangChainTouristPipeline:
    def __init__(self, rag):
        self.rag = rag
        self.chain = (
            RunnableLambda(self._parse)
            | RunnableLambda(self._retrieve)
            | RunnableLambda(self._build_prompt)
            | RunnableLambda(self._generate)
        )

    def _parse(self, query: str):
        return {"query": query, "intent": parse_query(query)}

    def _retrieve(self, state):
        grouped = self.rag.retriever.retrieve(state["intent"])
        ranked = self.rag.reranker.rank(state["query"], grouped)
        state["grouped"] = grouped
        state["ranked"] = ranked
        return state

    def _build_prompt(self, state):
        state["prompt"] = self.rag.build_generation_prompt(
            state["query"],
            {"intent": state["intent"], "ranked": state["ranked"]},
        )
        return state

    def _generate(self, state):
        state["answer"] = self.rag.model.generate(state["prompt"])
        return state

    def invoke(self, query: str):
        return self.chain.invoke(query)
