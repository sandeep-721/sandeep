from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from retrieval.hybrid_search import HybridSearch
from generation.context_builder import ContextBuilder
from generation.provider_manager import LLMProviderManager
from retrieval.vector_store import VectorStore


class RAG:
    def __init__(
        self,
        project_root=None,
        candidate_limit=30,
        result_limit=8,
        max_context_chars=16000,
    ):
        self.project_root = (
            Path(project_root).resolve()
            if project_root
            else None
        )

        self.search = HybridSearch()

        self.context_builder = ContextBuilder(
            max_chars=max_context_chars
        )

        self.llm = LLMProviderManager()

        self.candidate_limit = candidate_limit
        self.result_limit = result_limit

    def ask(self, question, project=None):
        search_kwargs = {
            "query": question,
            "limit": self.result_limit,
            "candidate_limit": self.candidate_limit,
        }

        if project:
            search_kwargs["project"] = project
        elif self.project_root:
            search_kwargs["project"] = self.project_root.name

        results = self.search.search(
            **search_kwargs
        )

        if not results:
            return {
                "answer": (
                    "The indexed project evidence does not "
                    "contain relevant information needed to "
                    "answer this question."
                ),
                "sources": [],
                "results": [],
            }

        evidence_data = (
            self.context_builder.build_evidence(
                results
            )
        )

        evidence = evidence_data["text"]
        sources = evidence_data["sources"]

        if not evidence:
            return {
                "answer": (
                    "The indexed project evidence does not "
                    "contain usable evidence for this question."
                ),
                "sources": [],
                "results": [],
            }

        prompt = (
            "Answer the user's question using ONLY the "
            "evidence blocks below.\n\n"

            "GROUNDING RULES:\n"
            "1. Use only information explicitly supported "
            "by the evidence blocks.\n"
            "2. Every technical claim must have an evidence "
            "reference such as [E1] or [E2].\n"
            "3. Put the evidence reference immediately "
            "after the claim it supports.\n"
            "4. Evidence references must correspond exactly "
            "to the supplied evidence blocks.\n"
            "5. Never invent evidence IDs.\n"
            "6. Never invent code, methods, fields, classes, "
            "control flow, behavior, or implementation details.\n"
            "7. Do not turn a method name into an explanation "
            "of its behavior unless the evidence shows "
            "that behavior.\n"
            "8. Do not claim that something is guaranteed, "
            "correct, efficient, dynamic, safe, reusable, "
            "or suitable unless the evidence explicitly "
            "supports that characterization.\n"
            "9. Do not use outside knowledge.\n"
            "10. Do not infer missing implementation details.\n"
            "11. Do not use words such as 'likely', "
            "'probably', 'appears', or 'presumably' to fill "
            "missing information.\n"
            "12. If the evidence does not establish a fact, "
            "say exactly: \"The indexed project evidence "
            "does not confirm this.\"\n"
            "13. Do not create a Sources section. Python "
            "will generate the source list separately.\n"
            "14. When describing code, prefer the actual "
            "class, method, field, and variable names from "
            "the evidence.\n"
            "15. Do not reproduce code unless the supplied "
            "evidence contains the code needed for the "
            "specific explanation.\n\n"

            "EVIDENCE BLOCKS:\n"
            f"{evidence}\n\n"

            "USER QUESTION:\n"
            f"{question}\n\n"

            "Write a concise technical answer. "
            "Every substantive technical statement must "
            "have an inline evidence reference."
        )

        answer = self.llm.generate(
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.05,
        )

        return {
            "answer": answer.rstrip(),
            "sources": sources,
            "results": results,
        }

    def close(self):
        self.search.close()
        self.llm.close()
        VectorStore.shutdown()