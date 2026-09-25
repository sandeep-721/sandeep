from dotenv import load_dotenv

load_dotenv()

from pathlib import Path

from retrieval.hybrid_search import HybridSearch
from generation.context_builder import ContextBuilder
from generation.claim_verifier import ClaimVerifier
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

        self.claim_verifier = ClaimVerifier()

        self.llm = LLMProviderManager()

        self.candidate_limit = candidate_limit
        self.result_limit = result_limit

    MAX_REPAIR_ATTEMPTS = 1

    @staticmethod
    def _result_key(result):
        payload = result.get("payload", {})

        return (
            result.get("id"),
            payload.get("source"),
            payload.get("file_hash"),
            payload.get("chunk_index"),
            payload.get("chunk_start"),
            payload.get("chunk_end"),
        )

    @classmethod
    def _merge_results(cls, primary, repaired):
        merged = []
        seen = set()

        for result in [*primary, *repaired]:
            key = cls._result_key(result)

            if key in seen:
                continue

            seen.add(key)
            merged.append(result)

        return merged

    @staticmethod
    def _verification_quality(verification):
        defect_count = (
            len(verification.invalid_citations)
            + len(verification.uncited_claims)
            + len(verification.weak_support)
        )

        return (
            1 if verification.verified else 0,
            verification.citation_coverage,
            -defect_count,
        )

    @staticmethod
    def _build_repair_query(question, verification):
        claim_ids = set(
            verification.uncited_claims
        ) | set(
            verification.weak_support
        )

        claims = [
            claim.text
            for claim in verification.claims
            if claim.claim_id in claim_ids
        ]

        if not claims:
            return question

        return (
            f"{question}\n\n"
            "Additional evidence retrieval focus:\n"
            + "\n".join(
                f"- {claim}"
                for claim in claims
            )
        )

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

        evidence_packet = (
            self.context_builder.build_packet(
                results,
                query=question,
            )
        )

        evidence = evidence_packet.text
        sources = [
            source.to_dict()
            for source in evidence_packet.sources
        ]

        if evidence_packet.is_empty():
            return {
                "answer": (
                    "The indexed project evidence does not "
                    "contain usable evidence for this question."
                ),
                "sources": [],
                "results": [],
                "evidence_packet": evidence_packet.to_dict(),
                "verification": None,
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

        final_answer = answer.rstrip()

        verification = self.claim_verifier.verify(
            final_answer,
            evidence_packet,
        )

        repair_metadata = {
            "attempts": 0,
            "performed": False,
            "used": False,
            "added_results": 0,
        }

        if (
            not verification.verified
            and self.MAX_REPAIR_ATTEMPTS > 0
        ):
            repair_metadata["attempts"] = 1

            repair_query = self._build_repair_query(
                question,
                verification,
            )

            repair_search_kwargs = dict(
                search_kwargs
            )

            repair_search_kwargs["query"] = (
                repair_query
            )

            repair_search_kwargs["candidate_limit"] = max(
                self.candidate_limit * 2,
                self.result_limit * 4,
            )

            repair_search_kwargs["limit"] = max(
                self.result_limit * 2,
                16,
            )

            repaired_results = self.search.search(
                **repair_search_kwargs
            )

            merged_results = self._merge_results(
                results,
                repaired_results,
            )

            added_results = (
                len(merged_results) - len(results)
            )

            repair_metadata["added_results"] = (
                added_results
            )

            if added_results > 0:
                repair_metadata["performed"] = True

                repaired_packet = (
                    self.context_builder.build_packet(
                        merged_results,
                        query=question,
                    )
                )

                repaired_prompt = (
                    "The previous draft was not fully "
                    "grounded in the available evidence. "
                    "Answer the user's question again using "
                    "ONLY the evidence blocks below.\n\n"
                    "GROUNDING REPAIR RULES:\n"
                    "1. Every substantive technical claim "
                    "must have a valid evidence reference.\n"
                    "2. Use only evidence that explicitly "
                    "supports the claim.\n"
                    "3. Never invent evidence IDs.\n"
                    "4. Do not preserve an unsupported claim "
                    "just because it appeared in the previous "
                    "draft.\n"
                    "5. If the evidence does not establish a "
                    "fact, say exactly: "
                    "\"The indexed project evidence does "
                    "not confirm this.\"\n\n"
                    "EVIDENCE BLOCKS:\n"
                    f"{repaired_packet.text}\n\n"
                    "USER QUESTION:\n"
                    f"{question}\n\n"
                    "Write a concise technical answer with "
                    "inline evidence references."
                )

                repaired_answer = (
                    self.llm.generate(
                        prompt=repaired_prompt,
                        max_new_tokens=128,
                        temperature=0.05,
                    )
                    .rstrip()
                )

                repaired_verification = (
                    self.claim_verifier.verify(
                        repaired_answer,
                        repaired_packet,
                    )
                )

                if (
                    self._verification_quality(
                        repaired_verification
                    )
                    > self._verification_quality(
                        verification
                    )
                ):
                    final_answer = repaired_answer
                    verification = (
                        repaired_verification
                    )
                    results = merged_results
                    evidence_packet = repaired_packet
                    sources = [
                        source.to_dict()
                        for source in (
                            evidence_packet.sources
                        )
                    ]
                    repair_metadata["used"] = True

        return {
            "answer": final_answer,
            "sources": sources,
            "results": results,
            "evidence_packet": evidence_packet.to_dict(),
            "verification": verification.to_dict(),
            "repair": repair_metadata,
        }

    def close(self):
        self.search.close()
        self.llm.close()
        VectorStore.shutdown()
