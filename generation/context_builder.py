from collections import OrderedDict


class ContextBuilder:
    def __init__(self, max_chars=12000):
        self.max_chars = max_chars

    @staticmethod
    def _result_key(result):
        payload = result.get("payload", {})

        source = payload.get("source", "")
        chunk_index = payload.get(
            "chunk_index",
            "",
        )

        return (
            source,
            chunk_index,
            payload.get("chunk_start", ""),
            payload.get("chunk_end", ""),
        )

    @classmethod
    def _deduplicate_results(cls, results):
        unique = []
        seen = set()

        for result in results:
            key = cls._result_key(result)

            if key in seen:
                continue

            seen.add(key)
            unique.append(result)

        return unique

    @staticmethod
    def _group_by_source(results):
        groups = OrderedDict()

        for result in results:
            payload = result.get(
                "payload",
                {},
            )

            source = payload.get(
                "source",
                "unknown",
            )

            groups.setdefault(
                source,
                [],
            ).append(result)

        return groups

    @staticmethod
    def _metadata_lines(payload, result):
        lines = []

        software = payload.get("software")
        version = payload.get(
            "software_version"
        )
        project = payload.get("project")
        language = payload.get("language")

        symbol = payload.get("symbol")
        symbol_kind = payload.get(
            "symbol_kind"
        )

        class_name = payload.get(
            "class_name"
        )
        member_name = payload.get(
            "member_name"
        )
        member_kind = payload.get(
            "member_kind"
        )

        score = result.get(
            "final_rerank_score"
        )

        if score is None:
            score = result.get(
                "rerank_score"
            )

        if score is None:
            score = result.get(
                "score"
            )

        if software:
            lines.append(
                f"Software: {software}"
            )

        if version:
            lines.append(
                f"Version: {version}"
            )

        if project:
            lines.append(
                f"Project: {project}"
            )

        if language:
            lines.append(
                f"Language: {language}"
            )

        if symbol:
            if symbol_kind:
                lines.append(
                    f"Symbol: {symbol} ({symbol_kind})"
                )
            else:
                lines.append(
                    f"Symbol: {symbol}"
                )

        if class_name:
            lines.append(
                f"Class: {class_name}"
            )

        if member_name:
            lines.append(
                f"Member: {member_name}"
            )

        if member_kind:
            lines.append(
                f"Member kind: {member_kind}"
            )

        if score is not None:
            lines.append(
                f"Relevance score: {score:.6f}"
            )

        return lines

    def _build_evidence_block(
        self,
        result,
        evidence_id,
    ):
        payload = result.get(
            "payload",
            {},
        )

        text = payload.get(
            "text",
            "",
        )

        if not text.strip():
            return ""

        source = payload.get(
            "source",
            "unknown",
        )

        chunk_index = payload.get(
            "chunk_index",
            "unknown",
        )

        chunk_start = payload.get(
            "chunk_start"
        )

        chunk_end = payload.get(
            "chunk_end"
        )

        metadata = self._metadata_lines(
            payload,
            result,
        )

        lines = [
            f"[{evidence_id}]",
            f"File: {source}",
            f"Chunk: {chunk_index}",
        ]

        if chunk_start is not None:
            lines.append(
                f"Range: {chunk_start}-{chunk_end}"
            )

        lines.extend(metadata)

        lines.extend(
            [
                "Evidence:",
                text,
            ]
        )

        return "\n".join(lines)

    def build_evidence(self, results):
        """
        Build canonical evidence blocks for LLM grounding.

        Returns:
            {
                "text": str,
                "sources": list[dict],
            }
        """

        if not results:
            return {
                "text": "",
                "sources": [],
            }

        results = self._deduplicate_results(
            results
        )

        groups = self._group_by_source(
            results
        )

        sections = []
        sources = []

        total_chars = 0
        evidence_index = 0
        source_index = 0

        for source, source_results in groups.items():

            source_index += 1

            source_results = sorted(
                source_results,
                key=lambda result: (
                    result.get(
                        "payload",
                        {},
                    ).get(
                        "chunk_index",
                        0,
                    )
                ),
            )

            source_evidence = []

            source_payload = (
                source_results[0].get(
                    "payload",
                    {},
                )
            )

            source_chunks = []

            for result in source_results:

                payload = result.get(
                    "payload",
                    {},
                )

                if not payload.get(
                    "text",
                    "",
                ).strip():
                    continue

                evidence_index += 1

                evidence_id = (
                    f"E{evidence_index}"
                )

                block = self._build_evidence_block(
                    result=result,
                    evidence_id=evidence_id,
                )

                if not block:
                    continue

                projected_size = (
                    total_chars
                    + len(block)
                    + 2
                )

                if projected_size > self.max_chars:
                    break

                source_evidence.append(
                    block
                )

                total_chars = projected_size

                chunk_index = payload.get(
                    "chunk_index"
                )

                if chunk_index is not None:
                    source_chunks.append(
                        chunk_index
                    )

            if not source_evidence:
                continue

            sections.append(
                "\n\n".join(
                    source_evidence
                )
            )

            sources.append(
                {
                    "source": source,
                    "source_index": source_index,
                    "project": source_payload.get(
                        "project"
                    ),
                    "software": source_payload.get(
                        "software"
                    ),
                    "software_version": source_payload.get(
                        "software_version"
                    ),
                    "language": source_payload.get(
                        "language"
                    ),
                    "chunks": sorted(
                        set(source_chunks)
                    ),
                    "chunk_count": len(
                        set(source_chunks)
                    ),
                }
            )

        return {
            "text": "\n\n".join(sections),
            "sources": sources,
        }

    def build(self, results):
        """
        Backward-compatible context builder.

        Returns only the evidence text.
        """

        evidence = self.build_evidence(
            results
        )

        return evidence["text"]

    def build_sources(self, results):
        """
        Return structured source information.
        """

        evidence = self.build_evidence(
            results
        )

        return evidence["sources"]