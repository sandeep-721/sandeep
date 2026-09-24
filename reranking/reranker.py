import re

from sentence_transformers import CrossEncoder

from config.settings import RERANKER_MODEL, DEVICE

from retrieval.lexical_index import LexicalIndex
from retrieval.query_intent import QueryIntentDetector





class Reranker:

    CROSS_WEIGHT = 0.70

    LEXICAL_WEIGHT = 0.55

    STRUCTURAL_WEIGHT = 0.45

    ANSWER_EVIDENCE_WEIGHT = 0.50

    RRF_WEIGHT = 0.20

    SOURCE_AUTHORITY_WEIGHT = 0.20



    EXACT_SYMBOL_BONUS = 1.50

    EXACT_FILENAME_BONUS = 1.20

    STRUCTURAL_COMPONENT_BONUS = 0.20



    # Structural evidence is intentionally reduced for broad

    # natural-language queries. Explicit identifier queries retain

    # the full structural weight.

    NATURAL_LANGUAGE_STRUCTURAL_WEIGHT = 0.08



    MAX_RERANK_TEXT_CHARS = 6000



    def __init__(self):

        self.model = CrossEncoder(RERANKER_MODEL, device=DEVICE)



    def score(self, query: str, document: str) -> float:

        score = self.model.predict(

            [(query, document)],

            show_progress_bar=False,

        )

        return float(score[0])



    @staticmethod

    def _query_tokens(query: str) -> set[str]:

        if not query:

            return set()



        return {

            token.lower()

            for token in LexicalIndex.tokenize(query)

            if len(token) >= 2

        }



    @staticmethod

    def _query_identifiers(query: str) -> set[str]:

        if not query:

            return set()



        identifiers = re.findall(

            r"\b[A-Za-z_][A-Za-z0-9_]*\b",

            query,

        )



        return {

            identifier.lower()

            for identifier in identifiers

            if len(identifier) >= 3

        }



    @staticmethod

    def _is_explicit_identifier_query(query: str) -> bool:

        if not query:

            return False



        normalized = query.strip()



        if not normalized:

            return False



        normalized = normalized.strip("\"'`.,:;()[]{}")



        if re.fullmatch(

            r"[A-Za-z_][A-Za-z0-9_.-]*",

            normalized,

        ):

            return True



        tokens = normalized.split()



        if not tokens or len(tokens) > 3:

            return False



        return all(

            re.fullmatch(

                r"[A-Za-z_][A-Za-z0-9_.-]*",

                token.strip("\"'`.,:;()[]{}"),

            )

            for token in tokens

        )



    @staticmethod

    def _normalized_value(value) -> str:

        if value is None:

            return ""



        return str(value).strip().lower()



    @classmethod

    def _filename_stem(cls, path) -> str:

        if not path:

            return ""



        normalized = str(path).replace("\\", "/")

        filename = normalized.rsplit("/", 1)[-1]



        if "." in filename:

            filename = filename.rsplit(".", 1)[0]



        return filename.lower()



    @classmethod

    def _lexical_evidence(

        cls,

        query: str,

        payload: dict,

    ) -> float:

        query_tokens = cls._query_tokens(query)



        if not query_tokens:

            return 0.0



        text_fields = [

            payload.get("text"),

            payload.get("source"),

            payload.get("path"),

            payload.get("symbol"),

            payload.get("symbol_kind"),

            payload.get("class_name"),

            payload.get("member_name"),

            payload.get("member_kind"),

            payload.get("chunk_type"),

            payload.get("language"),

            payload.get("software"),

            payload.get("software_version"),

            payload.get("project"),

            payload.get("repository"),

            payload.get("branch"),

        ]



        combined_text = " ".join(

            str(value)

            for value in text_fields

            if value

        )



        document_tokens = set(

            LexicalIndex.tokenize(combined_text)

        )



        if not document_tokens:

            return 0.0



        overlap = query_tokens & document_tokens



        if not overlap:

            return 0.0



        coverage = len(overlap) / max(len(query_tokens), 1)



        return min(coverage, 1.0)



    @classmethod

    def _answer_evidence(

        cls,

        query: str,

        payload: dict,

    ) -> float:

        query_tokens = cls._query_tokens(query)



        if not query_tokens:

            return 0.0



        body = str(payload.get("text", ""))



        body_tokens = set(

            LexicalIndex.tokenize(body)

        )



        if not body_tokens:

            return 0.0



        overlap = query_tokens & body_tokens



        if not overlap:

            return 0.0



        body_coverage = len(overlap) / max(

            len(query_tokens),

            1,

        )



        metadata_fields = [

            payload.get("source"),

            payload.get("path"),

            payload.get("symbol"),

            payload.get("class_name"),

            payload.get("member_name"),

            payload.get("chunk_type"),

        ]



        metadata_text = " ".join(

            str(value)

            for value in metadata_fields

            if value

        )



        metadata_tokens = set(

            LexicalIndex.tokenize(metadata_text)

        )



        metadata_overlap = query_tokens & metadata_tokens



        metadata_coverage = len(metadata_overlap) / max(

            len(query_tokens),

            1,

        )



        score = (

            0.70 * body_coverage

            + 0.30 * metadata_coverage

        )



        return min(score, 1.0)



    @classmethod

    def _structural_evidence(

        cls,

        query: str,

        payload: dict,

    ) -> dict:

        identifiers = cls._query_identifiers(query)



        if not identifiers:

            return {

                "score": 0.0,

                "tier": 0,

                "exact": False,

                "exact_symbol": False,

                "exact_filename": False,

            }



        symbol = cls._normalized_value(

            payload.get("symbol")

        )



        class_name = cls._normalized_value(

            payload.get("class_name")

        )



        member_name = cls._normalized_value(

            payload.get("member_name")

        )



        source = payload.get("source", "")

        path = payload.get("path", "")



        filename_stem = cls._filename_stem(

            source or path

        )



        exact_symbol = (

            bool(symbol)

            and symbol in identifiers

        )



        exact_class = (

            bool(class_name)

            and class_name in identifiers

        )



        exact_member = (

            bool(member_name)

            and member_name in identifiers

        )



        exact_filename = (

            bool(filename_stem)

            and filename_stem in identifiers

        )



        score = 0.0

        tier = 0



        if exact_symbol:

            score += 1.00

            tier = max(tier, 4)



        if exact_class:

            score += 0.95

            tier = max(tier, 4)



        if exact_member:

            score += 0.95

            tier = max(tier, 4)



        if exact_filename:

            score += 0.85

            tier = max(tier, 3)



        structural_values = [

            symbol,

            class_name,

            member_name,

            filename_stem,

        ]



        for value in structural_values:

            if not value:

                continue



            value_tokens = set(

                LexicalIndex.tokenize(value)

            )



            if not value_tokens:

                continue



            for identifier in identifiers:

                identifier_tokens = set(

                    LexicalIndex.tokenize(identifier)

                )



                if not identifier_tokens:

                    continue



                overlap = (

                    value_tokens

                    & identifier_tokens

                )



                if overlap:

                    score += min(

                        0.10 * len(overlap),

                        0.20,

                    )



                    tier = max(tier, 2)



        return {

            "score": score,

            "tier": tier,

            "exact": tier >= 3,

            "exact_symbol": (

                exact_symbol

                or exact_class

                or exact_member

            ),

            "exact_filename": exact_filename,

        }



    @classmethod

    def _build_reranker_text(

        cls,

        payload: dict,

    ) -> str:

        text = str(

            payload.get("text", "")

        )



        if len(text) > cls.MAX_RERANK_TEXT_CHARS:

            text = text[:cls.MAX_RERANK_TEXT_CHARS]



        structural_prefix = []



        source = payload.get("source")



        if source:

            structural_prefix.append(

                f"Source: {source}"

            )



        path = payload.get("path")



        if path and path != source:

            structural_prefix.append(

                f"Path: {path}"

            )



        class_name = payload.get("class_name")



        if class_name:

            structural_prefix.append(

                f"Class: {class_name}"

            )



        member_name = payload.get("member_name")



        if member_name:

            structural_prefix.append(

                f"Member: {member_name}"

            )



        member_kind = payload.get("member_kind")



        if member_kind:

            structural_prefix.append(

                f"Member kind: {member_kind}"

            )



        symbol = payload.get("symbol")



        if (

            symbol

            and symbol != class_name

            and symbol != member_name

        ):

            structural_prefix.append(

                f"Symbol: {symbol}"

            )



        chunk_type = payload.get("chunk_type")



        if chunk_type:

            structural_prefix.append(

                f"Chunk type: {chunk_type}"

            )



        language = payload.get("language")



        if language:

            structural_prefix.append(

                f"Language: {language}"

            )



        software = payload.get("software")



        if software:

            structural_prefix.append(

                f"Software: {software}"

            )



        if structural_prefix:

            return (

                "\n".join(structural_prefix)

                + "\n\n"

                + text

            )



        return text



    @staticmethod

    def _normalize_scores(

        scores,

    ) -> list[float]:

        values = [

            float(score)

            for score in scores

        ]



        if not values:

            return []



        minimum = min(values)

        maximum = max(values)



        if maximum - minimum < 1e-8:

            return [

                0.5

                for _ in values

            ]



        return [

            (value - minimum)

            / (maximum - minimum)

            for value in values

        ]



    QUERY_INTENT_SOURCE_AUTHORITY = {

        "code": {

            "rag_code": 0.60,

            "project_code": 0.55,

            "rag_documentation": 0.08,

            "project_documentation": 0.05,

            "rag_configuration": 0.08,

            "project_configuration": 0.05,

            "rag_history": 0.03,

            "project_history": 0.03,

            "rag_test": -0.20,

            "project_test": -0.15,

            "third_party_documentation": 0.00,

            "third_party_code": 0.03,

        },

        "documentation": {

            "rag_documentation": 0.70,

            "project_documentation": 0.65,

            "third_party_documentation": 0.30,

            "rag_code": 0.10,

            "project_code": 0.10,

            "rag_configuration": 0.10,

            "project_configuration": 0.08,

            "rag_history": 0.08,

            "project_history": 0.08,

            "rag_test": -0.05,

            "project_test": -0.05,

            "third_party_code": 0.03,

        },

        "configuration": {

            "rag_configuration": 0.90,

            "project_configuration": 0.80,

            "rag_code": 0.20,

            "project_code": 0.15,

            "rag_documentation": 0.15,

            "project_documentation": 0.12,

            "rag_history": 0.05,

            "project_history": 0.05,

            "rag_test": -0.10,

            "project_test": -0.10,

            "third_party_documentation": 0.05,

            "third_party_code": 0.05,

        },

        "history": {

            "rag_history": 0.75,

            "project_history": 0.70,

            "rag_documentation": 0.15,

            "project_documentation": 0.12,

            "rag_code": 0.08,

            "project_code": 0.08,

            "rag_configuration": 0.05,

            "project_configuration": 0.05,

            "rag_test": -0.05,

            "project_test": -0.05,

            "third_party_documentation": 0.03,

            "third_party_code": 0.00,

        },

        "test": {

            "rag_test": 0.90,

            "project_test": 0.85,

            "rag_code": 0.20,

            "project_code": 0.15,

            "rag_documentation": 0.10,

            "project_documentation": 0.10,

            "rag_configuration": 0.10,

            "project_configuration": 0.08,

            "rag_history": 0.03,

            "project_history": 0.03,

            "third_party_documentation": 0.03,

            "third_party_code": 0.03,

        },

        "general": {

            "rag_configuration": 0.20,

            "rag_code": 0.20,

            "rag_documentation": 0.15,

            "project_configuration": 0.18,

            "project_code": 0.15,

            "project_documentation": 0.15,

            "rag_history": 0.10,

            "project_history": 0.10,

            "third_party_documentation": 0.05,

            "third_party_code": 0.05,

            "rag_test": -0.15,

            "project_test": -0.10,

        },

    }



    @staticmethod
    def _query_intent(query: str) -> str:
        return QueryIntentDetector.detect(query)

    @classmethod

    def _source_authority(

        cls,

        payload: dict,

        query_intent: str = "general",

    ) -> float:

        source_type = str(

            payload.get("source_type", "")

        ).strip().lower()



        priors = cls.QUERY_INTENT_SOURCE_AUTHORITY.get(

            query_intent,

            cls.QUERY_INTENT_SOURCE_AUTHORITY["general"],

        )



        return priors.get(source_type, 0.0)



    @staticmethod

    def _rrf_rank_score(

        rank: int,

        total: int,

    ) -> float:

        if total <= 1:

            return 1.0



        return 1.0 - (

            (rank - 1)

            / (total - 1)

        )



    def rerank(

        self,

        query: str,

        documents: list[dict],

        limit: int = 5,

    ):

        if not documents:

            return []



        pairs = []



        for document in documents:

            payload = document.get(

                "payload",

                {},

            )



            pairs.append(

                (

                    query,

                    self._build_reranker_text(

                        payload

                    ),

                )

            )



        cross_encoder_scores = self.model.predict(

            pairs,

            show_progress_bar=False,

        )



        normalized_cross_scores = (

            self._normalize_scores(

                cross_encoder_scores

            )

        )



        rrf_ranked = sorted(

            documents,

            key=lambda item: item.get(

                "score",

                0.0,

            ),

            reverse=True,

        )



        rrf_rank_by_id = {

            document.get("id"): rank

            for rank, document

            in enumerate(

                rrf_ranked,

                start=1,

            )

        }



        explicit_identifier_query = (

            self._is_explicit_identifier_query(

                query

            )

        )



        scored = []



        total_documents = len(documents)



        structural_weight = (

            self.STRUCTURAL_WEIGHT

            if explicit_identifier_query

            else self.NATURAL_LANGUAGE_STRUCTURAL_WEIGHT

        )



        for index, (

            document,

            raw_cross_score,

        ) in enumerate(

            zip(

                documents,

                cross_encoder_scores,

            )

        ):

            payload = document.get(

                "payload",

                {},

            )



            point_id = document.get("id")



            rrf_rank = rrf_rank_by_id.get(

                point_id,

                total_documents,

            )



            rrf_score = self._rrf_rank_score(

                rrf_rank,

                total_documents,

            )



            cross_score = (

                normalized_cross_scores[index]

            )



            lexical_score = self._lexical_evidence(

                query=query,

                payload=payload,

            )



            answer_score = self._answer_evidence(

                query=query,

                payload=payload,

            )



            structural = self._structural_evidence(

                query=query,

                payload=payload,

            )



            structural_score = structural["score"]

            query_intent = self._query_intent(query)



            source_authority = self._source_authority(

                payload=payload,

                query_intent=query_intent,

            )



            final_score = (

                self.CROSS_WEIGHT

                * cross_score

                + self.LEXICAL_WEIGHT

                * lexical_score

                + structural_weight

                * structural_score

                + self.ANSWER_EVIDENCE_WEIGHT

                * answer_score

                + self.RRF_WEIGHT

                * rrf_score

                + self.SOURCE_AUTHORITY_WEIGHT

                * source_authority

            )



            if explicit_identifier_query:

                if structural["exact_symbol"]:

                    final_score += (

                        self.EXACT_SYMBOL_BONUS

                    )



                elif structural["exact_filename"]:

                    final_score += (

                        self.EXACT_FILENAME_BONUS

                    )



                elif structural["tier"] == 2:

                    final_score += (

                        self.STRUCTURAL_COMPONENT_BONUS

                    )



            if answer_score >= 0.75:

                final_score += 0.20



            elif answer_score >= 0.50:

                final_score += 0.10



            scored.append(

                {

                    **document,

                    "rerank_score": float(

                        raw_cross_score

                    ),

                    "rerank_normalized_score": (

                        cross_score

                    ),

                    "rrf_rank": rrf_rank,

                    "rrf_preservation_score": (

                        rrf_score

                    ),

                    "lexical_score": (

                        lexical_score

                    ),

                    "answer_evidence_score": (

                        answer_score

                    ),

                    "structural_score": (

                        structural_score

                    ),

                    "source_authority": (

                        source_authority

                    ),

                    "query_intent": (

                        query_intent

                    ),

                    "structural_weight_used": (

                        structural_weight

                    ),

                    "structural_tier": (

                        structural["tier"]

                    ),

                    "structural_exact": (

                        structural["exact"]

                    ),

                    "explicit_identifier_query": (

                        explicit_identifier_query

                    ),

                    "final_rerank_score": (

                        final_score

                    ),

                }

            )



        scored.sort(

            key=lambda item: (

                item["final_rerank_score"],

                item["structural_tier"],

                item["answer_evidence_score"],

                item["rerank_normalized_score"],

            ),

            reverse=True,

        )



        return scored[:limit]



    def close(self):

        pass
