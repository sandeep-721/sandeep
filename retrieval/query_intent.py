from __future__ import annotations

import re


class QueryIntentDetector:
    """Shared, deterministic query-intent classification for retrieval stages."""

    @staticmethod
    def normalize(query: str) -> str:
        if not query:
            return ""
        return re.sub(r"\s+", " ", query.strip().lower())

    @classmethod
    def detect(cls, query: str) -> str:
        normalized = cls.normalize(query)

        if not normalized:
            return "general"

        test_patterns = (
            r"\btest\b",
            r"\btests\b",
            r"\btesting\b",
            r"\btest case\b",
            r"\btest cases\b",
            r"\bunit test\b",
            r"\bintegration test\b",
            r"\btest coverage\b",
            r"\bdiagnos",
            r"\bevaluation\b",
            r"\bevaluate\b",
            r"\bbenchmark\b",
            r"\bbenchmarks\b",
            r"\bregression test\b",
        )
        if any(re.search(pattern, normalized) for pattern in test_patterns):
            return "test"

        code_behavior_patterns = (
            r"\bwhich script\b.*\bverif\w*\b",
            r"\bwhich script\b.*\bvalidat\w*\b",
            r"\bwhich script\b.*\bcheck\w*\b",
            r"\bwhich script\b.*\bassert\w*\b",
        )
        if any(
            re.search(pattern, normalized)
            for pattern in code_behavior_patterns
        ):
            return "code"

        configuration_patterns = (
            r"\bconfig\b",
            r"\bconfiguration\b",
            r"\bconfigured\b",
            r"\bsetting\b",
            r"\bsettings\b",
            r"\boption\b",
            r"\boptions\b",
            r"\bparameter\b",
            r"\bparameters\b",
            r"\bdefault\b",
            r"\bdevice\b",
            r"\bmodel\b",
            r"\bmodels\b",
            r"\bpath\b",
            r"\benvironment variable\b",
            r"\bchunk size\b",
            r"\boverlap\b",
            r"\bwhere is .* configured\b",
            r"\bwhere .* configured\b",
            r"\bwhat .* configured\b",
        )
        if any(
            re.search(pattern, normalized)
            for pattern in configuration_patterns
        ):
            return "configuration"

        history_patterns = (
            r"\bhistory\b",
            r"\bchangelog\b",
            r"\brelease\b",
            r"\breleased\b",
            r"\bversion\s+[0-9]",
            r"\broadmap\b",
            r"\bpreviously\b",
            r"\bwhat did .* add\b",
            r"\bwhat did .* change\b",
            r"\badded in\b",
            r"\bintroduced in\b",
            r"\bchanged in\b",
            r"\bnew in version\b",
        )
        if any(
            re.search(pattern, normalized)
            for pattern in history_patterns
        ):
            return "history"

        documentation_patterns = (
            r"\bdocumentation\b",
            r"\bdocument\b",
            r"\bdocs\b",
            r"\bguide\b",
            r"\bguidance\b",
            r"\bmanual\b",
            r"\bworkflow\b",
            r"\breadme\b",
            r"\binstructions\b",
            r"\brecommended workflow\b",
        )
        if any(
            re.search(pattern, normalized)
            for pattern in documentation_patterns
        ):
            return "documentation"

        code_patterns = (
            r"\bimplementation\b",
            r"\bimplemented\b",
            r"\bfunction\b",
            r"\bmethod\b",
            r"\bclass\b",
            r"\bproperty\b",
            r"\bfield\b",
            r"\bvariable\b",
            r"\bcode\b",
            r"\bsource code\b",
            r"\balgorithm\b",
            r"\blogic\b",
            r"\bscript\b",
            r"\bquery\b",
            r"\bretrieval\b",
            r"\bembedding\b",
            r"\breranker\b",
            r"\bsearch\b",
            r"\bindex\b",
            r"\bindexed\b",
            r"\bvector\b",
            r"\bmember\b",
            r"\bsymbol\b",
            r"\bmapping\b",
            r"\bvalidate\w*\b",
            r"\bremove\w*\b",
            r"\bdelete\w*\b",
            r"\bcreate\w*\b",
            r"\bcombine\w*\b",
            r"\bdetect\w*\b",
            r"\bregenerat\w*\b",
            r"\bwait\w*\b",
            r"\binvoke\w*\b",
            r"\bcall(?:s|ed|ing)?\b",
            r"\breturn(?:s|ed|ing)?\b",
            r"\bused by\b",
            r"\bcalled from\b",
        )
        if any(re.search(pattern, normalized) for pattern in code_patterns):
            return "code"

        generic_documentation_patterns = (
            r"\bhow does\b",
            r"\bhow do\b",
            r"\bwhat does\b",
            r"\bwhy does\b",
            r"\bwhat is\b",
            r"\bwhat are\b",
            r"\bexplain\b",
        )
        if any(
            re.search(pattern, normalized)
            for pattern in generic_documentation_patterns
        ):
            return "documentation"

        return "general"


def detect_query_intent(query: str) -> str:
    """Convenience wrapper used by callers that do not need the class."""
    return QueryIntentDetector.detect(query)
