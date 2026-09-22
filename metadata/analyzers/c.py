from metadata.analyzers.cstyle import (
    CStyleSymbolAnalyzer,
)


class CSymbolAnalyzer(CStyleSymbolAnalyzer):
    """
    C adapter using the shared C-style analyzer.
    """

    language = "c"