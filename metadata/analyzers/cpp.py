from metadata.analyzers.cstyle import (
    CStyleSymbolAnalyzer,
)


class CppSymbolAnalyzer(CStyleSymbolAnalyzer):
    """
    C++ adapter using the shared C-style analyzer.
    """

    language = "cpp"