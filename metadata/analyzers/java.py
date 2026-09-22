from metadata.analyzers.cstyle import (
    CStyleSymbolAnalyzer,
)


class JavaSymbolAnalyzer(CStyleSymbolAnalyzer):
    """
    Java adapter using the shared C-style structural parser.
    """

    language = "java"