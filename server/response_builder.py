def build_ask_response(
    query,
    project,
    result,
    config,
):
    return {
        "query": query,
        "project": project,
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "results": result.get("results", []),
        "evidence_packet": result.get(
            "evidence_packet",
            None,
        ),
        "verification": result.get(
            "verification",
            None,
        ),
        "repair": result.get(
            "repair",
            {
                "attempts": 0,
                "performed": False,
                "used": False,
                "added_results": 0,
            },
        ),
        "model": {
            "provider": config.provider,
            "model": config.model,
            "base_url": config.base_url,
        },
    }
