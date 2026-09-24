from retrieval.query_plan import QueryPlanner


def test_code_behavior_plan():
    plan = QueryPlanner.build(
        "Which script verifies baking configuration values and restores state?"
    )

    assert plan.intent == "code"
    assert plan.explicit_identifier_query is False
    assert plan.retrieval_weights == (1.25, 0.75)
    assert plan.retrieval_modes == ("dense", "sparse", "structural")


def test_exact_symbol_plan():
    plan = QueryPlanner.build("FindBone")

    assert plan.intent == "general"
    assert "findbone" in plan.identifiers
    assert "findbone" in plan.class_candidates
    assert plan.explicit_identifier_query is True
    assert plan.retrieval_modes == ("dense", "sparse", "structural")


def test_configuration_plan():
    plan = QueryPlanner.build(
        "Which project setting specifies the Unity editor version?"
    )

    assert plan.intent == "configuration"
    assert plan.retrieval_weights == (0.80, 1.20)
    assert plan.explicit_identifier_query is False


def test_general_plan():
    plan = QueryPlanner.build(
        "What information is available?"
    )

    assert plan.intent == "general"
    assert plan.retrieval_weights == (1.00, 1.00)
    assert plan.retrieval_modes == ("dense", "sparse")
