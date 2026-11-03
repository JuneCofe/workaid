import pytest

from backend.retrieval.evaluation import load_cases, run_comparison

HOLDOUT = "statute_holdout.jsonl"


@pytest.fixture(scope="module")
def reports():
    return run_comparison()


def test_pipeline_far_outperforms_raw_sentence_search(reports):
    baseline, pipeline = reports

    assert pipeline.hit_at(5) >= 0.75  # keyword-only path: no embedding key or reranker in tests
    assert pipeline.hit_at(5) > baseline.hit_at(5)
    assert pipeline.mrr() >= 0.65


def test_pipeline_abstains_on_non_legal_questions(reports):
    assert reports[1].abstained() == 1.0


def test_holdout_queries_keep_a_reasonable_hit_rate():
    from backend.retrieval.evaluation import DEFAULT_CASES

    _, pipeline = run_comparison(load_cases(DEFAULT_CASES.with_name(HOLDOUT)))

    assert pipeline.hit_at(5) >= 0.60
    assert pipeline.abstained() == 1.0
