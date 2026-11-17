import json
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

from app.evaluation.metrics import citation_scores, mean_metrics, retrieval_scores, token_scores
from app.evaluation.runner import (
    DEFAULT_DATASET,
    _compare_reports,
    _offline_results,
    evaluate,
    load_dataset,
)


def test_retrieval_metrics_measure_hit_recall_mrr_and_ndcg():
    relevant = {("doc", 1): 3, ("doc", 2): 1}

    scores = retrieval_scores([("doc", 3), ("doc", 1)], relevant, cutoff=2)

    expected_ndcg = (7 / 1.584962500721156) / (7 + 1 / 1.584962500721156)
    assert scores == {
        "hit_rate": 1.0,
        "recall": 0.5,
        "mrr": 0.5,
        "ndcg": expected_ndcg,
    }


def test_token_scores_are_case_insensitive_and_deterministic():
    assert token_scores("The result is 42 units.", "42 units") == pytest.approx({
        "precision": 0.4,
        "recall": 1.0,
        "f1": 4 / 7,
    })


def test_citation_scores_compare_document_and_page_pairs():
    scores = citation_scores([("doc", 1), ("doc", 2)], [("doc", 1)])

    assert scores == {"precision": 0.5, "recall": 1.0, "f1": 2 / 3}


def test_aggregate_metrics_keep_groups_separate():
    aggregate = mean_metrics(
        [
            {"retrieval": {"mrr": 1.0}, "citations": {"recall": 0.5}},
            {"retrieval": {"mrr": 0.0}, "citations": {"recall": 1.0}},
        ]
    )

    assert aggregate == {
        "citations": {"recall": 0.75},
        "retrieval": {"mrr": 0.5},
    }


def test_offline_results_are_repeatable_and_use_the_golden_cases():
    dataset = load_dataset(DEFAULT_DATASET)

    first = _offline_results(dataset, candidates=5, top_k=3, rrf_k=60)
    second = _offline_results(dataset, candidates=5, top_k=3, rrf_k=60)

    assert first == second
    assert len(first) == len(dataset["cases"])
    assert all(result["retrieval"]["hit_rate"] == 1 for result in first)
    assert all("answer_quality" not in result for result in first)


def test_offline_evaluation_writes_metadata_and_metrics_without_cloud_calls():
    with TemporaryDirectory() as directory:
        output = Path(directory) / "report.json"
        args = Namespace(
            mode="offline",
            dataset=DEFAULT_DATASET,
            output=output,
            compare=None,
            candidates=10,
            top_k=5,
            rrf_k=60,
            judge=False,
            chunking_strategy=None,
        )
        with (
            patch("app.evaluation.runner.RagService", side_effect=AssertionError("not offline")),
            patch("app.evaluation.runner.MilvusStore", side_effect=AssertionError("not offline")),
        ):
            report = evaluate(args)
            output.write_text(json.dumps(report))

        loaded = json.loads(output.read_text())

    assert loaded["schema_version"] == 1
    assert loaded["git_commit"] is None or isinstance(loaded["git_commit"], str)
    assert loaded["config"]["mode"] == "offline"
    assert loaded["aggregate"]["retrieval"]["mrr"] == 11 / 12
    assert len(loaded["cases"]) == 6


def test_live_mode_refuses_to_run_without_gemini_key():
    with (
        patch(
            "app.evaluation.runner.get_settings",
            return_value=type(
                "Settings",
                (),
                {"gemini_api_key": "", "milvus_collection": "app_collection"},
            )(),
        ),
        patch("app.evaluation.runner.MilvusStore", side_effect=AssertionError("must fail closed")),
    ):
        try:
            from app.evaluation.runner import _live_results

            _live_results(
                load_dataset(DEFAULT_DATASET),
                "rag_eval_0123456789ab",
                candidates=10,
                top_k=5,
                rrf_k=60,
                judge_enabled=False,
                chunking_strategy=None,
            )
        except ValueError as error:
            assert "GEMINI_API_KEY" in str(error)
        else:
            raise AssertionError("Expected live evaluation to require a Gemini key")


def test_live_mode_refuses_to_use_application_milvus_collection():
    settings = type(
        "Settings",
        (),
        {"gemini_api_key": "test-key", "milvus_collection": "app_collection"},
    )()
    with (
        patch("app.evaluation.runner.get_settings", return_value=settings),
        patch("app.evaluation.runner.MilvusStore", side_effect=AssertionError("must fail closed")),
    ):
        try:
            from app.evaluation.runner import _live_results

            _live_results(
                load_dataset(DEFAULT_DATASET),
                "app_collection",
                candidates=10,
                top_k=5,
                rrf_k=60,
                judge_enabled=False,
                chunking_strategy=None,
            )
        except ValueError as error:
            assert "must not match MILVUS_COLLECTION" in str(error)
        else:
            raise AssertionError("Expected live evaluation to reject the application collection")


def test_dataset_cannot_include_unstructured_data():
    with TemporaryDirectory() as directory:
        path = Path(directory) / "invalid.json"
        path.write_text('{"documents": [], "cases": []}')
        try:
            load_dataset(path)
        except ValueError as error:
            assert "documents and cases" in str(error)
        else:
            raise AssertionError("Expected empty evaluation dataset to be rejected")


def test_baseline_comparison_reports_metric_deltas(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {"mode": "offline", "dataset": "gold.json", "cutoff": 5},
                "aggregate": {"retrieval": {"mrr": 0.5}},
            }
        )
    )
    current = {
        "schema_version": 1,
        "config": {"mode": "offline", "dataset": "gold.json", "cutoff": 5},
        "aggregate": {"retrieval": {"mrr": 0.75}},
    }

    assert _compare_reports(current, baseline_path) == {
        "baseline": "baseline.json",
        "deltas": {"retrieval": {"mrr": 0.25}},
    }


def test_baseline_comparison_rejects_different_cutoffs(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "config": {"mode": "offline", "dataset": "gold.json", "cutoff": 3},
                "aggregate": {},
            }
        )
    )
    current = {
        "schema_version": 1,
        "config": {"mode": "offline", "dataset": "gold.json", "cutoff": 5},
        "aggregate": {},
    }

    try:
        _compare_reports(current, baseline_path)
    except ValueError as error:
        assert "same cutoff" in str(error)
    else:
        raise AssertionError("Expected comparison with a different cutoff to fail")
