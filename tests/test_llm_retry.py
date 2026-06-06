"""Tests for retryable LLM/embedding exception classification."""

from google.genai import errors as genai_errors

from src.llm import _is_retryable_exception


def test_google_genai_rate_limit_error_is_retryable():
    exc = genai_errors.ClientError(429, {"error": {"message": "quota"}})

    assert _is_retryable_exception(exc) is True


def test_google_genai_bad_request_error_is_not_retryable():
    exc = genai_errors.ClientError(400, {"error": {"message": "bad request"}})

    assert _is_retryable_exception(exc) is False
