"""Tool-model definition files are parsed once and cached process-wide
(perf/cache-error-model-defs): the YAML/JSON parses + the JSON name-resolution
walk were re-run on every Survey. These tests pin: (1) the cache produces
covariance identical to a cold parse, (2) repeated construction hits the cache
(no re-parse), and (3) clear_error_model_cache resets it.
"""
import numpy as np
import pytest

from welleng.survey import Survey, SurveyHeader
from welleng.errors import tool_errors as te


def _survey():
    n = 200
    md = np.linspace(0, 30 * (n - 1), n)
    inc = np.clip(np.linspace(0, 90, n), 0, 60)
    azi = np.linspace(0, 120, n) % 360
    return Survey(
        md=md, inc=inc, azi=azi, header=SurveyHeader(),
        error_model="ISCWSA MWD Rev5.11",
    )


def test_cached_cov_identical_to_cold():
    te.clear_error_model_cache()
    cold = _survey().cov_nev      # parses from disk
    warm = _survey().cov_nev      # served from cache
    assert np.allclose(cold, warm, equal_nan=True)


def test_repeated_construction_hits_cache():
    te.clear_error_model_cache()
    _survey()                     # primes the cache (a YAML model)
    before = te._load_yaml_model.cache_info()
    _survey()                     # must be served from cache, not re-parsed
    after = te._load_yaml_model.cache_info()
    assert after.hits > before.hits        # a cache hit occurred
    assert after.misses == before.misses   # no new parse


def test_yaml_path_does_not_walk_json_tree():
    # the MWD model resolves via YAML -> the ~100-file JSON resolution walk must
    # NOT run (it did, unconditionally, before the lazy reorder).
    te.clear_error_model_cache()
    _survey()
    assert te._resolve_json_model.cache_info().misses == 0


def test_clear_cache_resets():
    te.clear_error_model_cache()
    _survey()
    assert te._load_yaml_model.cache_info().currsize > 0
    te.clear_error_model_cache()
    assert te._load_yaml_model.cache_info().currsize == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
