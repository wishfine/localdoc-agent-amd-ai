"""Consistency tests for pluggable embedding/generation backends."""

import pytest

from localdoc.backends.gpu_backend import AMDGPUBackend
from localdoc.backends.local_llm_backend import LocalLLMBackend
from localdoc.backends.npu_backend import AMDNPUBackend
from localdoc.backends.simulated_npu import SimulatedNPUBackend


@pytest.mark.parametrize(
    "backend_factory",
    [AMDGPUBackend, AMDNPUBackend, SimulatedNPUBackend, LocalLLMBackend],
)
def test_backend_transform_uses_fitted_document_vocabulary(backend_factory):
    backend = backend_factory()
    docs = ["苹果 香蕉 水果", "火箭 卫星 发射", "咖啡 提神 饮品"]

    doc_vectors = backend.fit_and_embed(docs)
    query_vectors = backend.transform(["苹果 水果"])

    assert len(query_vectors) == 1
    assert len(query_vectors[0]) == len(doc_vectors[0])
    assert len(query_vectors[0]) > 0


@pytest.mark.parametrize(
    "backend_factory",
    [AMDGPUBackend, AMDNPUBackend, SimulatedNPUBackend],
)
def test_backend_generation_string_context_fallback_is_full_text(backend_factory):
    backend = backend_factory()
    context = "完整上下文 fallback sentence"

    answer = backend.generate_answer("zzzzzz", context)

    assert answer == context
