"""EagleDraftWorker shares target embed/lm_head before KV pool sizing (#44145).

The scheduler sizes the target KV pool after the draft worker is built but
before EagleDraftWorker.alloc_memory_pool() re-binds the draft's embed/lm_head
onto the target's tensors. On untied-embedding checkpoints the draft holds a
duplicate embed_tokens (and lm_head) at profiling time, permanently shrinking
the KV budget. The fix binds eagerly in __init__ — see
EagleWorkerV2._share_embed_and_head_before_kv_sizing — and this test pins both
the eager bind and the no-op guard in alloc_memory_pool.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

VOCAB, HIDDEN = 32, 16


def _worker_like():
    """An EagleDraftWorker-shaped object without running its heavy __init__."""
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

    worker = EagleDraftWorker.__new__(EagleDraftWorker)
    target_embed = torch.randn(VOCAB, HIDDEN)
    target_head = torch.randn(VOCAB, HIDDEN)

    worker.target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            model=SimpleNamespace(
                lm_head=None,
                get_embed_and_head=lambda: (target_embed, target_head),
            ),
            model_config=SimpleNamespace(model_path="m", revision=None, vocab_size=VOCAB),
            load_config=SimpleNamespace(load_format="auto"),
        )
    )

    class DraftModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(VOCAB, HIDDEN)
            self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)
            self.hot_token_id = None

        def set_embed_and_head(self, embed, head):
            del self.embed_tokens.weight
            del self.lm_head.weight
            self.embed_tokens.weight = embed
            self.lm_head.weight = head

    worker.draft_runner = SimpleNamespace(model=DraftModel())
    worker.draft_worker = SimpleNamespace(
        alloc_memory_pool=lambda **kw: SimpleNamespace(
            memory_pool_config=SimpleNamespace(
                max_total_num_tokens=64, max_running_requests=4
            ),
            req_to_token_pool=object(),
            token_to_kv_pool_allocator=object(),
        )
    )
    worker.hot_token_id = None
    worker.speculative_algorithm = SimpleNamespace(is_eagle3=lambda: False)
    worker.req_to_token_pool = None
    worker.token_to_kv_pool_allocator = None
    return worker, target_embed, target_head


class TestEagerEmbedSharing(CustomTestCase):
    def test_eager_bind_shares_target_tensors(self):
        worker, target_embed, target_head = _worker_like()
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.get_spec",
            return_value=SimpleNamespace(
                speculative_algorithm=SimpleNamespace(is_eagle3=lambda: False),
                speculative_token_map=None,
                speculative_use_rejection_sampling=False,
            ),
        ):
            worker.init_token_map()
            worker.init_lm_head()
            worker._embed_and_head_shared = True

        self.assertIs(worker.draft_runner.model.embed_tokens.weight, target_embed)
        self.assertIs(worker.draft_runner.model.lm_head.weight, target_head)

        # The deferred bind in alloc_memory_pool must be skipped (no-op), so
        # the shared identity survives pool allocation.
        embed_before = worker.draft_runner.model.embed_tokens.weight
        with patch(
            "sglang.srt.speculative.eagle_worker_v2.get_spec",
            return_value=SimpleNamespace(
                speculative_use_rejection_sampling=False,
            ),
        ):
            worker.alloc_memory_pool()
        self.assertIs(worker.draft_runner.model.embed_tokens.weight, embed_before)

    def test_guard_skips_when_model_has_no_setter(self):
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        worker = EagleDraftWorker.__new__(EagleDraftWorker)
        worker.draft_runner = SimpleNamespace(model=SimpleNamespace())  # no setter
        # Must not raise and must not mark shared.
        worker._share_embed_and_head_before_kv_sizing()
        self.assertFalse(getattr(worker, "_embed_and_head_shared", False))

    def test_failed_eager_bind_falls_back_to_deferred(self):
        from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

        worker = EagleDraftWorker.__new__(EagleDraftWorker)

        class FailingModel:
            def set_embed_and_head(self, embed, head):
                raise RuntimeError("boom")

        worker.draft_runner = SimpleNamespace(model=FailingModel())

        def failing_init_lm_head():
            raise RuntimeError("boom")

        worker.init_lm_head = failing_init_lm_head
        worker.init_token_map = lambda: None
        # Must swallow the failure and leave the flag unset so the deferred
        # bind in alloc_memory_pool still runs.
        worker._share_embed_and_head_before_kv_sizing()
        self.assertFalse(getattr(worker, "_embed_and_head_shared", False))


if __name__ == "__main__":
    unittest.main()
