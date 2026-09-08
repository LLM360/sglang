import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
import torch

from sglang.srt.layers import sampler as sampler_module
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=180, suite="stage-b-test-1-gpu-small")


class TestSamplingMask(unittest.TestCase):
    def setUp(self):
        self.sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(self.sampler)
        self.sampler.use_nan_detection = False
        self.sampler.rl_on_policy_target = None
        self.sampler.enable_deterministic = False
        self.sampler.use_log_softmax_logprob = False
        self.sampler.use_ascend_backend = False

    def _run_sampler(
        self,
        backend,
        probs,
        top_k,
        top_p,
        min_p,
        return_sampling_mask,
    ):
        logits_output = LogitsProcessorOutput(
            next_token_logits=torch.tensor([probs], device="cuda").log()
        )
        sampling_info = SimpleNamespace(
            temperatures=torch.ones((1, 1), device="cuda"),
            top_ks=torch.tensor([top_k], dtype=torch.int32, device="cuda"),
            top_ps=torch.tensor([top_p], device="cuda"),
            min_ps=torch.tensor([min_p], device="cuda"),
            is_all_greedy=top_k == 1,
            need_top_k_sampling=top_k != TOP_K_ALL,
            need_top_p_sampling=top_p != 1.0,
            need_min_p_sampling=min_p > 0.0,
            return_sampling_masks=[True] if return_sampling_mask else None,
            sampling_seed=None,
            has_custom_logit_processor=False,
            grammars=None,
            device="cuda",
        )
        with patch(
            "sglang.srt.layers.sampler.get_global_server_args",
            return_value=SimpleNamespace(sampling_backend=backend),
        ):
            token_ids = self.sampler(
                logits_output,
                sampling_info,
                return_logprob=False,
                top_logprobs_nums=[0],
                token_ids_logprobs=[None],
                positions=torch.zeros(1, dtype=torch.int64, device="cuda"),
            )
        return int(token_ids.item()), logits_output

    def _assert_sampler_case(
        self, backend, probs, top_k, top_p, min_p, expected_support
    ):
        torch.cuda.manual_seed(1234)
        rng_state = torch.cuda.get_rng_state()
        token_without, output_without = self._run_sampler(
            backend, probs, top_k, top_p, min_p, False
        )
        torch.cuda.set_rng_state(rng_state)
        token_with, output_with = self._run_sampler(
            backend, probs, top_k, top_p, min_p, True
        )

        self.assertEqual(token_with, token_without)
        self.assertIsNone(output_without.next_token_sampling_mask_idx)
        self.assertIsNone(output_without.next_token_sampling_logprobs)

        support = output_with.next_token_sampling_mask_idx[0]
        sampling_logprob = output_with.next_token_sampling_logprobs[0]
        self.assertEqual(set(support), set(expected_support))
        self.assertIn(token_with, support)
        support_mass = sum(probs[token_id] for token_id in expected_support)
        expected_logprob = math.log(probs[token_with] / support_mass)
        self.assertAlmostEqual(sampling_logprob, expected_logprob, places=6)

    def _assert_api_alignment(self, output_ids, meta_info):
        masks = meta_info["output_token_sampling_mask"]
        logprobs = meta_info["output_token_sampling_logprobs"]
        self.assertEqual(len(masks), len(output_ids))
        self.assertEqual(len(logprobs), len(output_ids))
        self.assertEqual(
            meta_info["output_token_sampling_mask_length"], len(output_ids)
        )
        for output_id, mask, logprob in zip(output_ids, masks, logprobs):
            self.assertIn(output_id, mask)
            self.assertTrue(math.isfinite(logprob))

    def test_pytorch_sampler_correctness(self):
        cases = [
            ("pure_top_p", [0.4, 0.3, 0.2, 0.1], TOP_K_ALL, 0.6, 0.0, {0, 1}),
            ("top_k_top_p", [0.4, 0.3, 0.2, 0.1], 2, 0.5, 0.0, {0, 1}),
            ("min_p", [0.4, 0.3, 0.2, 0.1], TOP_K_ALL, 1.0, 0.6, {0, 1}),
            ("greedy", [0.4, 0.3, 0.2, 0.1], 1, 1.0, 0.0, {0}),
        ]
        for name, probs, top_k, top_p, min_p, support in cases:
            with self.subTest(name=name):
                self._assert_sampler_case(
                    "pytorch", probs, top_k, top_p, min_p, support
                )

    def test_flashinfer_sampler_correctness(self):
        tie_probs = [0.4, 0.2, 0.2, 0.1, 0.1]

        with (
            patch.object(
                sampler_module,
                "top_k_renorm_prob",
                wraps=sampler_module.top_k_renorm_prob,
            ) as top_k_renorm,
            patch.object(
                sampler_module,
                "top_p_renorm_prob",
                wraps=sampler_module.top_p_renorm_prob,
            ) as top_p_renorm,
        ):
            self._run_sampler("flashinfer", tie_probs, 2, 0.45, 0.0, False)
        top_k_renorm.assert_not_called()
        top_p_renorm.assert_not_called()

        with patch.object(
            sampler_module,
            "top_k_renorm_prob",
            wraps=sampler_module.top_k_renorm_prob,
        ) as top_k_renorm:
            self._assert_sampler_case(
                "flashinfer",
                [0.4, 0.3, 0.2, 0.1],
                TOP_K_ALL,
                0.6,
                0.0,
                {0, 1},
            )
        top_k_renorm.assert_not_called()

        cases = [
            ("cutoff_tie", tie_probs, 2, 0.45, 0.0, {0, 1, 2}),
            ("min_p", [0.4, 0.3, 0.2, 0.1], TOP_K_ALL, 1.0, 0.6, {0, 1}),
            ("greedy", [0.4, 0.3, 0.2, 0.1], 1, 1.0, 0.0, {0}),
        ]
        for name, probs, top_k, top_p, min_p, support in cases:
            with self.subTest(name=name):
                self._assert_sampler_case(
                    "flashinfer", probs, top_k, top_p, min_p, support
                )

    def test_public_api_contract(self):
        model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--mem-fraction-static",
                "0.7",
                "--chunked-prefill-size",
                "4",
                "--tokenizer-worker-num",
                2,
            ],
        )
        try:
            sampling_params = {
                "temperature": 1.0,
                "top_p": 0.8,
                "top_k": -1,
                "max_new_tokens": 3,
                "ignore_eos": True,
            }
            response = requests.post(
                base_url + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": sampling_params,
                    "return_sampling_mask": True,
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)
            generate_output = response.json()
            self.assertEqual(len(generate_output["output_ids"]), 3)
            self._assert_api_alignment(
                generate_output["output_ids"], generate_output["meta_info"]
            )

            session_id = requests.post(
                base_url + "/open_session",
                json={"capacity_of_str_len": 1000},
                timeout=60,
            ).json()
            session_payload = {
                "text": "Generate within a session.",
                "session_params": {"id": session_id},
                "sampling_params": {**sampling_params, "max_new_tokens": 1},
                "return_sampling_mask": True,
            }
            response = requests.post(
                base_url + "/generate",
                json=session_payload,
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)
            session_output = response.json()
            self.assertEqual(len(session_output["output_ids"]), 1)
            self._assert_api_alignment(
                session_output["output_ids"], session_output["meta_info"]
            )
            response = requests.post(
                base_url + "/generate",
                json={
                    **session_payload,
                    "session_params": {"id": session_id + "-missing"},
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 400, response.text)

            response = requests.post(
                base_url + "/generate",
                json={
                    "text": "Cache this prefix without decoding.",
                    "sampling_params": {
                        **sampling_params,
                        "max_new_tokens": 0,
                    },
                    "return_sampling_mask": True,
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)
            prefill_only_output = response.json()
            self.assertEqual(prefill_only_output["output_ids"], [])
            self._assert_api_alignment(
                prefill_only_output["output_ids"], prefill_only_output["meta_info"]
            )

            response = requests.post(
                base_url + "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Name a capital city."}],
                    "temperature": 1.0,
                    "top_p": 0.8,
                    "top_k": -1,
                    "max_tokens": 3,
                    "ignore_eos": True,
                    "return_sampling_mask": True,
                    "return_meta_info": True,
                    "return_completion_token_ids": True,
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)
            choice = response.json()["choices"][0]
            self.assertEqual(len(choice["completion_token_ids"]), 3)
            self._assert_api_alignment(
                choice["completion_token_ids"], choice["meta_info"]
            )

            greedy_payload = {
                "text": "The capital of Germany is",
                "sampling_params": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 1,
                    "max_new_tokens": 3,
                    "ignore_eos": True,
                },
                "return_logprob": True,
            }
            without_mask = requests.post(
                base_url + "/generate", json=greedy_payload, timeout=60
            )
            with_mask = requests.post(
                base_url + "/generate",
                json={**greedy_payload, "return_sampling_mask": True},
                timeout=60,
            )
            self.assertEqual(without_mask.status_code, 200, without_mask.text)
            self.assertEqual(with_mask.status_code, 200, with_mask.text)
            without_mask = without_mask.json()
            with_mask = with_mask.json()
            self.assertEqual(with_mask["output_ids"], without_mask["output_ids"])
            self.assertEqual(
                with_mask["meta_info"]["output_token_logprobs"],
                without_mask["meta_info"]["output_token_logprobs"],
            )
            self._assert_api_alignment(with_mask["output_ids"], with_mask["meta_info"])
            for key in (
                "output_token_sampling_mask",
                "output_token_sampling_logprobs",
                "output_token_sampling_mask_length",
            ):
                self.assertNotIn(key, without_mask["meta_info"])

            rejected_rid = "sampling-mask-rejection"
            response = requests.post(
                base_url + "/generate",
                json={
                    "rid": rejected_rid,
                    "text": "Reject this request",
                    "sampling_params": {
                        "top_p": 1.0,
                        "top_k": -1,
                        "min_p": 0.0,
                        "max_new_tokens": 1,
                    },
                    "return_sampling_mask": True,
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 400, response.text)

            response = requests.post(
                base_url + "/generate",
                json={
                    "rid": rejected_rid,
                    "text": "Retry this request",
                    "sampling_params": {
                        "top_p": 0.8,
                        "top_k": -1,
                        "max_new_tokens": 1,
                        "ignore_eos": True,
                    },
                    "return_sampling_mask": True,
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 200, response.text)
            retry_output = response.json()
            self.assertEqual(len(retry_output["output_ids"]), 1)
            self._assert_api_alignment(
                retry_output["output_ids"], retry_output["meta_info"]
            )

            response = requests.post(
                base_url + "/generate",
                json={
                    "text": "Reject streaming.",
                    "sampling_params": sampling_params,
                    "stream": True,
                    "return_sampling_mask": True,
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 400, response.text)

            response = requests.post(
                base_url + "/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Reject streaming."}],
                    "top_p": 0.8,
                    "top_k": -1,
                    "max_tokens": 1,
                    "stream": True,
                    "return_sampling_mask": True,
                    "return_meta_info": True,
                },
                timeout=60,
            )
            self.assertEqual(response.status_code, 400, response.text)
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
