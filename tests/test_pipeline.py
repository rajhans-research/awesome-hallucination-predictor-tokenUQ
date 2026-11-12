"""
tests/test_pipeline.py
----------------------
Unit and edge-case tests for data alignment, baseline scorers, label extraction,
and metric computation.
"""

import os
import sys
import unittest
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import align_sentences_to_tokens
from src.preprocessing import extract_labels_for_protocol
from src.baseline_methods import (
    score_token_mean,
    score_token_min,
    score_token_entropy5,
)
from src.evaluation import compute_sentence_metrics


class TestTokenUQPipeline(unittest.TestCase):

    def test_alignment_standard_and_unicode(self):
        full_text = "\nSirið Stenberg is from Týr. She plays violin."
        sentences = ["Sirið Stenberg is from Týr.", "She plays violin."]
        # Simulated text_offset matching character positions starting at offset 50
        # \n=50, Sirið=51, ' '=56, Stenberg=57, ' '=65, is=66, ' '=68, from=69, ' '=73, Týr=74, .=77, ' '=78, She=79, ...
        offsets = [50, 51, 57, 66, 69, 74, 77, 79, 83, 89, 95]
        aligned = align_sentences_to_tokens(sentences, full_text, offsets)
        self.assertEqual(len(aligned), 2)
        self.assertTrue(len(aligned[0]) >= 5)
        self.assertTrue(len(aligned[1]) >= 3)

    def test_alignment_empty_edge_case(self):
        self.assertEqual(align_sentences_to_tokens(["Hello."], "", []), [[]])

    def test_baseline_scorers_math(self):
        sent_entry = {
            "token_logprobs": [-0.1, -2.0, -0.5],
            "top_logprobs": [
                {"a": -0.1, "b": -2.3},
                {"x": -1.0, "y": -1.0},
                {"z": 0.0},
            ],
        }
        self.assertAlmostEqual(score_token_mean(sent_entry), (0.1 + 2.0 + 0.5) / 3.0, places=6)
        self.assertAlmostEqual(score_token_min(sent_entry), 2.0, places=6)
        # Entropy of {"z": 0.0} is 0.0; {"x": -1.0, "y": -1.0} is 1.0 bit
        ent = score_token_entropy5(sent_entry, use_perplexity=False)
        self.assertTrue(0.0 < ent < 1.0)

    def test_baseline_empty_tokens_edge_case(self):
        empty_sent = {"token_logprobs": [], "top_logprobs": []}
        self.assertEqual(score_token_mean(empty_sent), 0.0)
        self.assertEqual(score_token_min(empty_sent), 0.0)
        self.assertEqual(score_token_entropy5(empty_sent), 0.0)

    def test_label_protocols(self):
        dummy_records = [
            {
                "sentences": [
                    {"annotation": "accurate"},
                    {"annotation": "minor_inaccurate"},
                    {"annotation": "major_inaccurate"},
                ]
            },
            {
                "sentences": [
                    {"annotation": "major_inaccurate"},
                    {"annotation": "major_inaccurate"},
                ]
            },
        ]
        k_nf, y_nf = extract_labels_for_protocol(dummy_records, "nonfact")
        self.assertEqual(y_nf.tolist(), [0, 1, 1, 1, 1])

        k_nfs, y_nfs = extract_labels_for_protocol(dummy_records, "nonfact_star")
        # Passage 1 (100% major) is excluded! Only passage 0 remains
        self.assertEqual(y_nfs.tolist(), [0, 0, 1])

        k_maj, y_maj = extract_labels_for_protocol(dummy_records, "major_only")
        # minor_inaccurate excluded
        self.assertEqual(y_maj.tolist(), [0, 1, 1, 1])

    def test_evaluation_perfect_and_inverse(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int32)
        y_perfect = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        res_p = compute_sentence_metrics(y_true, y_perfect, n_bootstrap=50, seed=42)
        self.assertAlmostEqual(res_p["auroc"], 100.0, places=4)
        self.assertAlmostEqual(res_p["auc_pr"], 100.0, places=4)

        y_inverse = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        res_i = compute_sentence_metrics(y_true, y_inverse, n_bootstrap=50, seed=42)
        self.assertAlmostEqual(res_i["auroc"], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
