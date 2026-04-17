import math
import unittest
import warnings

import torch

from src.model import Model
from src.trainer import Trainer, l2s_collate_fn


class TestL2SegRegression(unittest.TestCase):
    def test_collate_warns_and_remaps_unknown_ar_tokens(self):
        batch = [
            {
                "ar_sequences": [0, 1, 3, 999],
                "nar_labels": [1.0, 0.0],
                "state_dict": {
                    "global_node_indices": [1, 2],
                    "depot_xy": torch.tensor([0.0, 0.0], dtype=torch.float32),
                    "node_xy": torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32),
                    "node_demand": torch.tensor([1.0, 1.0], dtype=torch.float32),
                    "tour_index": torch.tensor([0, 1], dtype=torch.long),
                    "neighbours": torch.tensor([[0, 2], [1, 0]], dtype=torch.long),
                },
            }
        ]

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = l2s_collate_fn(batch, global_end_token=3, pad_token=4)

        self.assertTrue(any("Unknown AR token IDs" in str(w.message) for w in caught))
        self.assertEqual(output["ar_sequences"].shape, (1, 4))
        self.assertEqual(output["ar_sequences"][0].tolist(), [0, 1, 3, 3])

    def test_ar_loss_respects_pad_and_phase_weights(self):
        trainer = Trainer.__new__(Trainer)
        trainer.device = torch.device("cpu")
        trainer.trainer_params = {
            "ar_loss_weight": 1.0,
            "ar_delete_weight": 2.0,
            "ar_insert_weight": 0.5,
        }

        nar_logits = torch.zeros((1, 2), dtype=torch.float32)
        nar_labels = torch.zeros((1, 2), dtype=torch.float32)
        ar_sequences = torch.tensor([[0, 1, 2, 4]], dtype=torch.long)

        ar_logits = torch.zeros((1, 4, 5), dtype=torch.float32)
        ar_logits[0, 2, :] = torch.tensor([100.0, -100.0, -100.0, -100.0, -100.0])

        total_loss, loss_nar, loss_ar = trainer._compute_l2seg_loss(
            nar_logits, nar_labels, ar_logits, ar_sequences, pad_token=4
        )

        expected_loss_ar = 1.25 * math.log(5.0)
        self.assertTrue(torch.isclose(loss_ar, torch.tensor(expected_loss_ar), atol=1e-6))
        self.assertTrue(torch.isclose(total_loss, loss_nar + loss_ar, atol=1e-6))

    def test_sanitize_starting_node_warns_and_clamps(self):
        model = Model.__new__(Model)
        torch.nn.Module.__init__(model)
        model.PAD_TOKEN = 4

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clamped = model._sanitize_starting_node(999)

        self.assertEqual(clamped, 4)
        self.assertTrue(any("will be clamped" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
