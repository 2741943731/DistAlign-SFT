import importlib.util
import unittest

if importlib.util.find_spec("torch") is not None:
    import torch

    from src.losses import causal_lm_loss
else:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CausalLossTests(unittest.TestCase):
    def test_analytical_loss_and_gradient_after_shift_and_mask(self):
        probs = torch.tensor(
            [
                [[0.6, 0.4], [0.2, 0.8], [0.3, 0.7], [0.7, 0.3]],
                [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.5, 0.5]],
            ],
            dtype=torch.double,
        )
        labels = torch.tensor([[1, -100, 0, -100], [0, 0, 1, -100]])
        # Row 0 has one supervised target and row 1 has two. Some prompt
        # logits predict a completion token; the prompt labels do not contribute.
        pairs = [(0, 1, 0), (1, 0, 0), (1, 1, 1)]

        for loss_type in ("sft", "dft"):
            with self.subTest(loss_type=loss_type):
                logits = probs.log().requires_grad_()
                loss = causal_lm_loss(logits, labels, loss_type=loss_type)
                loss.backward()

                expected_losses = []
                expected_grads = torch.zeros_like(logits)
                for row, position, target in pairs:
                    p = probs[row, position, target]
                    weight = p if loss_type == "dft" else 1
                    expected_losses.append(-weight * p.log())
                    gradient = probs[row, position].clone()
                    gradient[target] -= 1
                    expected_grads[row, position] = weight * gradient / len(pairs)

                torch.testing.assert_close(loss, torch.stack(expected_losses).mean())
                torch.testing.assert_close(logits.grad, expected_grads)

    def test_empty_shifted_supervision_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no supervised next-token"):
            causal_lm_loss(torch.zeros(1, 3, 2), torch.tensor([[1, -100, -100]]))

    def test_finite_low_precision_gradient(self):
        logits = torch.tensor([[[30, -30], [0, 0]]], dtype=torch.bfloat16, requires_grad=True)
        loss = causal_lm_loss(logits, torch.tensor([[-100, 1]]), loss_type="dft")
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertGreater(loss.item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all().item())
