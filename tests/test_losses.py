import importlib.util
import math
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

        for loss_type in ("sft", "dft", "idft"):
            with self.subTest(loss_type=loss_type):
                logits = probs.log().requires_grad_()
                loss = causal_lm_loss(logits, labels, loss_type=loss_type)
                loss.backward()

                expected_losses = []
                expected_grads = torch.zeros_like(logits)
                for row, position, target in pairs:
                    distribution = probs[row, position]
                    p = distribution[target]
                    if loss_type == "idft":
                        entropy = -sum(q * math.log(q) for q in distribution.tolist())
                        gamma = max(0.1, min(2.0, math.exp(-math.log(p.item()) - entropy)))
                        weight = p ** gamma
                    else:
                        weight = p if loss_type == "dft" else 1
                    expected_losses.append(-weight * p.log())
                    gradient = distribution.clone()
                    gradient[target] -= 1
                    expected_grads[row, position] = weight * gradient / len(pairs)

                torch.testing.assert_close(loss, torch.stack(expected_losses).mean())
                torch.testing.assert_close(logits.grad, expected_grads)

    def test_idft_distinguishes_contexts_with_the_same_target_probability(self):
        # Both targets have p=0.25. The first context is uniform; the second
        # has a likely alternative, so IDFT applies smaller weight there.
        probs = torch.tensor([[[0.25, 0.25, 0.25, 0.25], [0.25, 0.73, 0.01, 0.01],
                               [0.25, 0.25, 0.25, 0.25]]], dtype=torch.double)
        labels = torch.tensor([[-100, 0, 0]])
        grads = {}
        for loss_type in ("dft", "idft"):
            logits = probs.log().requires_grad_()
            causal_lm_loss(logits, labels, loss_type=loss_type).backward()
            grads[loss_type] = logits.grad[0, :2, 0]
        torch.testing.assert_close(grads["dft"][0], grads["dft"][1])
        torch.testing.assert_close(grads["idft"][0], grads["dft"][0])
        torch.testing.assert_close(grads["idft"][1], grads["dft"][1] * 0.25)

    def test_idft_floors_near_zero_target_probability_in_bfloat16(self):
        logits = torch.tensor([[[1000, -1000], [0, 0]]], dtype=torch.bfloat16, requires_grad=True)
        loss = causal_lm_loss(logits, torch.tensor([[-100, 1]]), loss_type="idft")
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(math.log(1e10) * 1e-20), rtol=1e-5, atol=0)
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

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
