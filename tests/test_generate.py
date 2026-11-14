import unittest

from src.generate import generate_rows


class GenerateTests(unittest.TestCase):
    def test_generation_only_sees_problem_and_grader_gets_only_first_completion(self):
        question = {"id": "test", "problem": "What is a half?", "answer": "0.50", "solution": "SECRET_SOLUTION"}
        prompts = []
        raw = "A half is \\boxed{1/2}.\n\nQuestion: A different problem. \\boxed{4}"

        def generate_one(prompt):
            prompts.append(prompt)
            return raw, 32, True

        result, = generate_rows([question], generate_one)
        self.assertIn(question["problem"], prompts[0])
        self.assertNotIn(question["answer"], prompts[0])
        self.assertNotIn(question["solution"], prompts[0])
        self.assertEqual(result["answer"], "0.50")
        self.assertEqual(result["completion"], "A half is \\boxed{1/2}.")
        self.assertEqual(result["raw_completion"], raw)
        self.assertTrue(result["hit_token_limit"])
