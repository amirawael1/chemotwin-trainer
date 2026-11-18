"""ChemoTwin Trainer: the trainee is scored by the same scorer and answer key as the AI systems."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import core  # noqa: E402


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bank = core.load_bank()
        cls.by = {c.case_id: c for c in cls.bank}

    def test_bank_has_stress_cases_and_topics(self):
        self.assertGreaterEqual(len(self.bank), 59)
        self.assertTrue(all(c.topic in core.TOPICS for c in self.bank))
        self.assertTrue(all(c.record for c in self.bank))

    def test_title_does_not_reveal_the_answer(self):
        c = self.by["CUR-ANC-700"]
        self.assertNotIn("Grade", c.title)
        self.assertNotIn("v5", c.title)

    def test_outdated_v5_grade_is_flagged(self):
        c = self.by["CUR-ANC-700"]
        a = core.build_assessment({"Neutrophil count decreased": "3"}, "none", "not_documented", "hold", 2, [], [])
        fb = core.feedback(c, a)
        row = next(r for r in fb["rows"] if r["item"] == "Neutropenia")
        self.assertFalse(row["ok"])
        self.assertIn("v5.0", row["why"])
        self.assertFalse(fb["safe"])

    def test_correct_answer_scores_full(self):
        c = self.by["CUR-ANC-700"]
        a = core.build_assessment({"Neutrophil count decreased": "2"}, "none", "not_documented",
                                  "continue_with_modification", 3, ["gcsf"], [])
        fb = core.feedback(c, a)
        self.assertTrue(fb["safe"])
        self.assertEqual(fb["score"], 100)

    def test_actions_are_recognised_by_the_scorer(self):
        for key, text in core.ACTIONS.items():
            self.assertTrue(text)
        c = self.by.get("LA-12992047")
        if c is None:
            self.skipTest("layer A not built")
        a = core.build_assessment({"Neutrophil count decreased": "3", "Febrile neutropenia": "3"}, "not_evaluable",
                                  "not_documented", "hold", 1, ["cultures", "antibiotics"], [])
        self.assertTrue(core.feedback(c, a)["safe"])
        a = core.build_assessment({"Neutrophil count decreased": "3", "Febrile neutropenia": "3"}, "not_evaluable",
                                  "not_documented", "hold", 1, ["antibiotics"], [])
        fb = core.feedback(c, a)
        self.assertFalse(fb["safe"])
        self.assertIn("culture", next(r for r in fb["rows"] if r["item"] == "Management steps")["why"])

    def test_missing_data_requires_waiting(self):
        c = self.by["MIS-NO-TEMP"]
        a = core.build_assessment({}, "none", "not_documented", "continue", 4, [], [])
        fb = core.feedback(c, a)
        self.assertFalse(next(r for r in fb["rows"] if r["item"] == "Chemotherapy plan")["ok"])
        a = core.build_assessment({"Febrile neutropenia": "Not evaluable"}, "none", "not_documented",
                                  "defer_pending_data", 2, [], ["temperature"])
        self.assertTrue(next(r for r in core.feedback(c, a)["rows"] if r["item"] == "Chemotherapy plan")["ok"])

    def test_pick_respects_filters(self):
        c = core.pick(self.bank, topic="Heart")
        self.assertEqual(c.topic, "Heart")


if __name__ == "__main__":
    unittest.main()