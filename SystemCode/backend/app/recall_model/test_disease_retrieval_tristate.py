"""Contract and integration checks for explicit negative evidence."""
import unittest

import numpy as np

from disease_retrieval_tristate import DiseaseRetriever, encode, load_vocab, observed


class TristateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.vocab = load_vocab()

    def test_unknown_is_distinct_from_denial(self):
        unknown, _, _ = encode(['cough'], [], self.vocab)
        denied, _, _ = encode(['cough'], ['fever'], self.vocab)
        s = len(self.vocab['evidence_ids'])
        fever = self.vocab['evidence_ids'].index('E_91')
        self.assertEqual(unknown[0, fever+s], 0)
        self.assertEqual(denied[0, fever+s], 1)
        np.testing.assert_array_equal(unknown[:, :s], denied[:, :s])

    def test_contradiction_across_aliases_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Contradictory'):
            encode(['发热'], ['E_91'], self.vocab)

    def test_unsupported_and_empty_input_rejected(self):
        for yes, no in [(['made_up_symptom'], []), ([], [])]:
            with self.assertRaises(ValueError):
                encode(yes, no, self.vocab)

    def test_simulated_answers_never_invent_or_invert_symptoms(self):
        truth = np.array([[1, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
        result = observed(truth, 1, 2, 42)
        yes, no = result[:, :4], result[:, 4:]
        self.assertTrue(np.all(yes <= truth))
        self.assertTrue(np.all(no <= 1-truth))
        self.assertTrue(np.all(yes*no == 0))
        self.assertTrue(np.all(yes.sum(axis=1) == 1))
        self.assertTrue(np.all(no.sum(axis=1) == 2))

    def test_loaded_model_uses_denial(self):
        retriever = DiseaseRetriever()
        unknown = retriever.retrieve(['cough'], [], 49)
        denied = retriever.retrieve(['cough'], ['fever'], 49)
        a = {v['disease_name']: v['score'] for v in unknown['candidates']}
        b = {v['disease_name']: v['score'] for v in denied['candidates']}
        self.assertEqual(len(a), 49)
        self.assertTrue(all(np.isfinite(v) for v in b.values()))
        self.assertTrue(any(abs(a[k]-b[k]) > 1e-5 for k in a))


if __name__ == '__main__':
    unittest.main()
