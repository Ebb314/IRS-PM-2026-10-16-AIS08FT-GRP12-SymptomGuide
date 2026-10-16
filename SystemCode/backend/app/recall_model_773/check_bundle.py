import unittest

import numpy as np

from disease_retrieval_773_tristate import DiseaseRetriever773Tristate


class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DiseaseRetriever773Tristate()

    def test_negative_inputs_and_full_candidate_space(self):
        result = self.model.retrieve(present_symptoms=['cough'],absent_symptoms=['fever'],topk=773)
        self.assertEqual(len(result['candidates']),773)
        self.assertEqual(result['input_dimensions'],754)
        self.assertEqual(result['evidence']['negative_symptoms'],['fever'])
        self.assertTrue(all(np.isfinite(c['score']) for c in result['candidates']))
        unknown = self.model.retrieve(present_symptoms=['cough'],topk=773)
        scores = {c['disease_id']:c['score'] for c in unknown['candidates']}
        self.assertTrue(any(abs(scores[c['disease_id']]-c['score'])>1e-5 for c in result['candidates']))

    def test_unknown_does_not_mean_negative(self):
        result = self.model.retrieve(symptom_states={'fever':'unknown'})
        self.assertEqual(result['status'],'needs_known_evidence')
        self.assertEqual(result['candidates'],[])

    def test_contradiction_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model.retrieve(present_symptoms=['发热'],absent_symptoms=['fever'])


if __name__=='__main__':
    unittest.main()
