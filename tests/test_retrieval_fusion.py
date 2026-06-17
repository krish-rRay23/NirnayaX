import unittest
from src.nirnayax.retrieval.fusion import reciprocal_rank_fusion

class TestRetrievalFusion(unittest.TestCase):
    def test_rrf_scoring(self):
        vec_docs = ["doc_A", "doc_B", "doc_C"]
        lex_docs = ["doc_B", "doc_A", "doc_D"]

        fused = reciprocal_rank_fusion(vec_docs, lex_docs, k=60)
        top_doc = fused[0]["doc_id"]
        # Both doc_A and doc_B have high ranks in both lists
        self.assertIn(top_doc, ["doc_A", "doc_B"])
        self.assertEqual(len(fused), 4)

if __name__ == '__main__':
    unittest.main()
