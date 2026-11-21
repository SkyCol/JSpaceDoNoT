"""Offline unit tests for GSM8K preparation, no HF download needed."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('prep_gsm',Path(__file__).resolve().parents[1]/'scripts'/'prepare_gsm8k.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)

class GSM8KPreparation(unittest.TestCase):
    def test_gold(self):
        self.assertEqual(prep.gold_from_official_answer('Multiply.\n#### 1,234'), '1234')
        self.assertEqual(prep.gold_from_official_answer('Reasoning.\n#### -2.50'), '-2.5')
    def test_bad_gold_rejected(self):
        for answer in ['foo', '#### 3 cats', '#### 3\nextra text', '#### NaN']:
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                prep.gold_from_official_answer(answer)
    def test_rows(self):
        rows, ids = prep.prepare_rows([{'question': 'How many?', 'answer': 'Work\n#### 4'}], 'train')
        self.assertEqual(rows[0]['gold'], '4')
        self.assertEqual(len(ids), 1)
    def test_duplicate_rejected(self):
        data = [{'question': ' A  B ', 'answer':'#### 1'}, {'question': 'A B','answer':'#### 2'}]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prep.prepare_rows(data, 'train')

if __name__ == '__main__':
    unittest.main()
