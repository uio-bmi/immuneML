from unittest import TestCase

from source.util.StringHelper import StringHelper


class TestStringHelper(TestCase):
    def test_camel_case_to_words(self):
        self.assertEqual(["Repertoire", "Dataset"], StringHelper.camel_case_to_words("RepertoireDataset"))
        self.assertEqual(["Repertoire", "Dataset"], StringHelper.camel_case_to_words("repertoireDataset"))
