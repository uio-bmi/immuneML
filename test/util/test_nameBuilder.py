from unittest import TestCase

from immuneML.util.NameBuilder import NameBuilder


class TestNameBuilder(TestCase):
    def test_build_name_from_dict(self):
        d = {
            "k1": "string1",
            "k2": 123,
            "k3": {
                "k1": {
                    "k1": "string2"
                }
            }
        }

        name = NameBuilder.build_name_from_dict(d)
        self.assertEqual(name, "k1_string1__k2_123__k3_{k1_{k1_string2}}")
