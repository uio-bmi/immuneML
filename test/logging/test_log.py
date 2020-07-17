import io
import sys
from unittest import TestCase

from source.environment.EnvironmentSettings import EnvironmentSettings
from source.logging.LogLevel import LogLevel
from source.logging.Logger import trace


class TestLog(TestCase):

    def test_trace(self):
        @trace
        class TestClass:
            def __init__(self, a):
                self.a = a

            def print_a(self, desc):
                print(desc + ": " + self.a)

        with self.assertRaises(Exception):

            output = io.StringIO()
            stdout = sys.stdout
            sys.stdout = output
            test_obj = TestClass(3)
            test_obj.print_a("sample desc")
            sys.stdout = stdout
            out_text = str(output.getvalue())
            self.assertTrue("Entering: __init__ with parameters" in out_text and "3)" in out_text)
            self.assertTrue("Exiting: __init__" in out_text)
            self.assertTrue("Entering: print_a with parameters" in out_text and "sample desc" in out_text)
            self.assertTrue("Exception in print_a : can only concatenate str (not \"int\") to str" in out_text)
            self.assertTrue("Exiting: print_a" in out_text)

        EnvironmentSettings.log_level = LogLevel.NONE
        output = io.StringIO()
        stdout = sys.stdout
        sys.stdout = output
        test_obj = TestClass("3")
        test_obj.print_a("sample desc")
        sys.stdout = stdout
        out_text = str(output.getvalue())
        self.assertEqual("sample desc: 3\n", out_text)
