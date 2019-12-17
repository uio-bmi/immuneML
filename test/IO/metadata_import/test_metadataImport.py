import shutil
from unittest import TestCase

from source.IO.metadata_import.MetadataImport import MetadataImport
from source.environment.EnvironmentSettings import EnvironmentSettings
from source.util.PathBuilder import PathBuilder


class TestMetadataImport(TestCase):
    def test_import_metadata(self):

        path = EnvironmentSettings.root_path + "test/tmp/metadata_import/"

        PathBuilder.build(path)

        with open(path + "metadata.csv", "w") as file:
            file.writelines("""filename,chain,donor,coeliac status (yes/no)
Mixcr_1234_TRA.clonotypes.TRA.txt,TRA,1234,no
Mixcr_1234_TRB.clonotypes.TRB.txt,TRB,1234,no
Mixcr_1363_TRA.clonotypes.TRA.txt,TRA,1363,yes
Mixcr_1363_TRB.clonotypes.TRB.txt,TRB,1363,no
Mixcr_1365_TRA.clonotypes.TRA.txt,TRA,1365,yes
Mixcr_1365_TRB.clonotypes.TRB.txt,TRB,1365,no""")

        mapping = MetadataImport.import_metadata(path + "metadata.csv")

        self.assertEqual(len(mapping), 6)
        self.assertTrue(all(["rep_file" in item.keys() and "metadata" in item.keys() for item in mapping]))
        self.assertTrue(all(["coeliac status (yes/no)" in item["metadata"] for item in mapping]))

        shutil.rmtree(path)
