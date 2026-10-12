import shutil
import zipfile

from immuneML.app.ImmuneMLApp import compress_results
from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder


def test_compress_results():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "compress_results")
    result_path = PathBuilder.build(path / "output")
    (result_path / "log.txt").write_text("done")

    zip_path = compress_results(result_path)
    assert zip_path.name == "output.zip" and zip_path.is_file()
    with zipfile.ZipFile(zip_path) as archive:
        assert "output/log.txt" in archive.namelist()
    assert result_path.is_dir()

    shutil.rmtree(path)
