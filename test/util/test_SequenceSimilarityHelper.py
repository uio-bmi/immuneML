import shutil

import numpy as np
import pandas as pd

from immuneML.environment.EnvironmentSettings import EnvironmentSettings
from immuneML.util.PathBuilder import PathBuilder
from immuneML.util.SequenceSimilarityHelper import SequenceSimilarityHelper


def get_compairr_path():
    compairr_path = next((p for p in EnvironmentSettings.compairr_paths if p.exists()), None)
    assert compairr_path is not None, "CompAIRR binary not found; build it to run this test"
    return str(compairr_path)


def test_smallest_distance_to_reference():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "sequence_similarity_distance/")

    reference = pd.Series(["CASSLGTEAFF", "CASSPQRGYTF"])
    # identical, one substitution, two substitutions, one deletion, unrelated; the index is kept in the result
    query = pd.Series(["CASSLGTEAFF", "CASSLGTEAFY", "CASSLGTEAYY", "CASSLGEAFF", "CAWSVGDTQYW"],
                      index=[10, 11, 12, 13, 14])

    distances = SequenceSimilarityHelper.smallest_distance_to_reference(query, reference, 2, path / "subst",
                                                                        compairr_path=get_compairr_path())
    assert list(distances.index) == [10, 11, 12, 13, 14]
    assert distances.iloc[:3].tolist() == [0, 1, 2]
    # without indels, the shorter sequence never matches
    assert np.isnan(distances.iloc[3]) and np.isnan(distances.iloc[4])

    distances = SequenceSimilarityHelper.smallest_distance_to_reference(query, reference, 2, path / "indels",
                                                                        compairr_path=get_compairr_path(), indels=True)
    assert distances.iloc[3] == 1

    # with distance 0, only exact copies count and CompAIRR is not needed
    distances = SequenceSimilarityHelper.smallest_distance_to_reference(query, reference, 0, path / "exact")
    assert distances.iloc[0] == 0 and distances.iloc[1:].isna().all()

    shutil.rmtree(path)


def test_similarity_components():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "sequence_similarity_components/")

    # CASSLF - CASSLY - CASKLY form a chain at one substitution each, so they are one component even though the ends
    # are two substitutions apart; CASLF joins them only with indels
    sequences = pd.Series(["CASSLF", "CASSLY", "CASKLY", "CGGGGG", "CASSLF", "CASLF"])

    components = SequenceSimilarityHelper.similarity_components(sequences, 1, path / "subst",
                                                                compairr_path=get_compairr_path())
    assert components.iloc[0] == components.iloc[1] == components.iloc[2] == components.iloc[4]
    assert components.iloc[3] != components.iloc[0] and components.iloc[5] != components.iloc[0]
    assert components.nunique() == 3

    components = SequenceSimilarityHelper.similarity_components(sequences, 1, path / "indels",
                                                                compairr_path=get_compairr_path(), indels=True)
    assert components.iloc[5] == components.iloc[0]
    assert components.nunique() == 2

    shutil.rmtree(path)


def test_receptors_are_similar_only_if_every_chain_is():
    path = PathBuilder.remove_old_and_build(EnvironmentSettings.tmp_test_path / "sequence_similarity_receptors/")

    # receptors 0 and 1 share a beta chain but their alpha chains differ a lot; 0 and 2 are one substitution apart on
    # both chains
    receptors = pd.DataFrame({'TRA': ["CAVRDSNYQLIW", "CILRDGQKLLF", "CAVRDSNYQLIF"],
                              'TRB': ["CASSLGTEAFF", "CASSLGTEAFF", "CASSLGTEAFY"]})

    components = SequenceSimilarityHelper.similarity_components(receptors, 1, path / "both",
                                                                compairr_path=get_compairr_path())
    assert components.iloc[0] == components.iloc[2] != components.iloc[1]

    # comparing the beta chain only, all three are similar
    components = SequenceSimilarityHelper.similarity_components(receptors[['TRB']], 1, path / "beta",
                                                                compairr_path=get_compairr_path())
    assert components.nunique() == 1

    reference = receptors.iloc[[0]]
    query = receptors.iloc[[1, 2]]
    distances = SequenceSimilarityHelper.smallest_distance_to_reference(query, reference, 2, path / "distance",
                                                                        compairr_path=get_compairr_path())
    # receptor 1: identical beta, but alpha too far, so not within 2; receptor 2: both chains one substitution away
    assert np.isnan(distances.iloc[0]) and distances.iloc[1] == 1

    distances = SequenceSimilarityHelper.smallest_distance_to_reference(query[['TRB']], reference[['TRB']], 2,
                                                                        path / "distance_beta",
                                                                        compairr_path=get_compairr_path())
    assert distances.tolist() == [0, 1]

    shutil.rmtree(path)
