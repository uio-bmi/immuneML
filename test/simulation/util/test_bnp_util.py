import dataclasses

import pytest
from bionumpy import DNAEncoding, AminoAcidEncoding
from bionumpy.bnpdataclass import bnpdataclass

from immuneML.simulation.util.bnp_util import make_bnp_dataclass_object_from_dicts, make_new_bnp_dataclass


def test_make_bnpdataclass_from_dict():

    dict_list = [
        {"a": 1, "b": 3, "sequence": "ACT", "c": "abc", "d": 1},
        {"a": 3, "b": 30, "sequence": "AGT", "c": "ed", "d": 0},
        {"a": 2, "b": 1, "sequence": "AC", "c": "efab", "d": 1}
    ]

    encoding_dict = {"sequence": DNAEncoding}

    obj = make_bnp_dataclass_object_from_dicts(dict_list, encoding_dict)
    print(obj)

    with pytest.raises(Exception):
        dict_list[0]["a"] = "e"
        make_bnp_dataclass_object_from_dicts(dict_list)

    with pytest.raises(Exception):
        del dict_list[0]["a"]
        make_bnp_dataclass_object_from_dicts(dict_list)


def test_make_bnp_dataclass():

    @bnpdataclass
    class DynamicBaseDC:
        sequence_aa: AminoAcidEncoding

    new_cls = make_new_bnp_dataclass([("sequence", DNAEncoding), ('signal1', int)], DynamicBaseDC)

    assert issubclass(new_cls, DynamicBaseDC)
    assert new_cls.__name__ == "DynamicDC"
    assert all(field.name in ['sequence_aa', 'sequence', 'signal1'] for field in dataclasses.fields(new_cls))
