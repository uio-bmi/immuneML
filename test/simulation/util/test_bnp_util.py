import pytest

from immuneML.simulation.util.bnp_util import make_bnp_dataclass_from_dicts


def test_make_bnpdataclass_from_dict():

    dict_list = [
        {"a": 1, "b": 3, "sequence": "ACT"},
        {"a": 3, "b": 30, "sequence": "AGT"},
        {"a": 2, "b": 1, "sequence": "AC"}
    ]

    obj = make_bnp_dataclass_from_dicts(dict_list)
    print(obj)

    with pytest.raises(Exception):
        dict_list[0]["a"] = "e"
        make_bnp_dataclass_from_dicts(dict_list)

    with pytest.raises(Exception):
        del dict_list[0]["a"]
        make_bnp_dataclass_from_dicts(dict_list)