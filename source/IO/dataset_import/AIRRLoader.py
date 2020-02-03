import warnings

import airr
import numpy as np

from source.IO.dataset_import.GenericLoader import GenericLoader
from source.environment.Constants import Constants


class AIRRLoader(GenericLoader):
    '''
    Loads data from an AIRR-formatted .tsv files into a RepertoireDataset containing SequenceRepertoires.
    '''

    def _read_preprocess_file(self, filepath, params):
        col_conversion = {"sequence_aa": "sequence_aas", "sequence": "sequences",
                                "v_call": "v_genes", "j_call": "j_genes", "locus": "chains",
                                "duplicate_count": "counts"}

        df = airr.load_rearrangement(filepath)

        col_conversion = self._set_clone_id_field(col_conversion, df, params)
        df = self._set_region_type(df, params)
        df = self._filter_out_frame(df, params)
        df = df.rename(columns=col_conversion)
        df = self._set_na_to_unknown(df)
        df = self._set_mandatory_cols(df, col_conversion, params)

        return df

    def _set_region_type(self, df, params):
        ''' Check to see if the sequence type is set to 'CDR3', this is the only currently supported type. '''
        if params["sequence_type"] == "CDR3":
            df["region_types"] = params["sequence_type"]
            return df
        else:
            # not supported (yet?)
            raise ImportError("AIRRLoader: only 'CDR3' sequence type import is supported")

    def _set_clone_id_field(self, conversion_dict, df, params):
        ''' Sets the sequence_identifier as clone_id or sequence_id
        sequence_id is always present in AIRR, but clone_id may be more appropriate when present'''
        if params["clone_id_as_identifier"]:
            if not "clone_id" in df.columns:
                warnings.warn("AIRRLoader: clone_id was set as sequence identifier but does not exist in the given data.", Warning)
            conversion_dict["clone_id"] = "sequence_identifiers"
        else:
            conversion_dict["sequence_id"] = "sequence_identifiers"

        return conversion_dict

    def _filter_out_frame(self, df, params):
        ''' Filter out the out of frame sequences if that is specified'''
        if params["remove_non_productive"]:
            df = df[df["productive"] == True]

        return df

    def _set_mandatory_cols(self, df, col_conversion, params):
        ''' Set values for mandatory columns to 'None' if they are not present,
         and add necessary additional columns'''
        for mandatory_col in col_conversion.values():
            if mandatory_col not in df:
                df[mandatory_col] = None

        keep_cols = list(col_conversion.values()) + params["additional_columns"]
        df = df[keep_cols]

        return df

    def _set_na_to_unknown(self, df):
        ''' Sets all types of unknown data values to the universal UNKNOWN constant'''
        return df.replace({key: Constants.UNKNOWN for key in ["unresolved", "no data", "na", "unknown", "null", "nan", np.nan]})
