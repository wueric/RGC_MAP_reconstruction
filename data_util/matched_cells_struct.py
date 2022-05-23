from collections import namedtuple

import numpy as np

from typing import List, Dict, Tuple

from abc import ABC

CellIDIndexPair = namedtuple('CellIDIndexPair', ['cell_id', 'cell_type_key', 'idx_pos'])

from dataclasses import dataclass

import warnings


class HasCellOrdering(ABC):

    def get_reference_cell_order(self, cell_type: str) -> List[int]:
        raise NotImplementedError()


@dataclass
class CellMatch:
    '''
    Represents match between a cell in the white noise dataset
        and cells in subsequent natural scenes datasets

    wn_cell_id: white noise cell ID
    matched_nscenes_ids: Dict, (natural scenes dataset name) -> (natural scenes cell ID)
        each cell ID is a match of wn_cell_id
    '''

    wn_cell_id: int
    matched_nscenes_ids: Dict[str, List[int]]


class RFCenterStruct:

    def __init__(self,
                 initial_dataset: Dict[str, Dict[int, np.ndarray]] = None):

        if initial_dataset is None:
            self.dataset = {}  # type: Dict[str, Dict[int, np.ndarray]]
        else:
            self.dataset = initial_dataset

    def add_cell_centers(self,
                         cell_type: str,
                         centers_dict: Dict[int, np.ndarray]):

        if cell_type not in self.dataset:
            self.dataset[cell_type] = centers_dict
        else:
            for cell_id, center_coordinate in centers_dict.items():
                self.dataset[cell_type][cell_id] = center_coordinate

    def get_datadict(self):
        return self.dataset

    def get_centers_as_dict_of_arrays(self,
                                      cell_ordering: Dict[str, List[int]]) -> Dict[str, np.ndarray]:

        output_dict = {}
        for cell_type, cell_id_list in cell_ordering.items():
            relevant_centers_dict = self.dataset[cell_type]
            output_dict[cell_type] = np.array([relevant_centers_dict[cell_id] for cell_id in cell_id_list])

        return output_dict


class OrderedMatchedCellsStruct(HasCellOrdering):
    '''
    Keyed by type, cells need to be ordered
    '''

    def __init__(self):
        self.main_datadump = {}  # type: Dict[str, List[CellMatch]]
        self.typed_index_mapping = {}  # type: Dict[str, Dict[int, int]]
        self.cell_type_ordering = []  # type: List[str]

        # added extra field to do cell type recovery
        self.wn_id_to_type = {} # type: Dict[int, str]
        
    def add_typed_match(self,
                        cell_type: str,
                        reference_cell_id: int,
                        matched_cells_by_dataset: Dict[str, List[int]]) -> None:
        if cell_type not in self.main_datadump:
            self.main_datadump[cell_type] = []
            self.typed_index_mapping[cell_type] = {}
            self.cell_type_ordering.append(cell_type)

        # these two lines must be in this order, otherwise we get an off-by-one
        self.typed_index_mapping[cell_type][reference_cell_id] = len(self.main_datadump[cell_type])
        self.main_datadump[cell_type].append(CellMatch(reference_cell_id, matched_cells_by_dataset))

        self.wn_id_to_type[reference_cell_id] = cell_type

    def get_cell_type_for_cell_id(self,
                                  wn_cell_id : int) -> str:
        return self.wn_id_to_type[wn_cell_id]

    def get_match_for_ds(self,
                         cell_type: str,
                         reference_cell_id: int,
                         to_ds_name: str) -> List[int]:

        lookup_idx = self.typed_index_mapping[cell_type][reference_cell_id]
        matching_ids = self.main_datadump[cell_type][lookup_idx].matched_nscenes_ids[to_ds_name]
        return matching_ids

    def get_match_ids_for_ds(self,
                             reference_cell_id: int,
                             to_ds_name: str) -> List[int]:
        ct_lookup = self.get_cell_type_for_cell_id(reference_cell_id)
        return self.get_match_for_ds(ct_lookup, reference_cell_id, to_ds_name)

    def get_reference_cell_order(self, cell_type: str) -> List[int]:
        return [x.wn_cell_id for x in self.main_datadump[cell_type]]

    def get_cell_order_for_ds_name(self, cell_type: str, ds_name: str) -> List[List[int]]:
        return [x.matched_nscenes_ids[ds_name] for x in self.main_datadump[cell_type]]

    def get_cell_types(self) -> List[str]:
        return [x for x in self.cell_type_ordering]

    def get_n_cells_by_type(self):
        return {key: len(val) for key, val in self.main_datadump.items()}

    def get_idx_for_same_type_cell_id_list(self, cell_type: str, cell_id_list: List[int]) -> List[int]:
        return [self.typed_index_mapping[cell_type][cell_id] for cell_id in cell_id_list]

    def get_concat_idx_for_cell_id(self, wn_cell_id: int) -> int:
        cell_type = self.get_cell_type_for_cell_id(wn_cell_id)
        all_offsets = self.compute_concatenated_cell_type_index_offset()
        return self.typed_index_mapping[cell_type][wn_cell_id] + all_offsets[cell_type]

    def compute_concatenated_cell_type_index_offset(self) -> Dict[str, int]:
        ret_struct = {}
        offset = 0
        for cell_type in self.cell_type_ordering:
            ret_struct[cell_type] = offset
            offset += len(self.main_datadump[cell_type])
        return ret_struct
