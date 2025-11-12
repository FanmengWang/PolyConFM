# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import logging
import numpy as np
from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    SortDataset,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from polyconfm.datas import (
    KeyDataset,
    TokenizeDataset,
    PropertyConformerDataset,
    DistanceDataset,
    EdgeTypeDataset,
    NormalizeDataset,
    RightPadDataset,
    RightPadDataset2D,
    RightPadDatasetCoord,
    RightPadDatasetDistance,
    data_utils,
    AppendTokenDataset,
    PrependTokenDataset,
    PrependCoordTokenDataset,
    AppendCoordTokenDataset,
    PygDataset,
)
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)

task_metainfo = {
    "Egc": {
        "mean": 4.523056519156519,
        "std": 1.5609125115915392,
    },
    "Egb": {
        "mean": 4.247908749999999,
        "std": 1.9480088440998657,
    },
    "Eea": {
        "mean": 2.3071345108695653,
        "std": 1.0876226021491733,
    },
    "Ei": {
        "mean": 6.268143783783784,
        "std": 0.9900736990459043,
    },
    "Xc": {
        "mean": 36.65307914023256,
        "std": 23.733336526779983,
    },
    "EPS": {
        "mean": 4.586439790575916,
        "std": 1.1178658122169747,
    },
    "Nc": {
        "mean": 1.9472968586387436,
        "std": 0.24005882032381717,
    },
    "Eat": {
        "mean": -5.938385009765625,
        "std": 0.36132267117500305,
    }
}

@register_task("polyconfm_property")
class PolyConFMPropertyTask(UnicoreTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument("--dict-name", default="dict.txt", help="dictionary file")
        parser.add_argument("--pad-to-multiple", type=int, default=8, help="padding alignment size")
        parser.add_argument("--num-classes", default=1, type=int, help="finetune downstream task classes numbers",)
        parser.add_argument("--classification-head-name", default="classification", help="finetune downstream task name")
        
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        flag = False
        try:  
            key1 = self.args.task_name.split('_')[1]
            self.mean = task_metainfo[key1]["mean"]
            self.std = task_metainfo[key1]["std"]
            flag = True
        except Exception:
            pass
        if not flag:
            try: 
                key2 = self.args.task_name.rsplit('_fold_', 1)[0]
                self.mean = task_metainfo[key2]["mean"]
                self.std = task_metainfo[key2]["std"]
                flag = True
            except Exception:
                pass
        if not flag:
            try:
                key3 = self.args.task_name
                self.mean = task_metainfo[key3]["mean"]
                self.std = task_metainfo[key3]["std"]
                flag = True
            except Exception:
                pass
        if not flag:
            self.mean = 0.0
            self.std = 1.0
        print(f"Task {self.args.task_name} mean: {self.mean}, std: {self.std}")
        
    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(args.dict_name)
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        
        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)
        
        def PrependAndAppendCoord(dataset, pre_token, app_token):
            dataset = PrependCoordTokenDataset(dataset, pre_token)
            return AppendCoordTokenDataset(dataset, app_token)
        
        dataset = PropertyConformerDataset(
                dataset, self.args.seed, split, "repeat_unit_atom_symbols", "repeat_unit_atom_coordinates"
            )
        
        tgt_dataset = KeyDataset(dataset, "target")

        psmi_dataset = KeyDataset(dataset, "psmi") # (bsz, )
        psmi_rep_dataset = KeyDataset(dataset, "psmi_rep") # (bsz, 768)
        whole_pyg_dataset = KeyDataset(dataset, "whole_pyg") # (bsz, )
        
        repeat_unit_smi_dataset = KeyDataset(dataset, "repeat_unit_smi") # (bsz, )
        repeat_unit_actual_num_dataset = KeyDataset(dataset, "repeat_unit_actual_num") # (bsz, )

        dataset = NormalizeDataset(dataset, "repeat_unit_atom_coordinates", normalize_coord=True)
         
        src_dataset = KeyDataset(dataset, "repeat_unit_atom_symbols")
        src_dataset = TokenizeDataset(src_dataset, self.dictionary)
        src_dataset = PrependAndAppend(src_dataset, self.dictionary.bos(), self.dictionary.eos()) # (bsz, n)
        
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary)) # (bsz, n, n)
        
        coord_dataset = KeyDataset(dataset, "repeat_unit_atom_coordinates")
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppendCoord(coord_dataset, 0.0, 0.0) # (bsz, r_n, n, 3)
                
        distance_dataset = DistanceDataset(coord_dataset) # (bsz, r_n, n, n)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "psmi_rep": psmi_rep_dataset,
                    "whole_pyg": PygDataset(whole_pyg_dataset),
                    "repeat_unit_smi": repeat_unit_smi_dataset, 
                    "repeat_unit_actual_num": repeat_unit_actual_num_dataset,
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                        pad_to_multiple=self.args.pad_to_multiple
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                        pad_to_multiple=self.args.pad_to_multiple
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                        pad_to_multiple=self.args.pad_to_multiple
                    ),
                    "src_distance": RightPadDatasetDistance(
                        distance_dataset,
                        pad_idx=0,
                        pad_to_multiple=self.args.pad_to_multiple
                    )
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(psmi_dataset),
            },
        )
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))
            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
