import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json


@DATASET_REGISTRY.register()
class HICERVIX_LEVEL2(DatasetBase):

    dataset_dir = "hicervix"

    all_class_names = [
        "Normal",
        "ECC",
        "RPC",
        "MPC",
        "PG",
        "Atrophy",
        "EMC",
        "HCG",
        "ASC-US",
        "LSIL",
        "ASC-H",
        "HSIL",
        "SCC",
        "AGC",
        "AGC-NOS",
        "AGC-FN",
        "ADC",
        "FUNGI",
        "ACTINO",
        "TRI",
        "HSV",
        "CC",
    ]

    def __init__(self, cfg):
        self.all_class_names = [
            "Normal",
            "ECC",
            "RPC",
            "MPC",
            "PG",
            "Atrophy",
            "EMC",
            "HCG",
            "ASC-US",
            "LSIL",
            "ASC-H",
            "HSIL",
            "SCC",
            "AGC",
            "AGC-NOS",
            "AGC-FN",
            "ADC",
            "FUNGI",
            "ACTINO",
            "TRI",
            "HSV",
            "CC",
        ]

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)
        split_filename = getattr(cfg.DATASET, "SPLIT_FILE", "") or "hicervix_level2.json"
        self.split_path = os.path.join(self.dataset_dir, split_filename)

        train, val, test = self.read_split(self.split_path, self.image_dir)

        super().__init__(train_x=train, val=val, test=test)

    @staticmethod
    def read_split(filepath, path_prefix):
        def _convert(items):
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath)
                item = Datum(impath=impath, label=int(label), classname=classname)
                out.append(item)
            return out

        print(f"Reading split from {filepath}")
        split = read_json(filepath)
        train = _convert(split["train"])
        test = _convert(split["test"])
        val = _convert(split.get("val", split["test"]))

        return train, val, test
