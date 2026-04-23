import os

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import read_json



@DATASET_REGISTRY.register()
class KVASIR(DatasetBase):

    dataset_dir = "kvasir"

    all_class_names = ['dyed lifted polyps',
                        'dyed resection margins',
                        'esophagitis',
                        'normal cecum',
                        'normal pylorus',
                        'normal z-line',
                        'polyps',
                        'ulcerative colitis']

    def __init__(self, cfg):
        self.all_class_names = ['dyed lifted polyps',
                                'dyed resection margins',
                                'esophagitis',
                                'normal cecum',
                                'normal pylorus',
                                'normal z-line',
                                'polyps',
                                'ulcerative colitis']

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "kvasir.json")

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
