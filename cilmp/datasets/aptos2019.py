import os
import pickle

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing, read_json



@DATASET_REGISTRY.register()
class APTOS(DatasetBase):

    dataset_dir = "aptos2019"

    all_class_names = [
                        'no diabetic retinopathy',
                        'mild diabetic retinopathy',
                        'moderate diabetic retinopathy',
                        'severe diabetic retinopathy',
                        'proliferative diabetic retinopathy'
                    ]

    def __init__(self, cfg):

        self.all_class_names = [
                                'no diabetic retinopathy',
                                'mild diabetic retinopathy',
                                'moderate diabetic retinopathy',
                                'severe diabetic retinopathy',
                                'proliferative diabetic retinopathy'
                            ]

        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir)
        self.split_path = os.path.join(self.dataset_dir, "aptos2019.json")
        self.split_fewshot_dir = os.path.join(self.dataset_dir, "split_fewshot")
        mkdir_if_missing(self.split_fewshot_dir)

        train, val, test = self.read_split(self.split_path, self.image_dir)

        num_shots = cfg.DATASET.NUM_SHOTS
        if num_shots >= 1:
            seed = cfg.SEED
            preprocessed = os.path.join(
                self.split_fewshot_dir, f"shot_{num_shots}-seed_{seed}.pkl"
            )

            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train, val = data["train"], data["val"]
            else:
                train = self.generate_fewshot_dataset(train, num_shots=num_shots)
                val = self.generate_fewshot_dataset(val, num_shots=min(num_shots, 4))
                data = {"train": train, "val": val}
                print(f"Saving preprocessed few-shot data to {preprocessed}")
                with open(preprocessed, "wb") as file:
                    pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

        # subsample = cfg.DATASET.SUBSAMPLE_CLASSES
        # train, val, test = OxfordPets.subsample_classes(train, val, test, subsample=subsample)
        # print('list.train:', len(train))

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
