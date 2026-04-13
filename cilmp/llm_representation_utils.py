import os.path as osp
from typing import Sequence


DATASET_NAME_TO_SUBDIR = {
    "ADAM": "adam",
    "APTOS": "aptos2019",
    "BLOODMNIST": "bloodmnist",
    "BUSI": "busi",
    "CHAOYANG": "chaoyang",
    "CPN_X_RAY": "cpn_x_ray",
    "DERM7PT": "derm7pt",
    "DERMAMNIST": "dermamnist",
    "FETAL_US": "fetal_us",
    "HICERVIX_LEVEL2": "hicervix_level2",
    "ISIC": "isic",
    "KVASIR": "kvasir",
    "ODIR": "odir",
    "ODIR2": "odir2",
    "PNEUMONIA2": "pneumonia2",
    "PNEUMONIA3": "pneumonia3",
}


def resolve_llm_representation_subdir(dataset_name: str = "", classnames: Sequence[str] = ()) -> str:
    normalized_name = dataset_name.strip().upper()
    if normalized_name in DATASET_NAME_TO_SUBDIR:
        return DATASET_NAME_TO_SUBDIR[normalized_name]

    classnames = tuple(classnames)
    if len(classnames) == 8:
        if "esophagitis" in classnames:
            return "kvasir"
        if "lymphocyte" in classnames:
            return "bloodmnist"
        return "odir"

    if len(classnames) == 7:
        if "actinic keratoses and intraepithelial carcinoma" in classnames:
            return "dermamnist"
        return "isic"

    if len(classnames) == 2:
        if "pneumonia" in classnames:
            return "pneumonia2"
        if "cataract" in classnames:
            return "odir2"
        return "adam"

    if len(classnames) == 3:
        if "malignant" in classnames:
            return "busi"
        if "covid" in classnames:
            return "cpn_x_ray"
        return "pneumonia3"

    if len(classnames) == 6:
        return "fetal_us"

    if len(classnames) == 5:
        if "nevus" in classnames:
            return "derm7pt"
        return "aptos2019"

    if len(classnames) == 4:
        return "chaoyang"

    raise ValueError(
        f"Unable to resolve llm_representations subdir for dataset_name={dataset_name!r} "
        f"and {len(classnames)} classnames."
    )


def resolve_llm_representation_path(
    classname: str,
    dataset_name: str = "",
    classnames: Sequence[str] = (),
) -> str:
    subdir = resolve_llm_representation_subdir(dataset_name=dataset_name, classnames=classnames)
    return osp.join("llm_representations", subdir, classname + ".pth")
