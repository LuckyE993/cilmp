import os.path as osp
from typing import Sequence

"""LLM 类别表示文件路径解析工具。

项目中每个类别都有一个离线提取的 LLM 隐表示文件（`.pth`）。
该模块根据数据集名/类别集合，定位对应子目录与文件路径。
"""

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
    "HICERVIX_5CLS": "hicervix_5cls",
    "HICERVIX_LEVEL2": "hicervix_level2",
    "ISIC": "isic",
    "KVASIR": "kvasir",
    "ODIR": "odir",
    "ODIR2": "odir2",
    "PNEUMONIA2": "pneumonia2",
    "PNEUMONIA3": "pneumonia3",
}


def resolve_llm_representation_subdir(dataset_name: str = "", classnames: Sequence[str] = ()) -> str:
    """解析 LLM 表示所在子目录。

    优先级：
    1) 直接用标准化后的数据集名映射；
    2) 若数据集名不可用，则用类别数量 + 关键类别词做兜底推断。
    """
    normalized_name = dataset_name.strip().upper()
    if normalized_name in DATASET_NAME_TO_SUBDIR:
        return DATASET_NAME_TO_SUBDIR[normalized_name]

    # 转 tuple 是为了保证只读语义，同时便于多次条件判断。
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

    # 仍无法推断时，显式报错，避免静默加载错误数据。
    raise ValueError(
        f"Unable to resolve llm_representations subdir for dataset_name={dataset_name!r} "
        f"and {len(classnames)} classnames."
    )


def resolve_llm_representation_path(
    classname: str,
    dataset_name: str = "",
    classnames: Sequence[str] = (),
) -> str:
    """返回某个类别的 LLM 表示文件路径。"""
    subdir = resolve_llm_representation_subdir(dataset_name=dataset_name, classnames=classnames)
    return osp.join("llm_representations", subdir, classname + ".pth")
