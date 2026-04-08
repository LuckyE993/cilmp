#!/usr/bin/env python3
"""Generate llm_representations/*.pth files for CILMP with transformers.

The paper and the current CILMP implementation expect one tensor per class:

- file path: llm_representations/<dataset>/<class_name>.pth
- tensor shape: [num_hidden_layers, hidden_size]

Each row is the hidden state of the final EOS token from one decoder layer
after asking the LLM:

    "In an image, describe the distinctive visual features of {class_name}."

The current trainer hardcodes ``hidden_size=4096``, so this script validates
that by default.

Hugging Face example:
python scripts/generate_llm_representations.py \
  --dataset isic \
  --model-name meta-llama/Meta-Llama-3-8B-Instruct \
  --trust-remote-code \
  --save-metadata

ModelScope example:
python scripts/generate_llm_representations.py \
  --dataset isic \
  --model-source modelscope \
  --model-name LLM-Research/Meta-Llama-3-8B-Instruct \
  --trust-remote-code \
  --save-metadata
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class DatasetSpec:
    output_subdir: str
    classnames: Tuple[str, ...]
    aliases: Tuple[str, ...] = ()


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "adam": DatasetSpec(
        output_subdir="adam",
        classnames=(
            "normal fundus",
            "fundus with age-related macular degeneration",
        ),
    ),
    "aptos2019": DatasetSpec(
        output_subdir="aptos2019",
        classnames=(
            "no diabetic retinopathy",
            "mild diabetic retinopathy",
            "moderate diabetic retinopathy",
            "severe diabetic retinopathy",
            "proliferative diabetic retinopathy",
        ),
        aliases=("aptos",),
    ),
    "bloodmnist": DatasetSpec(
        output_subdir="bloodmnist",
        classnames=(
            "basophil",
            "eosinophil",
            "erythroblast",
            "immature granulocytes(myelocytes, metamyelocytes and promyelocytes)",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet",
        ),
    ),
    "busi": DatasetSpec(
        output_subdir="busi",
        classnames=("normal", "malignant", "benign"),
    ),
    "chaoyang": DatasetSpec(
        output_subdir="chaoyang",
        classnames=("normal", "serrated", "adenocarcinoma", "adenoma"),
    ),
    "cpn_x_ray": DatasetSpec(
        output_subdir="cpn_x_ray",
        classnames=("normal", "covid", "pneumonia"),
        aliases=("cpn-x-ray", "cpn_xray", "cpnxray"),
    ),
    "derm7pt": DatasetSpec(
        output_subdir="derm7pt",
        classnames=("carcinoma", "keratosis", "melanoma", "miscellaneous", "nevus"),
    ),
    "dermamnist": DatasetSpec(
        output_subdir="dermamnist",
        classnames=(
            "actinic keratoses and intraepithelial carcinoma",
            "basal cell carcinoma",
            "benign keratosis-like lesions",
            "dermatofibroma",
            "melanoma",
            "melanocytic nevi",
            "vascular lesions",
        ),
    ),
    "fetal_us": DatasetSpec(
        output_subdir="fetal_us",
        classnames=(
            "Fetal brain",
            "Fetal femur",
            "Fetal abdomen",
            "Fetal thorax",
            "Maternal cervix",
            "Other planes",
        ),
        aliases=("fetal-us", "fetalus"),
    ),
    "isic": DatasetSpec(
        output_subdir="isic",
        classnames=(
            "Melanoma",
            "Melanocytic nevus",
            "Basal cell carcinoma",
            "Actinic keratosis",
            "Benign keratosis",
            "Dermatofibroma",
            "Vascular lesion",
        ),
    ),
    "kvasir": DatasetSpec(
        output_subdir="kvasir",
        classnames=(
            "dyed lifted polyps",
            "dyed resection margins",
            "esophagitis",
            "normal cecum",
            "normal pylorus",
            "normal z-line",
            "polyps",
            "ulcerative colitis",
        ),
    ),
    "odir": DatasetSpec(
        output_subdir="odir",
        classnames=(
            "normal",
            "diabetes",
            "glaucoma",
            "cataract",
            "age related macular degeneration",
            "hypertension",
            "pathological myopia",
            "other diseases or abnormalities",
        ),
    ),
    "odir2": DatasetSpec(
        output_subdir="odir2",
        classnames=("normal", "cataract"),
        aliases=("odir_simple", "odir-simple"),
    ),
    "pneumonia2": DatasetSpec(
        output_subdir="pneumonia2",
        classnames=("normal", "pneumonia"),
        aliases=("pneumonia",),
    ),
    "pneumonia3": DatasetSpec(
        output_subdir="pneumonia3",
        classnames=("normal", "bacteria pneumonia", "virus pneumonia"),
    ),
}

ALIAS_TO_DATASET = {
    alias: name
    for name, spec in DATASET_SPECS.items()
    for alias in (name, *spec.aliases)
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate CILMP llm_representations .pth files by querying a "
            "decoder-only model from transformers and extracting the final "
            "EOS token hidden state from every decoder layer."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset",
        type=str,
        help="One built-in dataset name, for example: isic, adam, kvasir, pneumonia3.",
    )
    source_group.add_argument(
        "--all-datasets",
        action="store_true",
        help="Generate representations for every built-in medical dataset.",
    )
    source_group.add_argument(
        "--classes",
        nargs="+",
        help="Custom class names. Requires --output-subdir.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "llm_representations",
        help="Base output directory. Defaults to cilmp/llm_representations.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        help="Required with --classes. Output will be saved under output-dir/output-subdir.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help=(
            "Model identifier or local path. The paper used a 4096-dim "
            "decoder-only model such as LLaMA-3-8B."
        ),
    )
    parser.add_argument(
        "--model-source",
        type=str,
        default="auto",
        choices=("auto", "huggingface", "modelscope", "local"),
        help=(
            "How to resolve --model-name. 'auto' uses a local path when it exists, "
            "otherwise falls back to Hugging Face. Use 'modelscope' to download "
            "from ModelScope first."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        help="Optional model revision, branch, or tag for Hugging Face or ModelScope.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional cache directory for Hugging Face or ModelScope downloads.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use already-cached local files when resolving the model.",
    )
    parser.add_argument(
        "--modelscope-local-dir",
        type=Path,
        help="Optional explicit local directory for ModelScope snapshot_download().",
    )
    parser.add_argument(
        "--modelscope-token",
        type=str,
        help="Optional access token passed to ModelScope snapshot_download().",
    )
    parser.add_argument(
        "--question-template",
        type=str,
        default="In an image, describe the distinctive visual features of {classname}.",
        help="Prompt template used to query the LLM.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated response tokens per class.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for generation and forward passes.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "float16", "bfloat16", "float32"),
        help="Model dtype for loading.",
    )
    parser.add_argument(
        "--expected-hidden-size",
        type=int,
        default=4096,
        help="Expected LLM hidden size. CILMP currently hardcodes 4096.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="auto",
        choices=("auto", "always", "never"),
        help="Whether to wrap the question with tokenizer.apply_chat_template.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer/AutoModelForCausalLM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .pth files instead of skipping them.",
    )
    parser.add_argument(
        "--save-metadata",
        action="store_true",
        help="Save a metadata.json file next to the generated .pth files.",
    )
    return parser.parse_args()


def resolve_targets(args: argparse.Namespace) -> List[Tuple[str, Tuple[str, ...]]]:
    if args.dataset:
        dataset_name = normalize_dataset_name(args.dataset)
        spec = DATASET_SPECS[dataset_name]
        return [(spec.output_subdir, spec.classnames)]

    if args.all_datasets:
        return [
            (spec.output_subdir, spec.classnames)
            for _, spec in sorted(DATASET_SPECS.items(), key=lambda item: item[0])
        ]

    if not args.output_subdir:
        raise ValueError("--classes requires --output-subdir.")

    classnames = tuple(args.classes)
    if not classnames:
        raise ValueError("--classes requires at least one class name.")
    return [(args.output_subdir, classnames)]


def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower()
    if key not in ALIAS_TO_DATASET:
        supported = ", ".join(sorted(ALIAS_TO_DATASET))
        raise ValueError(f"Unsupported dataset '{name}'. Supported names: {supported}")
    return ALIAS_TO_DATASET[key]


def resolve_dtype(dtype_name: str, device: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if device.startswith("cuda"):
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def resolve_model_source(model_name: str, model_source: str) -> str:
    if model_source != "auto":
        return model_source
    return "local" if Path(model_name).expanduser().exists() else "huggingface"


def resolve_model_name_or_path(args: argparse.Namespace) -> Tuple[str, str]:
    source = resolve_model_source(args.model_name, args.model_source)

    if source == "local":
        model_path = Path(args.model_name).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_path}")
        return str(model_path), source

    if source == "huggingface":
        return args.model_name, source

    if source == "modelscope":
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "ModelScope support requires the 'modelscope' package to be installed."
            ) from exc

        download_kwargs = {
            "model_id": args.model_name,
            "local_files_only": args.local_files_only,
        }
        if args.revision:
            download_kwargs["revision"] = args.revision
        if args.cache_dir is not None:
            download_kwargs["cache_dir"] = str(args.cache_dir)
        if args.modelscope_local_dir is not None:
            download_kwargs["local_dir"] = str(args.modelscope_local_dir)
        if args.modelscope_token:
            download_kwargs["token"] = args.modelscope_token

        print(f"Resolving ModelScope model '{args.model_name}' ...")
        model_path = snapshot_download(**download_kwargs)
        print(f"ModelScope snapshot ready at '{model_path}'.")
        return model_path, source

    raise ValueError(f"Unsupported model source: {source}")


def build_tokenizer_load_kwargs(
    args: argparse.Namespace,
    resolved_source: str,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "trust_remote_code": args.trust_remote_code,
    }
    if resolved_source == "huggingface":
        if args.revision:
            kwargs["revision"] = args.revision
        if args.cache_dir is not None:
            kwargs["cache_dir"] = str(args.cache_dir)
        if args.local_files_only:
            kwargs["local_files_only"] = True
    return kwargs


def build_model_load_kwargs(
    args: argparse.Namespace,
    dtype: torch.dtype,
    resolved_source: str,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": args.trust_remote_code,
    }
    if resolved_source == "huggingface":
        if args.revision:
            kwargs["revision"] = args.revision
        if args.cache_dir is not None:
            kwargs["cache_dir"] = str(args.cache_dir)
        if args.local_files_only:
            kwargs["local_files_only"] = True
    return kwargs


def should_use_chat_template(tokenizer: AutoTokenizer, mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return bool(getattr(tokenizer, "chat_template", None))


def build_input_text(tokenizer: AutoTokenizer, question: str, chat_template_mode: str) -> str:
    if should_use_chat_template(tokenizer, chat_template_mode):
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return question


def get_hidden_size(model: AutoModelForCausalLM) -> Optional[int]:
    if hasattr(model.config, "hidden_size"):
        return int(model.config.hidden_size)
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "hidden_size"):
        return int(text_config.hidden_size)
    return None


def get_num_hidden_layers(model: AutoModelForCausalLM) -> Optional[int]:
    if hasattr(model.config, "num_hidden_layers"):
        return int(model.config.num_hidden_layers)
    text_config = getattr(model.config, "text_config", None)
    if text_config is not None and hasattr(text_config, "num_hidden_layers"):
        return int(text_config.num_hidden_layers)
    return None


def eos_token_ids(model: AutoModelForCausalLM, tokenizer: AutoTokenizer) -> List[int]:
    token_ids = getattr(model.generation_config, "eos_token_id", None)
    if token_ids is None:
        token_ids = tokenizer.eos_token_id
    if token_ids is None:
        return []
    if isinstance(token_ids, int):
        return [token_ids]
    return list(token_ids)


def strip_trailing_eos(tokens: Sequence[int], eos_ids: Sequence[int]) -> List[int]:
    output = list(tokens)
    while output and output[-1] in eos_ids:
        output.pop()
    return output


@torch.inference_mode()
def generate_representation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    question: str,
    device: str,
    max_new_tokens: int,
    chat_template_mode: str,
) -> Tuple[torch.Tensor, str]:
    input_text = build_input_text(tokenizer, question, chat_template_mode)
    encoded = tokenizer(input_text, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    eos_ids = eos_token_ids(model, tokenizer)
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids if eos_ids else None,
    )

    if eos_ids and generated[0, -1].item() not in eos_ids:
        eos_tensor = torch.tensor(
            [[eos_ids[0]]],
            device=generated.device,
            dtype=generated.dtype,
        )
        generated = torch.cat([generated, eos_tensor], dim=-1)

    attention_mask = torch.ones_like(generated, device=generated.device)
    outputs = model(
        input_ids=generated,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    hidden_states = outputs.hidden_states
    if not hidden_states:
        raise RuntimeError("Model forward pass did not return hidden states.")

    layer_outputs = hidden_states[1:]
    representation = torch.stack(
        [layer[0, -1, :].detach().to(torch.float32).cpu() for layer in layer_outputs],
        dim=0,
    )

    prompt_length = encoded["input_ids"].shape[-1]
    new_tokens = generated[0, prompt_length:].tolist()
    decoded = tokenizer.decode(
        strip_trailing_eos(new_tokens, eos_ids),
        skip_special_tokens=True,
    ).strip()
    return representation, decoded


def save_metadata(
    target_dir: Path,
    model_name: str,
    question_template: str,
    outputs: List[Dict[str, object]],
) -> None:
    metadata_path = target_dir / "metadata.json"
    metadata = {
        "model_name": model_name,
        "question_template": question_template,
        "classes": outputs,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    targets = resolve_targets(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dtype = resolve_dtype(args.dtype, args.device)
    model_name_or_path, resolved_source = resolve_model_name_or_path(args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        **build_tokenizer_load_kwargs(args, resolved_source),
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **build_model_load_kwargs(args, dtype, resolved_source),
    )
    model.to(args.device)
    model.eval()

    hidden_size = get_hidden_size(model)
    if hidden_size is None:
        raise RuntimeError("Could not infer model hidden_size from the transformers config.")
    if hidden_size != args.expected_hidden_size:
        raise ValueError(
            f"Model hidden_size={hidden_size}, but expected {args.expected_hidden_size}. "
            "The current CILMP code hardcodes llm_embed_dim=4096."
        )

    num_hidden_layers = get_num_hidden_layers(model)
    if num_hidden_layers is None:
        raise RuntimeError("Could not infer model num_hidden_layers from the config.")

    print(
        f"Loaded model '{args.model_name}' via {resolved_source} "
        f"(resolved='{model_name_or_path}', hidden_size={hidden_size}, "
        f"num_hidden_layers={num_hidden_layers}, dtype={dtype})."
    )

    for output_subdir, classnames in targets:
        target_dir = args.output_dir / output_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        metadata_entries: List[Dict[str, object]] = []

        print(f"\nGenerating representations for '{output_subdir}' ({len(classnames)} classes)")
        for classname in classnames:
            output_path = target_dir / f"{classname}.pth"
            question = args.question_template.format(classname=classname)

            if output_path.exists() and not args.overwrite:
                print(f"  skip: {output_path.name}")
                continue

            representation, response_text = generate_representation(
                model=model,
                tokenizer=tokenizer,
                question=question,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                chat_template_mode=args.chat_template,
            )

            if representation.shape != (num_hidden_layers, hidden_size):
                raise RuntimeError(
                    f"Unexpected representation shape {tuple(representation.shape)} for "
                    f"class '{classname}'. Expected ({num_hidden_layers}, {hidden_size})."
                )

            torch.save(representation, output_path)
            metadata_entries.append(
                {
                    "classname": classname,
                    "filename": output_path.name,
                    "question": question,
                    "response_text": response_text,
                    "shape": list(representation.shape),
                }
            )
            print(f"  saved: {output_path.name} {tuple(representation.shape)}")

        if args.save_metadata and metadata_entries:
            save_metadata(
                target_dir=target_dir,
                model_name=args.model_name,
                question_template=args.question_template,
                outputs=metadata_entries,
            )
            print(f"  metadata: {target_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
