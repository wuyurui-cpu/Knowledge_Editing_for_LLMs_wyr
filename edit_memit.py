import argparse
from typing import Dict, Any, Optional, List
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
import torch
from tqdm import tqdm

def read_dataset(file_path: str | Path) -> Any:
    with Path(file_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_dataset(data: Any, target_path: str | Path) -> None:
    target = Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def fix_random_seed(seed_value: int) -> None:
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def prepare_easyedit_root(root_path: str | Path = "external/EasyEdit") -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir(), "matplotlib-easyedit").resolve()),
    )
    abs_root = Path(root_path).resolve()
    pkg_dir = abs_root / "easyeditor"
    if not pkg_dir.exists():
        raise FileNotFoundError(
            f"EasyEdit package not found at {pkg_dir}. "
            "Clone it with: git clone https://github.com/zjunlp/EasyEdit.git external/EasyEdit"
        )
    root_str = str(abs_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _translate_device(target: Any, fallback: str) -> Any:
    if isinstance(target, str) and target.startswith("cuda"):
        return fallback
    if isinstance(target, torch.device) and target.type == "cuda":
        return torch.device(fallback)
    return target


def _adjust_args_for_device(args: tuple[Any, ...], kwargs: Dict[str, Any], fallback: str):
    new_args = tuple(_translate_device(a, fallback) for a in args)
    if "device" in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs["device"] = _translate_device(kwargs["device"], fallback)
        return new_args, new_kwargs
    return new_args, kwargs


def reroute_cuda_to_fallback(fallback_device: str = "cpu") -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "_easyedit_cuda_redirect_enabled", False):
        return fallback_device

    _orig_tensor_to = torch.Tensor.to
    _orig_module_to = torch.nn.Module.to
    _orig_tensor = torch.tensor
    _orig_zeros = torch.zeros
    _orig_ones = torch.ones
    _orig_empty = torch.empty
    _orig_full = torch.full

    def _new_tensor_to(self, *args, **kwargs):
        args, kwargs = _adjust_args_for_device(args, kwargs, fallback_device)
        return _orig_tensor_to(self, *args, **kwargs)

    def _new_module_to(self, *args, **kwargs):
        args, kwargs = _adjust_args_for_device(args, kwargs, fallback_device)
        return _orig_module_to(self, *args, **kwargs)

    def _new_tensor(*args, **kwargs):
        args, kwargs = _adjust_args_for_device(args, kwargs, fallback_device)
        return _orig_tensor(*args, **kwargs)

    def _new_zeros(*args, **kwargs):
        args, kwargs = _adjust_args_for_device(args, kwargs, fallback_device)
        return _orig_zeros(*args, **kwargs)

    def _new_ones(*args, **kwargs):
        args, kwargs = _adjust_args_for_device(args, kwargs, fallback_device)
        return _orig_ones(*args, **kwargs)

    def _new_empty(*args, **kwargs):
        args, kwargs = _adjust_args_for_device(args, kwargs, fallback_device)
        return _orig_empty(*args, **kwargs)

    def _new_full(*args, **kwargs):
        args, kwargs = _adjust_args_for_device(args, kwargs, fallback_device)
        return _orig_full(*args, **kwargs)

    torch.Tensor.to = _new_tensor_to
    torch.nn.Module.to = _new_module_to
    torch.tensor = _new_tensor
    torch.zeros = _new_zeros
    torch.ones = _new_ones
    torch.empty = _new_empty
    torch.full = _new_full
    torch._easyedit_cuda_redirect_enabled = True
    return fallback_device


def fix_nethook_trace() -> None:
    from easyeditor.util import nethook

    if getattr(nethook, "_qwen_assignment_trace_patch", False):
        return

    class CustomTrace(nethook.contextlib.AbstractContextManager):
        def __init__(
            self,
            module,
            layer=None,
            retain_output=True,
            retain_input=False,
            clone=False,
            detach=False,
            retain_grad=False,
            edit_output=None,
            stop=False,
        ):
            self._holder = self
            self.layer = layer
            if layer is not None:
                module = nethook.get_module(module, layer)

            def _wrapper(m, inputs, kwargs, output):
                if retain_input:
                    if len(inputs) > 0:
                        self._holder.input = nethook.recursive_copy(
                            inputs[0] if len(inputs) == 1 else inputs,
                            clone=clone,
                            detach=detach,
                            retain_grad=False,
                        )
                    elif kwargs is not None and "hidden_states" in kwargs:
                        self._holder.input = nethook.recursive_copy(
                            kwargs["hidden_states"],
                            clone=clone,
                            detach=detach,
                            retain_grad=False,
                        )
                    else:
                        self._holder.input = None

                if edit_output:
                    output = nethook.invoke_with_optional_args(
                        edit_output,
                        output=output,
                        layer=self.layer,
                    )

                if retain_output:
                    self._holder.output = nethook.recursive_copy(
                        output,
                        clone=clone,
                        detach=detach,
                        retain_grad=retain_grad,
                    )
                    if retain_grad:
                        output = nethook.recursive_copy(
                            self._holder.output,
                            clone=True,
                            detach=False,
                        )

                if stop:
                    raise nethook.StopForward()
                return output

            try:
                self.registered_hook = module.register_forward_hook(_wrapper, with_kwargs=True)
            except TypeError:
                def _legacy(m, inp, out):
                    return _wrapper(m, inp, None, out)
                self.registered_hook = module.register_forward_hook(_legacy)
            self.stop = stop

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()
            if self.stop and issubclass(exc_type, nethook.StopForward):
                return True
            return None

        def close(self):
            self.registered_hook.remove()

    nethook.Trace = CustomTrace
    nethook._qwen_assignment_trace_patch = True


def patch_local_stats_loading() -> None:
    from easyeditor.models.memit import memit_main

    if getattr(memit_main, "_qwen_assignment_local_stats_patch", False):
        return

    _original_layer_stats = memit_main.layer_stats

    def _local_layer_stats(
        model,
        tokenizer,
        layer_name,
        stats_dir,
        ds_name,
        to_collect,
        model_name=None,
        sample_size=None,
        precision=None,
        batch_tokens=None,
        download=True,
        progress=None,
        force_recompute=False,
        hparams=None,
    ):
        ds_path = Path(ds_name)
        if not ds_path.exists():
            return _original_layer_stats(
                model,
                tokenizer,
                layer_name,
                stats_dir,
                ds_name,
                to_collect,
                model_name=model_name,
                sample_size=sample_size,
                precision=precision,
                batch_tokens=batch_tokens,
                download=download,
                progress=progress,
                force_recompute=force_recompute,
                hparams=hparams,
            )

        from tqdm.auto import tqdm
        from easyeditor.models.rome.tok_dataset import (
            TokenizedDataset,
            dict_to_,
            flatten_masked_batch,
            length_collation,
        )
        from easyeditor.util import nethook
        from easyeditor.util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

        stat_types = {
            "mom2": SecondMoment,
            "mean": Mean,
            "norm_mean": NormMean,
        }

        raw_items = read_dataset(ds_path)
        texts = []
        for rec in raw_items:
            parts = [
                rec.get("prompt"),
                rec.get("ground_truth"),
                rec.get("target_new"),
                rec.get("rephrase_prompt"),
            ]
            text = " ".join(str(p) for p in parts if p)
            if text:
                texts.append({"text": text})

        if not texts:
            raise ValueError(f"No text records found in local stats file: {ds_path}")

        if hasattr(model.config, "n_positions"):
            npos = model.config.n_positions
        elif hasattr(model.config, "max_sequence_length"):
            npos = model.config.max_sequence_length
        elif hasattr(model.config, "max_position_embeddings"):
            npos = model.config.max_position_embeddings
        elif hasattr(model.config, "seq_length"):
            npos = model.config.seq_length
        else:
            npos = 4096

        if getattr(model.config, "model_type", "") and "qwen2" in model.config.model_type:
            npos = 4096

        if batch_tokens is None:
            batch_tokens = npos * 3
        if precision is None:
            precision = "float64"
        dtype = getattr(torch, precision)
        effective_sample_size = min(sample_size or len(texts), len(texts))

        if model_name is None:
            model_name = model.config._name_or_path.rsplit("/")[-1]
        safe_name = ds_path.stem
        size_suffix = f"_{effective_sample_size}"
        cache_path = (
            Path(stats_dir)
            / f"{model_name}/{safe_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
        )

        print(f"Computing covariance locally from {ds_path} ({effective_sample_size} samples).")
        ds = TokenizedDataset(texts, tokenizer, maxlen=npos) if not cache_path.exists() else None
        progress = progress or tqdm

        combined_stat = CombinedStat(**{k: stat_types[k]() for k in to_collect})
        loader = tally(
            combined_stat,
            ds,
            cache=(cache_path if not force_recompute else None),
            sample_size=effective_sample_size,
            batch_size=100,
            collate_fn=length_collation(batch_tokens),
            pin_memory=torch.cuda.is_available(),
            random_sample=1,
            num_workers=0,
        )
        batch_count = -(-effective_sample_size // 100)
        with torch.no_grad():
            for batch_group in progress(loader, total=batch_count):
                for batch in batch_group:
                    batch = dict_to_(batch, f"cuda:{hparams.device}")
                    with nethook.Trace(model, layer_name, retain_input=True, retain_output=False, stop=True) as tr:
                        model(**batch)
                    feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                    combined_stat.add(feats.to(dtype=dtype))
        return combined_stat

    memit_main.layer_stats = _local_layer_stats
    memit_main._qwen_assignment_local_stats_patch = True


def build_locality_inputs(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    return {
        "neighborhood": {
            "prompt": [r["locality_prompt"] for r in records],
            "ground_truth": [r["locality_ground_truth"] for r in records],
        }
    }



def main() -> None:
    parser = argparse.ArgumentParser(description="MEMIT")
    parser.add_argument("--data", default="datasets/baseline_data.json")
    parser.add_argument("--hparams", default="hparams/MEMIT/qwen2.5-0.5b.yaml")
    parser.add_argument("--out", default="output/memit_output.json")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prepare_easyedit_root()
    used_device = reroute_cuda_to_fallback()
    if used_device != "cuda":
        print(f"CUDA is unavailable; redirecting EasyEdit CUDA calls to {used_device}.")

    from easyeditor import BaseEditor, MEMITHyperParams
    fix_nethook_trace()
    patch_local_stats_loading()

    fix_random_seed(args.seed)
    all_records: List[Dict[str, Any]] = read_dataset(args.data)
    if args.limit:
        all_records = all_records[: args.limit]

    hyper = MEMITHyperParams.from_hparams(args.hparams)
    hyper.batch_size = len(all_records)
    logging.getLogger("easyeditor.editors.editor").handlers.clear()
    editor = BaseEditor.from_hparams(hyper)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    metrics, _, _ = editor.batch_edit(
        prompts=[r["prompt"] for r in all_records],
        ground_truth=[r["ground_truth"] for r in all_records],
        target_new=[r["target_new"] for r in all_records],
        rephrase_prompts=[r["rephrase_prompt"] for r in all_records],
        locality_inputs=build_locality_inputs(all_records),
        subject=[r["subject"] for r in all_records],
        sequential_edit=False,
    )

    elapsed = time.perf_counter() - start_time
    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

    write_dataset(
        {
            "algorithm": "MEMIT",
            "hparams": args.hparams,
            "num_records": len(all_records),
            "elapsed_seconds": elapsed,
            "peak_memory_gb": peak_mem,
            "case_ids": [r.get("case_id") for r in all_records],
            "metrics": metrics,
        },
        args.out,
    )

    print("\nMEMIT summary")
    print(f"records: {len(all_records)}")
    print(f"耗时: {elapsed:.2f}")
    print(f"显存占用: {peak_mem if peak_mem is not None else 'cpu'}")


if __name__ == "__main__":
    main()