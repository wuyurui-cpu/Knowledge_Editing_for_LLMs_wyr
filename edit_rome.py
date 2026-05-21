import argparse
import json
import logging
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
from tqdm import tqdm

def read_json_file(file_path: str | Path) -> Any:
    """加载 JSON 文件"""
    with Path(file_path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def write_json_file(data: Any, output_path: str | Path) -> None:
    """保存 JSON 文件"""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def initialize_random_seed(seed_value: int) -> None:
    """设定随机种子保证可重复性"""
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def setup_easyedit_environment(custom_path: str | Path = "external/EasyEdit") -> None:
    """配置 EasyEdit 的路径和环境变量"""
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir(), "matplotlib-easyedit").resolve()))
    root_dir = Path(custom_path).resolve()
    pkg_dir = root_dir / "easyeditor"
    if not pkg_dir.exists():
        raise FileNotFoundError(
            f"EasyEdit 包未找到: {pkg_dir}。请先克隆仓库: "
            "git clone https://github.com/zjunlp/EasyEdit.git external/EasyEdit"
        )
    root_str = str(root_dir)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _convert_cuda_to_fallback(value: Any, fallback_device: str) -> Any:
    """递归替换 CUDA 设备为后备设备"""
    if isinstance(value, str) and value.startswith("cuda"):
        return fallback_device
    if isinstance(value, torch.device) and value.type == "cuda":
        return torch.device(fallback_device)
    return value


def _adapt_cuda_arguments(args: Tuple[Any, ...], kwargs: Dict[str, Any], fallback: str):
    """修改函数参数中的 CUDA 设备"""
    new_args = tuple(_convert_cuda_to_fallback(a, fallback) for a in args)
    if "device" in kwargs:
        new_kwargs = dict(kwargs)
        new_kwargs["device"] = _convert_cuda_to_fallback(kwargs["device"], fallback)
        return new_args, new_kwargs
    return new_args, kwargs


def patch_cuda_calls(fallback_target: str = "cpu") -> str:
    """劫持 PyTorch 中所有尝试使用 CUDA 的调用，重定向到可用设备"""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch, "_easyedit_cuda_redirect_enabled", False):
        return fallback_target

    # 保存原始方法
    orig_tensor_to = torch.Tensor.to
    orig_module_to = torch.nn.Module.to
    orig_tensor = torch.tensor
    orig_zeros = torch.zeros
    orig_ones = torch.ones
    orig_empty = torch.empty
    orig_full = torch.full

    def _patched_tensor_to(self, *args, **kwargs):
        args, kwargs = _adapt_cuda_arguments(args, kwargs, fallback_target)
        return orig_tensor_to(self, *args, **kwargs)

    def _patched_module_to(self, *args, **kwargs):
        args, kwargs = _adapt_cuda_arguments(args, kwargs, fallback_target)
        return orig_module_to(self, *args, **kwargs)

    def _patched_tensor(*args, **kwargs):
        args, kwargs = _adapt_cuda_arguments(args, kwargs, fallback_target)
        return orig_tensor(*args, **kwargs)

    def _patched_zeros(*args, **kwargs):
        args, kwargs = _adapt_cuda_arguments(args, kwargs, fallback_target)
        return orig_zeros(*args, **kwargs)

    def _patched_ones(*args, **kwargs):
        args, kwargs = _adapt_cuda_arguments(args, kwargs, fallback_target)
        return orig_ones(*args, **kwargs)

    def _patched_empty(*args, **kwargs):
        args, kwargs = _adapt_cuda_arguments(args, kwargs, fallback_target)
        return orig_empty(*args, **kwargs)

    def _patched_full(*args, **kwargs):
        args, kwargs = _adapt_cuda_arguments(args, kwargs, fallback_target)
        return orig_full(*args, **kwargs)

    torch.Tensor.to = _patched_tensor_to
    torch.nn.Module.to = _patched_module_to
    torch.tensor = _patched_tensor
    torch.zeros = _patched_zeros
    torch.ones = _patched_ones
    torch.empty = _patched_empty
    torch.full = _patched_full
    torch._easyedit_cuda_redirect_enabled = True
    return fallback_target


def apply_nethook_patch() -> None:
    """为 EasyEdit 的 nethook.Trace 打补丁，解决 Qwen 模型兼容性问题"""
    from easyeditor.util import nethook
    if getattr(nethook, "_qwen_assignment_trace_patch", False):
        return

    class PatchedTrace(nethook.contextlib.AbstractContextManager):
        def __init__(
            self, module, layer=None, retain_output=True, retain_input=False,
            clone=False, detach=False, retain_grad=False, edit_output=None, stop=False
        ):
            retainer = self
            self.layer = layer
            if layer is not None:
                module = nethook.get_module(module, layer)

            def _hook_fn(m, inputs, kwargs, output):
                if retain_input:
                    if len(inputs) > 0:
                        retainer.input = nethook.recursive_copy(
                            inputs[0] if len(inputs) == 1 else inputs,
                            clone=clone, detach=detach, retain_grad=False
                        )
                    elif kwargs is not None and "hidden_states" in kwargs:
                        retainer.input = nethook.recursive_copy(
                            kwargs["hidden_states"], clone=clone, detach=detach, retain_grad=False
                        )
                    else:
                        retainer.input = None
                if edit_output:
                    output = nethook.invoke_with_optional_args(edit_output, output=output, layer=self.layer)
                if retain_output:
                    retainer.output = nethook.recursive_copy(
                        output, clone=clone, detach=detach, retain_grad=retain_grad
                    )
                    if retain_grad:
                        output = nethook.recursive_copy(retainer.output, clone=True, detach=False)
                if stop:
                    raise nethook.StopForward()
                return output

            try:
                self.registered_hook = module.register_forward_hook(_hook_fn, with_kwargs=True)
            except TypeError:
                def _legacy_hook(m, inputs, output):
                    return _hook_fn(m, inputs, None, output)
                self.registered_hook = module.register_forward_hook(_legacy_hook)
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

    nethook.Trace = PatchedTrace
    nethook._qwen_assignment_trace_patch = True


def create_locality_inputs(data_entries: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    """根据记录构造 locality 评估所需的输入"""
    return {
        "neighborhood": {
            "prompt": [entry["locality_prompt"] for entry in data_entries],
            "ground_truth": [entry["locality_ground_truth"] for entry in data_entries],
        }
    }



def execute_single_edit(record: Dict[str, Any], hyperparams_yaml: str, entry_index: int) -> Dict[str, Any]:
    """对单条数据进行 ROME 编辑，返回评测结果"""
    setup_easyedit_environment()
    effective_device = patch_cuda_calls()
    if effective_device != "cuda":
        print(f"CUDA 不可用，将 EasyEdit 的 CUDA 调用重定向至 {effective_device}。")

    from easyeditor import BaseEditor, ROMEHyperParams
    apply_nethook_patch()

    hparams = ROMEHyperParams.from_hparams(hyperparams_yaml)
    logging.getLogger("easyeditor.editors.editor").handlers.clear()
    editor = BaseEditor.from_hparams(hparams)

    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    metrics, _, _ = editor.edit(
        prompts=[record["prompt"]],
        ground_truth=[record["ground_truth"]],
        target_new=[record["target_new"]],
        rephrase_prompts=[record["rephrase_prompt"]],
        locality_inputs=create_locality_inputs([record]),
        subject=[record["subject"]],
        sequential_edit=False,
    )

    elapsed_sec = time.perf_counter() - start_time
    peak_gpu_memory = None
    if torch.cuda.is_available():
        peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024 ** 3

    return {
        "case_id": entry_index,
        "prompt": record["prompt"],
        "target_new": record["target_new"],
        "elapsed_seconds": elapsed_sec,
        "peak_memory_gb": peak_gpu_memory,
        "metrics": metrics,
    }

def entry_point() -> None:
    parser = argparse.ArgumentParser(description="ROME 知识编辑脚本")
    parser.add_argument("--data", default="datasets/baseline_data.json", help="输入数据文件")
    parser.add_argument("--hparams", default="hparams/ROME/qwen2.5-0.5b.yaml", help="超参数配置文件")
    parser.add_argument("--out", default="output/rome_output.json", help="输出结果文件")
    parser.add_argument("--limit", type=int, default=None, help="限制处理的样本数量")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    args = parser.parse_args()

    initialize_random_seed(args.seed)

    all_records: List[Dict[str, Any]] = read_json_file(args.data)
    if args.limit is not None:
        all_records = all_records[:args.limit]

    results_per_case = []
    for idx, record_item in enumerate(tqdm(all_records, desc="ROME 编辑进度")):
        results_per_case.append(execute_single_edit(record_item, args.hparams, idx))

    output_data = {
        "algorithm": "ROME",
        "hparams": args.hparams,
        "num_records": len(results_per_case),
        "case_results": results_per_case,
    }
    write_json_file(output_data, args.out)

    print("\nROME 编辑完成，结果摘要如上。")


if __name__ == "__main__":
    entry_point()