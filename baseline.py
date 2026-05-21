import argparse
import json
import random
import re
import torch
from pathlib import Path
from typing import Any, Dict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 数据读写工具方法
def read_json_file(file_path: str | Path) -> Any:
    with Path(file_path).open(mode='r', encoding='utf-8') as file:
        return json.load(file)

def write_json_file(content: Any, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open(mode='w', encoding='utf-8') as file:
        json.dump(content, file, ensure_ascii=False, indent=2)

# 固定随机种子保证实验可复现
def fix_random_seed(seed_value: int) -> None:
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

# 文本标准化处理
def clean_text(raw_text: str) -> str:
    return " ".join(raw_text.lower().strip().split())

# 校验答案是否存在于模型输出中
def check_answer_exist(model_out: str, ans) -> bool:
    if ans is None or ans == "":
        return False
    return clean_text(ans) in clean_text(model_out)

# 加载语言模型与分词器
def load_model_and_tokenizer(model_path: str, data_type: str = "auto", map_device: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model_dtype = "auto"
    if data_type == "float16":
        model_dtype = torch.float16
    elif data_type == "bfloat16":
        model_dtype = torch.bfloat16
    elif data_type == "float32":
        model_dtype = torch.float32

    llm_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=model_dtype, device_map=map_device, trust_remote_code=True
    )
    llm_model.eval()
    return llm_model, tokenizer

# 模型推理生成，过滤无效字符
@torch.no_grad()
def model_inference(llm, tok, input_prompt: str, new_tokens: int = 5) -> str:
    inputs = tok(input_prompt, return_tensors="pt").to(llm.device)
    
    outputs = llm.generate(
        **inputs,
        max_new_tokens=new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    
    # 清洗输出：去除选项、横线、换行等冗余内容
    result = tok.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    result = re.sub(r'[_A-Za-z0-9]\.', '', result)
    result = re.sub(r'_+', '', result)
    result = re.sub(r'\n+', ' ', result)
    result = result.strip().split('.')[0]
    return result[:15]

# 处理单条数据记录
def process_single_record(llm, tok, record: Dict[str, Any], gen_tokens: int) -> Dict[str, Any]:
    base_out = model_inference(llm, tok, record["prompt"], gen_tokens)
    rephrase_out = model_inference(llm, tok, record["rephrase_prompt"], gen_tokens)
    local_out = model_inference(llm, tok, record["locality_prompt"], gen_tokens)

    return {
        **record,
        "baseline_output": base_out,
        "baseline_rephrase_output": rephrase_out,
        "baseline_locality_output": local_out,
        "baseline_contains_target": check_answer_exist(base_out, record["target_new"]),
        "baseline_contains_ground_truth": check_answer_exist(base_out, record.get("ground_truth")),
        "locality_contains_ground_truth": check_answer_exist(local_out, record.get("locality_ground_truth")),
    }

# 主执行函数
def run_main() -> None:
    arg_parser = argparse.ArgumentParser(description="模型知识编辑基线测试脚本")
    arg_parser.add_argument("--model", default="Qwen2.5-0.5B")
    arg_parser.add_argument("--data", default="datasets/baseline_data.json")
    arg_parser.add_argument("--out", default="output/baseline_output.json")
    arg_parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    arg_parser.add_argument("--device-map", default="auto")
    arg_parser.add_argument("--max-new-tokens", type=int, default=32)
    arg_parser.add_argument("--seed", type=int, default=0)
    arguments = arg_parser.parse_args()

    fix_random_seed(arguments.seed)
    data_records = read_json_file(arguments.data)
    llm_model, tokenizer = load_model_and_tokenizer(arguments.model, arguments.dtype, arguments.device_map)

    eval_results = []
    for item in tqdm(data_records, desc="基线测试运行中"):
        eval_results.append(process_single_record(llm_model, tokenizer, item, arguments.max_new_tokens))

    write_json_file({
        "model_name": arguments.model,
        "total_records": len(eval_results),
        "evaluation_results": eval_results
    }, arguments.out)

    # 格式化打印对齐结果
    print("\n" + "="*75)
    print(f"{'ID':<5}{'Model Output':<20}{'Contains New':<15}{'Contains Old':<15}")
    print("="*75)
    for index, result_item in enumerate(eval_results, 1):
        print(f"{index:<5}{result_item['baseline_output']:<20}{str(result_item['baseline_contains_target']):<15}{str(result_item['baseline_contains_ground_truth']):<15}")
    print("="*75)

if __name__ == "__main__":
    run_main()