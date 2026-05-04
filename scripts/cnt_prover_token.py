import json
import os
from collections import defaultdict
from transformers import AutoTokenizer

# ================= 配置区域 =================
# 模型路径 (请确保该路径下包含 tokenizer 文件)
MODEL_PATH = "/root/autodl-tmp/models/Goedel-Prover-V2-8B"
# 数据文件路径
DATA_FILE = "/root/autodl-tmp/step-proof/results/fdg_builder_grpo/cot_traces/formal_last_attempts.jsonl"
# ===========================================

def get_bin_name(count):
    """
    根据 token 数量返回对应的 2^n 分箱名称
    例如：128-256, 256-512
    """
    if count == 0:
        return "0"
    
    # 定义分箱边界 (2 的幂次)
    # 从 128 开始，也可以根据需求从 0 或 64 开始
    boundaries = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    
    for i, boundary in enumerate(boundaries):
        if count < boundary:
            if i == 0:
                return f"0-{boundary}"
            else:
                return f"{boundaries[i-1]}-{boundary}"
    
    return f"{boundaries[-1]}+"

def main():
    # 1. 加载 Tokenizer
    print(f"正在加载 tokenizer 从：{MODEL_PATH} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    except Exception as e:
        print(f"加载模型失败：{e}")
        print("请检查模型路径是否正确，或尝试添加 trust_remote_code=True")
        return

    # 2. 初始化统计容器
    bin_counts = defaultdict(int)
    total_lines = 0
    error_lines = 0
    max_tokens = 0

    # 3. 读取文件并统计
    print(f"正在处理文件：{DATA_FILE} ...")
    if not os.path.exists(DATA_FILE):
        print(f"错误：文件不存在 {DATA_FILE}")
        return

    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                raw_output = data.get("raw_output", "")
                
                if not isinstance(raw_output, str):
                    raw_output = str(raw_output)
                
                # 计算 token 数量
                # 注意：不同模型的 tokenizer 行为可能不同，这里使用 encode
                tokens = tokenizer.encode(raw_output)
                token_count = len(tokens)
                
                # 更新统计
                bin_name = get_bin_name(token_count)
                bin_counts[bin_name] += 1
                total_lines += 1
                if token_count > max_tokens:
                    max_tokens = token_count
                
                # 进度打印
                if line_num % 1000 == 0:
                    print(f"已处理 {line_num} 行...")
                    
            except json.JSONDecodeError:
                error_lines += 1
            except Exception as e:
                print(f"处理第 {line_num} 行时出错：{e}")
                error_lines += 1

    # 4. 输出结果
    print("\n" + "="*30)
    print("Token 分布统计结果 (基于 2^n 分箱)")
    print("="*30)
    print(f"总处理行数：{total_lines}")
    print(f"错误/跳过行数：{error_lines}")
    print(f"最大 Token 数：{max_tokens}")
    print("-" * 30)
    print(f"{'分箱范围':<15} | {'数量':<10} | {'占比':<10}")
    print("-" * 30)
    
    # 按分箱名称排序 (简单字符串排序可能不准确，建议按边界排序，这里为了简单直接输出)
    # 为了更好看，我们手动定义顺序
    ordered_bins = ["0-128", "128-256", "256-512", "512-1024", "1024-2048", 
                    "2048-4096", "4096-8192", "8192-16384", "16384-32768", "32768-65536", "65536+"]
    
    for bin_name in ordered_bins:
        count = bin_counts.get(bin_name, 0)
        if count > 0 or bin_name in ["0-128", "128-256", "256-512", "512-1024"]: # 只显示前几个或有数据的
            percentage = (count / total_lines * 100) if total_lines > 0 else 0
            print(f"{bin_name:<15} | {count:<10} | {percentage:.2f}%")
            
    # 显示其他有数据的分箱
    for bin_name, count in bin_counts.items():
        if bin_name not in ordered_bins and count > 0:
            percentage = (count / total_lines * 100) if total_lines > 0 else 0
            print(f"{bin_name:<15} | {count:<10} | {percentage:.2f}%")

    print("="*30)

if __name__ == "__main__":
    main()