# run_embedding_test.py



import torch

import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

import argparse

import time

import os

import random

import numpy as np



def setup_environment(args):

    """设置随机种子、精度和确定性模式"""

    # 1. 设置随机种子

    random.seed(args.seed)

    np.random.seed(args.seed)

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(args.seed)



    # 2. 设置确定性

    # use_deterministic_algorithms 是更现代、更全面的开关

    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)

    if args.deterministic:

        # 传统上也会设置这个，作为补充

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False



    # 3. 设置计算精度 (最关键的部分)

    # TF32仅在Ampere架构及更新的GPU上有效

    is_ampere_or_newer = False

    if torch.cuda.is_available():

        cap = torch.cuda.get_device_capability()

        if cap[0] >= 8:

            is_ampere_or_newer = True



    if args.precision == 'fp32':

        # 在Ampere+上，需显式关闭TF32才能获得纯FP32计算

        torch.backends.cuda.matmul.allow_tf32 = False

        torch.backends.cudnn.allow_tf32 = False

    elif args.precision == 'tf32' and is_ampere_or_newer:

        # 默认就是开启的，这里显式设置以明确

        torch.backends.cuda.matmul.allow_tf32 = True

        torch.backends.cudnn.allow_tf32 = True

    

    print(f"--- Environment Setup ---")

    print(f"Seed: {args.seed}")

    print(f"Deterministic: {args.deterministic}")

    print(f"Precision: {args.precision}")

    if torch.cuda.is_available():

        print(f"Device: {torch.cuda.get_device_name(0)}")

        print(f"Is Ampere or newer: {is_ampere_or_newer}")

        print(f"TF32 Matmul Allowed: {torch.backends.cuda.matmul.allow_tf32}")

    print("-------------------------")





def get_bge_embedding(model, tokenizer, text, device):

    """为BGE系列模型生成embedding的特定方法"""

    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():

        outputs = model(**inputs)

        # BGE模型的推荐做法: 取[CLS] token的输出并进行L2归一化

        embedding = outputs[0][:, 0]

        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding





def main(args):

    setup_environment(args)



    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    

    # --- 受控变量 ---

    MODEL_NAME = os.path.join(os.path.expanduser('~'), "pretrained_models/BAAI/bge-large-en-v1.5")

    QUERY = "Explain the theory of relativity in simple terms, contrasting the special and general theories."

    # ---



    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()



    # 准备保存路径

    os.makedirs(args.output_dir, exist_ok=True)

    det_str = "det" if args.deterministic else "nondet"

    output_filename = f"{args.precision}_{det_str}_run{args.run_id}.pt"

    output_path = os.path.join(args.output_dir, output_filename)



    print(f"Generating embedding for query: '{QUERY[:50]}...'")

    

    # 根据精度设置 autocast

    dtype = None

    if args.precision == 'fp16':

        dtype = torch.float16

    elif args.precision == 'bf16':

        dtype = torch.bfloat16



    start_time = time.perf_counter()

    

    if dtype:

        with torch.autocast(device_type=device.type, dtype=dtype):

            embedding = get_bge_embedding(model, tokenizer, QUERY, device)

    else: # FP32 or TF32

        embedding = get_bge_embedding(model, tokenizer, QUERY, device)



    duration = time.perf_counter() - start_time

    

    # 将embedding移回CPU以便保存

    embedding_cpu = embedding.squeeze().cpu()



    # 保存结果

    result = {

        "query": QUERY,

        "model_name": MODEL_NAME,

        "precision": args.precision,

        "deterministic": args.deterministic,

        "run_id": args.run_id,

        "seed": args.seed,

        "duration": duration,

        "embedding": embedding_cpu, # 保存为tensor

    }

    torch.save(result, output_path)

    

    print(f"Result saved to {output_path}")

    print(f"Embedding shape: {embedding_cpu.shape}")

    print(f"Time taken: {duration:.6f} seconds")

    # 打印前5个维度的值，用于快速 eyeball check

    print(f"Embedding preview: {embedding_cpu[:5].tolist()}")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test Embedding Reproducibility")

    parser.add_argument("--precision", type=str, required=True, choices=['fp32', 'tf32', 'fp16', 'bf16'], help="Computation precision.")

    parser.add_argument("--deterministic", action='store_true', help="Enable deterministic algorithms.")

    parser.add_argument("--run_id", type=int, default=0, help="An ID for the run, useful for multiple non-deterministic runs.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    parser.add_argument("--device", type=str, default="cuda", help="Device to run on ('cuda' or 'cpu').")

    parser.add_argument("--output_dir", type=str, default="embedding_results", help="Directory to save results.")

    

    args = parser.parse_args()

    main(args)
