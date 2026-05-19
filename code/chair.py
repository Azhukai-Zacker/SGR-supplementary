# 存为 eval_chair_only.py
import argparse
import pickle
from analyze_vacuity_chair_std import load_chair_evaluator_safe, load_captions, ChairWordStandard

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cap_file", type=str, required=True)
    parser.add_argument("--coco_path", type=str, default="<PATH_TO_COCO>/annotations")
    parser.add_argument("--cache", type=str, default="chair.pkl")
    args = parser.parse_args()

    # 1. 加载评测器
    evaluator = load_chair_evaluator_safe(args.cache, args.coco_path)
    
    # 2. 计算 CHAIR
    print(f"Evaluating: {args.cap_file}")
    # 这里直接调用 evaluator 内部的方法，因为我们之前的脚本把类定义进去了
    # 注意：我们的 analyze_... 脚本里的 evaluator 是 CHAIR 类
    # CHAIR 类有一个 compute_chair 方法
    
    cap_dict = evaluator.compute_chair(args.cap_file, "image_id", "caption")
    
    # 3. 打印结果
    metrics = cap_dict['overall_metrics']
    print("\n========= FINAL RESULTS =========")
    print(f"CHAIRs: {metrics['CHAIRs'] * 100:.2f}")
    print(f"CHAIRi: {metrics['CHAIRi'] * 100:.2f}")
    print(f"Recall: {metrics['Recall'] * 100:.2f}")
    print(f"Len   : {metrics['Len']:.2f}")
    print("=================================")

if __name__ == "__main__":
    main()