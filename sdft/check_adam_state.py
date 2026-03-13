import torch
import os
import sys

def check_optimizer_state(checkpoint_dir):
    """
    读取并检查指定 Checkpoint 目录下的优化器状态，寻找二阶动量。
    """
    # 1. 构造优化器文件的路径
    # 通常 HuggingFace Trainer 会将其保存为 optimizer.pt
    opt_path = os.path.join(checkpoint_dir, "optimizer.pt")

    print(f"[*] 正在检查路径: {checkpoint_dir}")

    # 检查文件是否存在
    if not os.path.exists(opt_path):
        print(f"[!] 错误: 在该目录下未找到 'optimizer.pt'。")
        print("    可能原因：")
        print("    1. 训练时使用了 DeepSpeed ZeRO 且未合并权重（文件可能分散在子文件夹中）。")
        print("    2. 训练脚本没有保存 optimizer state（只保存了模型权重）。")
        print("    3. 文件名不同（例如使用了自定义的保存逻辑）。")
        return

    print(f"[*] 发现优化器文件: {opt_path}")
    print("[*] 正在加载文件 (map_location='cpu')... 这可能需要几秒钟...")

    try:
        # 加载优化器状态，直接加载到 CPU 以避免显存爆炸
        opt_data = torch.load(opt_path, map_location="cpu")
    except Exception as e:
        print(f"[!] 加载失败: {e}")
        return

    # 2. 检查数据结构
    # PyTorch 优化器保存的字典通常包含 'state' 和 'param_groups'
    if 'state' not in opt_data:
        print("[!] 警告: 加载的字典中不包含 'state' 键，结构可能非标准。")
        print(f"    可用键: {opt_data.keys()}")
        return

    state_dict = opt_data['state']
    param_groups = opt_data.get('param_groups', [])

    print(f"[*] 优化器状态加载成功。包含 {len(state_dict)} 个参数的状态信息。")

    # 3. 寻找二阶动量 (exp_avg_sq)
    # 我们遍历第一个非空的参数状态来检查
    found_exp_avg_sq = False
    example_shape = None
    
    for param_id, stats in state_dict.items():
        if 'exp_avg_sq' in stats:
            found_exp_avg_sq = True
            example_shape = stats['exp_avg_sq'].shape
            
            # 打印一个样本数据的统计信息
            v_t = stats['exp_avg_sq']
            print(f"\n[SUCCESS] 成功找到二阶动量 'exp_avg_sq'！")
            print(f"    - 参数 ID: {param_id}")
            print(f"    - 张量形状: {v_t.shape} (这应该对应某个 LoRA 矩阵或 Bias)")
            print(f"    - 数据类型: {v_t.dtype}")
            print(f"    - 均值: {v_t.mean().item():.6e}")
            print(f"    - 最小值: {v_t.min().item():.6e}")
            print(f"    - 最大值: {v_t.max().item():.6e}")
            break
    
    if not found_exp_avg_sq:
        print("\n[!] 警告: 在 state 中未找到 'exp_avg_sq'。")
        print("    可能原因：")
        print("    1. 使用的不是 Adam/AdamW，而是 SGD (没有二阶动量)。")
        print("    2. 这是一个刚初始化的优化器，还没进行过 step 更新。")
    else:
        print("\n[*] 验证结论：")
        print("    你的 Checkpoint 中包含完整的 Adam 状态。")
        print("    你可以直接读取 `Adam.state[id]['exp_avg_sq']` 作为 Hessian 对角线的近似。")
        
        if len(param_groups) > 0:
            print(f"    优化器包含 {len(param_groups)} 个参数组。")
            # 简单的 LoRA 检查：看参数组里的学习率或名称
            print("    注意：optimizer.pt 使用整数 ID 索引参数。")
            print("    你需要将模型加载后，通过 `model.named_parameters()` 的 id 与此处对应，")
            print("    才能确定哪个 'exp_avg_sq' 对应哪个 LoRA 层。")

if __name__ == "__main__":
    # 用户指定的具体路径
    target_path = "/inspire/hdd/project/continuinglearinginlm/weiyuqi-CZXS25110007/sdft/checkpoints/magicoder/sdft/checkpoint-124"
    
    check_optimizer_state(target_path)