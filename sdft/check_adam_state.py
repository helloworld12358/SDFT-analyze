import os
import torch


def check_optimizer_state(checkpoint_dir: str) -> None:
    """Inspect optimizer.pt in a checkpoint and report Adam second-moment stats."""
    opt_path = os.path.join(checkpoint_dir, "optimizer.pt")
    print(f"[*] Checking: {checkpoint_dir}")

    if not os.path.exists(opt_path):
        print("[!] optimizer.pt not found")
        print("    Possible reasons:")
        print("    1. Optimizer state was not saved.")
        print("    2. You used a sharded optimizer state (e.g. ZeRO) not merged.")
        print("    3. Checkpoint path is incorrect.")
        return

    print(f"[*] Found optimizer file: {opt_path}")
    print("[*] Loading optimizer state on CPU...")

    try:
        opt_data = torch.load(opt_path, map_location="cpu")
    except Exception as exc:
        print(f"[!] Failed to load optimizer state: {exc}")
        return

    if "state" not in opt_data:
        print("[!] Loaded object does not contain 'state'.")
        print(f"    Available keys: {list(opt_data.keys())}")
        return

    state_dict = opt_data["state"]
    param_groups = opt_data.get("param_groups", [])

    print(f"[*] Loaded state for {len(state_dict)} parameters.")

    found = False
    for param_id, stats in state_dict.items():
        if "exp_avg_sq" not in stats:
            continue

        found = True
        v_t = stats["exp_avg_sq"]
        print("\n[SUCCESS] Found Adam second moment 'exp_avg_sq'.")
        print(f"    - Param ID: {param_id}")
        print(f"    - Shape: {tuple(v_t.shape)}")
        print(f"    - Dtype: {v_t.dtype}")
        print(f"    - Mean: {v_t.mean().item():.6e}")
        print(f"    - Min:  {v_t.min().item():.6e}")
        print(f"    - Max:  {v_t.max().item():.6e}")
        break

    if not found:
        print("\n[!] 'exp_avg_sq' not found in optimizer state.")
        print("    You may be using non-Adam optimizer or the optimizer is not stepped yet.")
        return

    print("\n[*] Validation summary:")
    print("    Optimizer state contains Adam second-moment buffers.")
    if param_groups:
        print(f"    Param groups: {len(param_groups)}")
        print("    Match param IDs with model.named_parameters() to map layers.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(base_dir, "checkpoints", "magicoder", "sdft", "checkpoint-124")
    check_optimizer_state(target_path)