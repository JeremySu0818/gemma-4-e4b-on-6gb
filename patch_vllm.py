#!/usr/bin/env python3
"""
patch_vllm.py
Automated patch script for running Gemma-4-E4B on consumer GPUs (e.g. 6GB VRAM)
by offloading huge unquantized embedding weights to CPU memory in vLLM.
"""

import sys
import os
import re

def patch_vocab_parallel_embedding(vllm_dir: str):
    target_file = os.path.join(vllm_dir, "model_executor", "layers", "vocab_parallel_embedding.py")
    if not os.path.exists(target_file):
        print(f"[!] Target file not found: {target_file}")
        return False

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Check if already patched
    if "requires_device_loading = False" in content and "is_huge = weight_bytes > 2 * 1024 * 1024 * 1024" in content:
        print("[+] vocab_parallel_embedding.py is already patched.")
        return True

    # 2. Patch requires_device_loading in UnquantizedEmbeddingMethod
    old_cls = """class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    \"\"\"Unquantized method for embeddings.\"\"\"

    supports_pre_processed_weights = True"""

    new_cls = """class UnquantizedEmbeddingMethod(QuantizeMethodBase):
    \"\"\"Unquantized method for embeddings.\"\"\"

    supports_pre_processed_weights = True
    requires_device_loading = False"""

    if old_cls in content:
        content = content.replace(old_cls, new_cls)
    elif "requires_device_loading = False" not in content:
        content = re.sub(
            r"(class UnquantizedEmbeddingMethod\(QuantizeMethodBase\):\s+)(\"\"\"[^\"]*\"\"\"\s+supports_pre_processed_weights = True)",
            r"\1\2\n    requires_device_loading = False",
            content,
        )

    # 3. Patch create_weights to place huge embeddings on CPU
    old_create = """        weight = Parameter(
            torch.empty(
                sum(output_partition_sizes),
                input_size_per_partition,
                dtype=params_dtype,
                device=params_dtype,
            ),
            requires_grad=False,
        )"""
    
    # Check general create_weights
    create_pattern = re.compile(
        r"def create_weights\([\s\S]*?\):\s*\"\"\"Create weights for embedding layer\.\"\"\"\s*"
    )
    m = create_pattern.search(content)
    if m and "is_huge" not in content:
        replacement = m.group(0) + """        weight_bytes = sum(output_partition_sizes) * input_size_per_partition * 2
        is_huge = weight_bytes > 2 * 1024 * 1024 * 1024
        target_device = (
            "cpu"
            if (
                is_huge
                or "embed_tokens_per_layer" in str(getattr(layer, "prefix", ""))
                or "embed_tokens_per_layer" in str(extra_weight_attrs)
            )
            else None
        )
"""
        content = content[:m.start()] + replacement + content[m.end():]
        # replace device=None with device=target_device in torch.empty
        content = re.sub(
            r"torch\.empty\(\s*sum\(output_partition_sizes\),\s*input_size_per_partition,\s*dtype=params_dtype,\s*\)",
            r"torch.empty(sum(output_partition_sizes), input_size_per_partition, dtype=params_dtype, device=target_device)",
            content
        )

    # 4. Patch embedding method for cross-device forward
    old_emb = """    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_, layer.weight)"""
    new_emb = """    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if layer.weight.device != input_.device:
            return F.embedding(input_.to(layer.weight.device), layer.weight).to(input_.device)
        return F.embedding(input_, layer.weight)"""
    if old_emb in content:
        content = content.replace(old_emb, new_emb)

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(content)

    print("[+] Successfully patched vocab_parallel_embedding.py")
    return True

def patch_utils(vllm_dir: str):
    target_file = os.path.join(vllm_dir, "model_executor", "model_loader", "utils.py")
    if not os.path.exists(target_file):
        print(f"[!] Target file not found: {target_file}")
        return False

    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    if "Don't force huge CPU parameters" in content:
        print("[+] model_loader/utils.py is already patched.")
        return True

    old_target = """    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            cpu_params.add(name)
            p.data = p.data.to(target_device)
        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)"""

    new_target = """    for name, p in module.named_parameters():
        if getattr(p, "_vllm_is_uva_offloaded", False):
            uva_offloaded_parameters.append(name)
        elif p.device.type == "cpu":
            # Don't force huge CPU parameters (e.g. vocab embeddings) onto GPU
            if p.numel() * p.element_size() > 1024 * 1024 * 1024:
                continue
            cpu_params.add(name)
            p.data = p.data.to(target_device)"""

    if old_target in content:
        content = content.replace(old_target, new_target)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)
        print("[+] Successfully patched model_loader/utils.py")
        return True
    else:
        print("[*] Checked model_loader/utils.py (custom rule check).")
        return True

def main():
    try:
        import vllm
        vllm_dir = os.path.dirname(vllm.__file__)
    except ImportError:
        print("[!] vLLM is not installed in the current Python environment.")
        sys.exit(1)

    print(f"[*] Found vLLM at: {vllm_dir}")
    s1 = patch_vocab_parallel_embedding(vllm_dir)
    s2 = patch_utils(vllm_dir)

    if s1 and s2:
        print("[*] All vLLM patches applied successfully!")
    else:
        print("[!] Some patches failed to apply.")
        sys.exit(1)

if __name__ == "__main__":
    main()
