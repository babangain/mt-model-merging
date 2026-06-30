#!/usr/bin/env python3
import os
import argparse
from typing import Any, List

import torch
from vllm import LLM, SamplingParams


def get_llm_model(llm: LLM) -> Any:
    return llm.llm_engine.model_executor.driver_worker.model_runner.model


def get_transformer_layers(llm_model: Any) -> List[Any]:
    if hasattr(llm_model, "model") and hasattr(llm_model.model, "layers"):
        return list(llm_model.model.layers)
    if hasattr(llm_model, "transformer") and hasattr(llm_model.transformer, "h"):
        return list(llm_model.transformer.h)
    raise RuntimeError("Could not find transformer layers on this vLLM model object.")


def get_mlp(layer: Any) -> Any:
    if hasattr(layer, "mlp"):
        return layer.mlp
    if hasattr(layer, "ffn"):
        return layer.ffn
    raise RuntimeError("Could not find MLP module on layer.")


def infer_intermediate_size(mlp: Any) -> int:
    if hasattr(mlp, "gate_up_proj"):
        w = getattr(mlp.gate_up_proj, "weight", None)
        if w is None:
            raise RuntimeError("gate_up_proj has no weight")
        out_features = w.shape[0]
        if out_features % 2 != 0:
            raise RuntimeError("gate_up_proj out_features not divisible by 2")
        return out_features // 2

    if hasattr(mlp, "gate_proj"):
        w = getattr(mlp.gate_proj, "weight", None)
        if w is None:
            raise RuntimeError("gate_proj has no weight")
        return w.shape[0]

    raise RuntimeError("Unsupported MLP structure. Expected gate_up_proj or gate_proj.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="fine-tuned model path/name for vLLM")
    ap.add_argument("--data_dir", type=str, required=True, help="from build script")
    ap.add_argument("--save_dir", type=str, required=True)
    ap.add_argument("--languages", type=str, default="hi bn ta te")
    ap.add_argument("--span", choices=["src", "tgt"], required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    llm = LLM(model=args.model, dtype=args.dtype,gpu_memory_utilization=0.3)
    sampling = SamplingParams(max_tokens=args.max_tokens)

    llm_model = get_llm_model(llm)
    layers = get_transformer_layers(llm_model)
    mlps = [get_mlp(l) for l in layers]

    num_layers = len(mlps)
    intermediate_size = infer_intermediate_size(mlps[0])

    langs = args.languages.split()
    lang_to_idx = {l: i for i, l in enumerate(langs)}
    lang_num = len(langs)

    over_zero = torch.zeros(num_layers, intermediate_size, lang_num, dtype=torch.int32, device="cuda")
    n_masked = torch.zeros(lang_num, dtype=torch.int64, device="cuda")

    state = {"mask": None, "lang_idx": None}

    def mlp_forward_factory(layer_idx: int):
        mlp = mlps[layer_idx]

        # Variant A: fused
        if hasattr(mlp, "gate_up_proj") and hasattr(mlp, "down_proj"):
            def forward(self, x):
                mask = state["mask"]
                lang_idx = state["lang_idx"]
                if mask is None or lang_idx is None:
                    raise RuntimeError("Mask/lang not set")

                gate_up, _ = self.gate_up_proj(x)  # [..., 2I]
                i2 = gate_up.size(-1)
                i = i2 // 2

                if gate_up.dim() == 3:
                    gate = torch.nn.SiLU()(gate_up[:, :, :i])
                    up = gate_up[:, :, i:]
                    act = gate.float()
                    m = mask.to(act.device).float().unsqueeze(-1)

                    over_zero[layer_idx, :, lang_idx] += ((act > 0).float() * m).sum(dim=(0, 1)).to(torch.int32)
                    n_masked[lang_idx] += int(m.sum().item())

                    y = gate * up
                    y, _ = self.down_proj(y)
                    return y

                if gate_up.dim() == 2:
                    gate = torch.nn.SiLU()(gate_up[:, :i])
                    up = gate_up[:, i:]
                    act = gate.float()
                    m = mask.to(act.device).float().reshape(-1, 1)

                    over_zero[layer_idx, :, lang_idx] += ((act > 0).float() * m).sum(dim=0).to(torch.int32)
                    n_masked[lang_idx] += int(m.sum().item())

                    y = gate * up
                    y, _ = self.down_proj(y)
                    return y

                raise ValueError(f"Unexpected shape: {gate_up.shape}")

            return forward

        # Variant B: qwen-like separated projections
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj") and hasattr(mlp, "down_proj"):
            def forward(self, x):
                mask = state["mask"]
                lang_idx = state["lang_idx"]
                if mask is None or lang_idx is None:
                    raise RuntimeError("Mask/lang not set")

                gate = torch.nn.SiLU()(self.gate_proj(x))
                up = self.up_proj(x)
                act = gate.float()

                if act.dim() == 3:
                    m = mask.to(act.device).float().unsqueeze(-1)
                    over_zero[layer_idx, :, lang_idx] += ((act > 0).float() * m).sum(dim=(0, 1)).to(torch.int32)
                    n_masked[lang_idx] += int(m.sum().item())
                elif act.dim() == 2:
                    m = mask.to(act.device).float().reshape(-1, 1)
                    over_zero[layer_idx, :, lang_idx] += ((act > 0).float() * m).sum(dim=0).to(torch.int32)
                    n_masked[lang_idx] += int(m.sum().item())
                else:
                    raise ValueError(f"Unexpected shape: {act.shape}")

                y = gate * up
                y = self.down_proj(y)
                return y

            return forward

        raise RuntimeError("Unsupported MLP module structure")

    # Install overrides
    for i in range(num_layers):
        mlps[i].forward = mlp_forward_factory(i).__get__(mlps[i], type(mlps[i]))

    # Run per language and save per-language files
    for lang in langs:
        lang_idx = lang_to_idx[lang]

        input_ids = torch.load(os.path.join(args.data_dir, f"input_ids.{lang}.pt"))
        src_mask = torch.load(os.path.join(args.data_dir, f"src_mask.{lang}.pt"))
        tgt_mask = torch.load(os.path.join(args.data_dir, f"tgt_mask.{lang}.pt"))

        mask = src_mask if args.span == "src" else tgt_mask

        n, t = input_ids.shape
        bs = args.batch_size
        print(f"Collecting lang={lang} span={args.span} N={n} T={t}")

        for start in range(0, n, bs):
            end = min(n, start + bs)
            batch_ids = input_ids[start:end]
            batch_mask = mask[start:end]

            state["mask"] = batch_mask.to("cuda")
            state["lang_idx"] = lang_idx

            llm.generate(prompt_token_ids=batch_ids.tolist(), sampling_params=sampling)

        out_path = os.path.join(args.save_dir, f"activation.{lang}.{args.span}.pt")
        torch.save(
            {
                "lang": lang,
                "span": args.span,
                "n": int(n_masked[lang_idx].item()),
                "over_zero": over_zero[:, :, lang_idx].detach().cpu(),
            },
            out_path,
        )
        print(f"  saved {out_path} n_masked={int(n_masked[lang_idx].item())}")

    # Save the full tensor too
    torch.save(
        {"langs": langs, "span": args.span, "n_masked": n_masked.detach().cpu(), "over_zero": over_zero.detach().cpu()},
        os.path.join(args.save_dir, f"activation.ALL.{args.span}.pt"),
    )


if __name__ == "__main__":
    main()
