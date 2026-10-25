"""
Text generation script.
Supports temperature, top-k, top-p, and comparing with HuggingFace GPT-2.
"""
import argparse
import torch
import tiktoken

from model import GPT, GPTConfig


def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model_cfg = GPTConfig(**ckpt["config"]["model"])
    model = GPT(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt.get("step", None)


@torch.no_grad()
def generate_text(model, enc, prompt, device,
                  max_new_tokens=100, temperature=1.0,
                  top_k=None, top_p=None, num_samples=1, seed=1337):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    tokens = enc.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device)
    idx = idx.unsqueeze(0).repeat(num_samples, 1)

    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    results = []
    for i in range(num_samples):
        seq = out[i].tolist()
        results.append(enc.decode(seq))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None,
                        help="path to checkpoint; if None, use HF GPT-2")
    parser.add_argument("--prompt", type=str, default="Hello, I'm a language model,")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        print("loading HuggingFace GPT-2 (for reference)...")
        model = GPT.from_pretrained("gpt2").to(device)
    else:
        print(f"loading checkpoint: {args.ckpt}")
        model, step = load_model_from_ckpt(args.ckpt, device)
        print(f"checkpoint step: {step}")

    enc = tiktoken.get_encoding("gpt2")

    print("=" * 60)
    print(f"prompt: {args.prompt}")
    print(f"temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}")
    print("=" * 60)

    results = generate_text(
        model, enc, args.prompt, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    for i, text in enumerate(results):
        print(f"\n--- sample {i+1} ---")
        print(text)


if __name__ == "__main__":
    main()