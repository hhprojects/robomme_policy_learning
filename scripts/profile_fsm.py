"""Profile FrameSamp+Modul on real frames: per-part timing, history-frame memory, and KV cache size.

Run from the repo root on a GPU node:
    XLA_PYTHON_CLIENT_PREALLOCATE=false uv run scripts/profile_fsm.py
"""
import dataclasses
import itertools
import json
import pathlib
import time

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tyro

from mme_vla_suite.models.integration.history_observation import HistAugObservation
from mme_vla_suite.models.integration.history_observation import preprocess_observation
from mme_vla_suite.models.integration.history_pi0 import make_attn_mask
from mme_vla_suite.policies import policy_config as _policy_config
from mme_vla_suite.training import config as _config
from openpi.shared import array_typing as at


@dataclasses.dataclass
class Args:
    ckpt_dir: str = "runs/ckpts/mme_vla_suite/perceptual-framesamp-modul/79999"
    config: str = "mme_vla_suite"
    history_frames: int = 64  # frames pushed into the memory buffer before profiling
    # real frames and the task goal from the full run
    frames_file: str = "runs/evaluation/perceptual-framesamp-modul/ckpt79999/seed7/frames/PickXtimes_ep0.npz"
    prompt: str = (
        "pick up the green cube and place it on the target, repeating this action three times, "
        "then press the button to stop"
    )
    num_steps: int = 10  # flow-matching steps, same as sample_actions' default
    warmup: int = 3
    iters: int = 50
    out: str = "runs/profiling/fsm.json"

def bench(fn, warmup, iters):
    """Median and p90 wall time in ms. block_until_ready waits for the GPU to finish."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "p90_ms": float(np.percentile(times, 90))}


def nbytes(tree):
    return int(sum(x.nbytes for x in jax.tree.leaves(tree) if hasattr(x, "nbytes")))


def describe(tree):
    return jax.tree.map(lambda x: {"shape": list(x.shape), "dtype": str(x.dtype), "bytes": int(x.nbytes)}, tree)


def gpu_mem():
    stats = jax.devices()[0].memory_stats() or {}
    return {"in_use": stats.get("bytes_in_use"), "peak": stats.get("peak_bytes_in_use")}


def jit_part(model, fn):
    """Like openpi's module_jit, but for any function of the model, not just its methods."""
    graphdef, state = nnx.split(model)

    def f(state, *args):
        return fn(nnx.merge(graphdef, state), *args)

    jitted = jax.jit(f)
    return lambda *args: jitted(state, *args)


def compiled_memory(model, fn, *args):
    """GPU memory one call needs, from the compiler. Live counters can't show it: loading sets their peak."""
    graphdef, state = nnx.split(model)

    def f(state, *args):
        return fn(nnx.merge(graphdef, state), *args)

    # lower() traces with placeholder values, which the observation's type checks reject
    with at.disable_typechecking():
        stats = jax.jit(f).lower(state, *args).compile().memory_analysis()
    return {
        "temp_bytes": int(stats.temp_size_in_bytes),
        "argument_bytes": int(stats.argument_size_in_bytes),
        "output_bytes": int(stats.output_size_in_bytes),
    }


def load_frames(path):
    data = np.load(path)
    return data["images"], data["wrist_images"], data["states"]


def real_obs(frames, index, prompt):
    images, wrist_images, states = frames
    index %= len(images)
    return {
        "observation/image": images[index],
        "observation/wrist_image": wrist_images[index],
        "observation/state": states[index],
        "prompt": prompt,
    }

def part_vit(m, obs):
    # embed_prefix = SigLIP on each camera image + prompt embedding lookup (the lookup is negligible)
    tokens, mask, ar_mask, _, _ = m.embed_prefix(obs)
    return tokens, mask, ar_mask


def part_memory(m, obs):
    mem_seq, mem_mask, _, _, _ = m.embed_memory(obs)
    return mem_seq, mem_mask


def part_llm(m, tokens, mask, ar_mask):
    attn_mask = make_attn_mask(mask, ar_mask)
    positions = jnp.cumsum(mask, axis=1) - 1
    _, kv_cache = m.PaliGemma.llm([tokens, None], mask=attn_mask, positions=positions)
    return kv_cache


def make_part_expert_step(dt):
    def part_expert_step(m, obs, x_t, t, prefix_mask, kv_cache, mem_seq, mem_mask):
        b = x_t.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = m.embed_suffix(obs, x_t, jnp.broadcast_to(t, b))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = m.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
            mem_seq=[None, mem_seq],
            mem_mask=[None, mem_mask],
        )
        v_t = m.action_out_proj(suffix_out[:, -m.action_horizon :])
        return x_t + dt * v_t

    return part_expert_step

def main(args: Args):
    frames = load_frames(args.frames_file)
    images, _, states = frames
    report = {
        "args": dataclasses.asdict(args),
        "device": str(jax.devices()[0]),
        "gpu_name": jax.devices()[0].device_kind,
        "frames_in_file": len(images),
    }

    # ---- load ----
    mem_before = gpu_mem()
    policy = _policy_config.create_trained_policy(_config.get_config(args.config), pathlib.Path(args.ckpt_dir))
    model = policy._model
    cfg = policy.config
    report["gpu_mem_after_load"] = gpu_mem()
    report["gpu_mem_before_load"] = mem_before

    # ---- fill the history buffer, timing add_buffer (ViT on history frames) ----
    chunk = cfg.streaming_obs_horizon  # the simulator client sends 16 frames at a time
    add_times = []
    for start in range(0, args.history_frames, chunk):
        n = min(chunk, args.history_frames - start)
        index = np.arange(start, start + n) % len(images)
        batch = {"images": images[index][:, None], "state": states[index]}  # same shape as pack_buffer
        t0 = time.perf_counter()
        policy.add_buffer(batch)  # ends with jax.device_get, so it is already synchronous
        add_times.append((time.perf_counter() - t0) * 1000)
    report["add_buffer_ms_per_16_frames"] = {"first_call_incl_compile": add_times[0], "median_after": float(np.median(add_times[1:]))}

    # ---- task 2: memory per history frame ----
    feats = policy.mem_buffer._history_feats
    one = feats[0]
    report["history_frame_host_bytes"] = {k: int(np.asarray(v).nbytes) for k, v in one.items()}
    report["history_frame_host_bytes"]["total"] = sum(report["history_frame_host_bytes"].values())
    report["history_frame_dtypes"] = {k: str(np.asarray(v).dtype) for k, v in one.items()}
    report["history_buffer_host_bytes_total"] = sum(sum(int(np.asarray(v).nbytes) for v in f.values()) for f in feats.values())
    report["history_frames_in_buffer"] = len(feats)

    # ---- build the exact observation infer() builds ----
    obs_dict = real_obs(frames, args.history_frames, args.prompt)
    inputs = policy._prepare_history(jax.tree.map(lambda x: x, obs_dict))
    inputs = policy._input_transform(inputs)
    observation = HistAugObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
    obs = jax.jit(lambda o: preprocess_observation(None, o, train=False))(observation)

    report["memory_input_to_model"] = describe(
        {k: getattr(obs, k) for k in ["static_image_emb", "static_pos_emb", "static_state_emb", "static_mask"]}
    )
    report["memory_frames_sampled"] = int(np.asarray(obs.static_mask).sum()) // cfg.token_per_image
        # ---- task 1: time each part ----
    vit = jit_part(model, part_vit)
    memory = jit_part(model, part_memory)
    llm = jit_part(model, part_llm)
    dt = -1.0 / args.num_steps
    expert_step = jit_part(model, make_part_expert_step(dt))

    tokens, prefix_mask, ar_mask = vit(obs)
    mem_seq, mem_mask = memory(obs)
    kv_cache = llm(tokens, prefix_mask, ar_mask)
    x_t = jax.random.normal(jax.random.key(0), (1, model.action_horizon, model.action_dim))
    t = jnp.float32(1.0)

    w, n = args.warmup, args.iters
    timing = {
        "vit_current_images": bench(lambda: vit(obs), w, n),
        "memory_encoder": bench(lambda: memory(obs), w, n),
        "llm_prefix": bench(lambda: llm(tokens, prefix_mask, ar_mask), w, n),
        "expert_one_step": bench(lambda: expert_step(obs, x_t, t, prefix_mask, kv_cache, mem_seq, mem_mask), w, n),
    }
    timing["expert_all_steps_est"] = {"median_ms": timing["expert_one_step"]["median_ms"] * args.num_steps}

    # whole fused sample_actions, and the full infer() including CPU-side history prep and transforms
    sample = policy._sample_actions
    key = jax.random.key(1)
    timing["sample_actions_fused"] = bench(lambda: sample(key, observation), w, n)
    next_frame = itertools.count(args.history_frames)
    timing["policy_infer_end_to_end"] = bench(lambda: policy.infer(real_obs(frames, next(next_frame), args.prompt)), w, n)
    timing["sum_of_parts"] = {
        "median_ms": sum(timing[k]["median_ms"] for k in ["vit_current_images", "memory_encoder", "llm_prefix", "expert_all_steps_est"])
    }
    report["timing"] = timing

    # ---- task 3: KV cache for one request ----
    report["prefix_tokens"] = int(tokens.shape[1])
    report["kv_cache"] = describe(kv_cache)
    report["kv_cache_total_bytes"] = nbytes(kv_cache)
    report["memory_seq_into_expert"] = describe({"mem_seq": mem_seq, "mem_mask": mem_mask})

    report["gpu_mem_end"] = gpu_mem()
    report["gpu_mem_per_request"] = compiled_memory(model, lambda m, *a: m.sample_actions(*a), key, observation)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))