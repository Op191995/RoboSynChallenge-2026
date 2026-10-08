# ----------------------------------------------------------------------------
# LeRobot ACT Policy Adapter for RoboSynChallenge
#
# Follows the same unified evaluation interface as policy/pi0:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info, truncated
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import os

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from policy.inference_timing import finish_inference, start_inference

# ---- BEGIN OVERLAP BLEND PATCH ----
# 由 patch_overlap_blend.py 注入；RSC_BLEND_EXEC=0/未设置时行为与官方完全一致。
_BLEND_EXEC_RAW = os.environ.get("RSC_BLEND_EXEC", "0")
_BLEND_STATE = {"queue": None, "used": 0, "exec": 0, "chunk": 0}


def _blend_enabled():
    return _BLEND_STATE["exec"] > 0


def _blend_reset():
    _BLEND_STATE["queue"] = None
    _BLEND_STATE["used"] = 0


def _blend_init(model):
    chunk = int(getattr(model.config, "chunk_size", 0) or 0)
    req = int(_BLEND_EXEC_RAW)
    if req < 0:
        req = chunk // 2
    _BLEND_STATE["chunk"] = chunk
    _BLEND_STATE["exec"] = max(0, min(req, chunk)) if chunk > 0 else 0
    _blend_reset()
    if _BLEND_STATE["exec"]:
        print(f"[overlap-blend] enabled: chunk={chunk} exec={_BLEND_STATE['exec']} "
              f"overlap={chunk - _BLEND_STATE['exec']} inference x{chunk / max(_BLEND_STATE['exec'], 1):.1f}")


def _blend_will_infer():
    q = _BLEND_STATE["queue"]
    return q is None or len(q) == 0 or _BLEND_STATE["used"] >= _BLEND_STATE["exec"]


def _blend_select(model, batch):
    """每 exec 步重新预测一次 chunk，重叠区线性混合后再执行。"""
    if not _blend_enabled():
        return model.select_action(batch)
    if _blend_will_infer():
        chunk = model.predict_action_chunk(batch)
        if chunk.ndim == 3:
            chunk = chunk[0]
        q = _BLEND_STATE["queue"]
        if q is not None and len(q) > 0:
            prev = torch.stack(q, dim=0).to(device=chunk.device, dtype=chunk.dtype)
            k = int(min(prev.shape[0], chunk.shape[0]))
            # 权重从 1/(k+1) 线性升到 k/(k+1)：切换瞬间几乎完全沿用旧预测（保证连续），
            # 随后平滑过渡到新预测（保证响应新观测）。
            w = (torch.arange(1, k + 1, device=chunk.device, dtype=chunk.dtype) / (k + 1)).unsqueeze(1)
            fused = (1.0 - w) * prev[:k] + w * chunk[:k]
            new_q = [fused[i] for i in range(k)] + [chunk[i] for i in range(k, chunk.shape[0])]
        else:
            new_q = [chunk[i] for i in range(chunk.shape[0])]
        _BLEND_STATE["queue"] = new_q
        _BLEND_STATE["used"] = 0
    act = _BLEND_STATE["queue"].pop(0)
    _BLEND_STATE["used"] += 1
    return act.unsqueeze(0) if act.ndim == 1 else act
# ---- END OVERLAP BLEND PATCH ----



def get_model(usr_args):
    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in usr_args.")

    device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
    cli_overrides = [f"--device={device}"]
    try:
        policy = ACTPolicy.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
    except TypeError as exc:
        if "cli_overrides" not in str(exc):
            raise
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
        policy = ACTPolicy.from_pretrained(checkpoint_path, config=config)
    policy.eval()

    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(usr_args.get("image_key_map") or {})
    image_keys = list(policy.config.image_features.keys())
    for image_key in image_keys:
        image_key_map.setdefault(
            image_key,
            image_key.removeprefix("observation.images."),
        )

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 8))
    policy.state_obs_path = usr_args.get("state_obs_path", "robot/qpos")
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.act_image_keys = image_keys
    policy.image_key_map = image_key_map

    if policy.act_step <= 0:
        raise ValueError(f"act_step must be positive, got {policy.act_step}.")

    _blend_init(policy)

    return policy


def eval(env, model, obs):
    final_obs = obs
    info = None
    truncated = False
    inference_times_s = []

    for _ in range(model.act_step):
        if _blend_enabled():
            runs_model_inference = _blend_will_infer()
        else:
            runs_model_inference = (
                model.config.temporal_ensemble_coeff is not None
                or len(model._action_queue) == 0
            )
        started_at = start_inference(model.act_device) if runs_model_inference else None
        state = final_obs
        for key in str(model.state_obs_path).split("/"):
            if key:
                state = state[key]
        state_tensor = state.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=torch.float32, device=model.act_device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        batch = {"observation.state": state_tensor}
        for image_key in model.act_image_keys:
            camera_name = model.image_key_map.get(
                image_key,
                image_key.removeprefix("observation.images."),
            )
            image = final_obs["sensor"][camera_name]["color"]
            image_tensor = image.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(image, torch.Tensor) else torch.as_tensor(image, dtype=torch.float32, device=model.act_device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
            if torch.max(image_tensor) > 1.5:
                image_tensor = image_tensor / 255.0
            batch[image_key] = image_tensor

        action = _blend_select(model, batch)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(f"Expected policy action shape [B, D], but got {tuple(action.shape)}.")

        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        policy_action_dim = int(action.shape[-1])
        if policy_action_dim != env_action_dim:
            message = f"Policy action has dim {policy_action_dim}, but env expects {env_action_dim}."
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        action_tensor = action.detach().to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        if runs_model_inference:
            finish_inference(started_at, inference_times_s, model.act_device)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if env.get_wrapper_attr("is_task_success")():
            break
        if isinstance(truncated, torch.Tensor):
            is_truncated = truncated.any().item()
        elif isinstance(truncated, np.ndarray):
            is_truncated = truncated.any()
        else:
            is_truncated = bool(truncated)
        if is_truncated:
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    model.reset()
    _blend_reset()
