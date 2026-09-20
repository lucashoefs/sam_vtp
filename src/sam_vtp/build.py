from importlib.resources import files
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam_vtp.checkpoint import overrides_from_state, read_checkpoint, strip_inert_keys

try:
    import sam2
except ImportError as e:
    raise ImportError(
        "sam2 is not installed. This project builds on the pinned submodule: "
        "pip install -e third_party/sam2"
    ) from e

UPSTREAM = "configs/sam2.1/sam2.1_hiera_t.yaml"


def build_config(state=None, train_config=None, delta="sam_vtp_hiera_t.yaml"):
    base = OmegaConf.load(Path(sam2.__file__).parent / UPSTREAM)
    tree = OmegaConf.create(
        {
            "model": {
                "_target_": "sam_vtp.modeling.sam_vtp.SAMVTP",
                "sam2_model": base.model,
            }
        }
    )

    assert_targets(tree)

    delta_path = files("sam_vtp") / "configs" / delta
    delta_cfg = OmegaConf.create(delta_path.read_text(encoding="utf-8"))

    cfg = OmegaConf.merge(tree, delta_cfg)  # struct OFF: delta adds keys

    assert_model_keys(cfg)
    assert_patches(cfg)

    OmegaConf.set_struct(cfg, True)  # lock schema before overrides
    if state is not None:
        for path, value in overrides_from_state(state, train_config):
            OmegaConf.update(cfg, path, value)
    return cfg


EXPECTED_TARGETS = {
    "model.sam2_model.image_encoder": "sam2.modeling.backbones.image_encoder.ImageEncoder",
    "model.sam2_model.image_encoder.neck": "sam2.modeling.backbones.image_encoder.FpnNeck",
    "model.sam2_model.memory_attention": "sam2.modeling.memory_attention.MemoryAttention",
    "model.sam2_model.memory_attention.layer": "sam2.modeling.memory_attention.MemoryAttentionLayer",
    "model.sam2_model.memory_attention.layer.self_attention": "sam2.modeling.sam.transformer.RoPEAttention",
    "model.sam2_model.memory_attention.layer.cross_attention": "sam2.modeling.sam.transformer.RoPEAttention",
    "model.sam2_model.memory_encoder": "sam2.modeling.memory_encoder.MemoryEncoder",
    "model.sam2_model.memory_encoder.mask_downsampler": "sam2.modeling.memory_encoder.MaskDownSampler",
    "model.sam2_model.memory_encoder.fuser": "sam2.modeling.memory_encoder.Fuser",
    "model.sam2_model.memory_encoder.fuser.layer": "sam2.modeling.memory_encoder.CXBlock",
}

EXPECTED_MODEL_KEYS = {
    "_target_",
    "sam2_model",
    "geometry_encoder",
    "refine_head",
    "camera",
    "geometry_hw",
    "memory_write_every",
    "input_normalization_mode",
}

EXPECTED_PATCHES = {
    "model": "sam_vtp.modeling.sam_vtp.SAMVTP",
    "model.sam2_model": "sam_vtp.modeling.sam_vtp_base.SAMVTPBase",
    "model.sam2_model.image_encoder.trunk": "sam_vtp.modeling.patches.CachedHiera",
    "model.sam2_model.image_encoder.neck.position_encoding": "sam_vtp.modeling.patches.CachedPositionEmbeddingSine",
    "model.sam2_model.memory_encoder.position_encoding": "sam_vtp.modeling.patches.CachedPositionEmbeddingSine",
    "model.sam2_model.memory_attention.layer.self_attention": "sam_vtp.modeling.patches.RoPEAttentionVTP",
    "model.sam2_model.memory_attention.layer.cross_attention": "sam_vtp.modeling.patches.RoPEAttentionVTP",
}


def assert_targets(cfg):
    for dotted, expected in EXPECTED_TARGETS.items():
        got = OmegaConf.select(cfg, f"{dotted}._target_")
        if got != expected:
            raise RuntimeError(
                f"upstream drift: {dotted}._target_ is {got!r}, expected "
                f"{expected!r}. Check the pinned sam2 commit."
            )


def assert_model_keys(cfg):
    extra = set(cfg.model.keys()) - EXPECTED_MODEL_KEYS
    if extra:
        raise RuntimeError(
            f"unexpected keys directly under model: {sorted(extra)}. "
            f"Anything inherited from SAM 2's config belongs under "
            f"model.sam2_model."
        )


def assert_patches(cfg):
    for dotted, expected in EXPECTED_PATCHES.items():
        got = OmegaConf.select(cfg, f"{dotted}._target_")
        if got != expected:
            raise RuntimeError(
                f"override did not land: {dotted}._target_ is {got!r}, "
                f"expected {expected!r}. Check the delta YAML nesting."
            )
    if cfg.model.sam2_model.memory_attention.layer.self_attention.embedding_dim != 256:
        raise RuntimeError(
            "memory_attention.layer.self_attention has no inherited "
            "embedding_dim -- the node was created fresh rather than merged."
        )


NORMALIZATION_MODES = ("0_255", "0_1", "-1_1", "auto")


def build_model(
    checkpoint,
    device="cpu",
    *,
    normalization_mode=None,
    memory_write_every=None,
    camera=None,
    multimask=None,
    components=None,
    **overrides,
):
    """Build SAM-VTP and load `checkpoint` into it.

    Args:
        normalization_mode: the range your images arrive in -- "0_255", "0_1",
            "-1_1", or "auto". Pin it explicitly in deployment; "auto" infers
            from each batch's range and a dark frame can fool it.
        memory_write_every: write memory only every k-th frame. The bank holds a
            fixed number of frames, so the time it spans is set by the frame
            rate; k>1 restores a training-like window on a fast camera. Every
            frame still gets a prediction.
        camera: intrinsics for the depth used by the geometry path. Defaults to
            the TartanGround camera the released checkpoints were trained on --
            which is almost certainly not yours. See CameraIntrinsics.
        multimask: normally read from the checkpoint's training config. It is
            invisible in the weights, so a checkpoint without a config and
            without this flag falls back to False.
        components: runtime switches, e.g. {"memory": False} for the frame-wise
            ablation. Keys are `memory`, `geometry`, `refine`. These select which
            of the checkpoint's weights are USED; they cannot add a component the
            checkpoint lacks.
    """
    if normalization_mode is not None and normalization_mode not in NORMALIZATION_MODES:
        raise ValueError(
            f"unknown normalization_mode {normalization_mode!r}; "
            f"expected one of {NORMALIZATION_MODES}")

    state, train_config = read_checkpoint(checkpoint)
    state = strip_inert_keys(state)
    cfg = build_config(state=state, train_config=train_config, **overrides)

    # Applied after overrides_from_state so an explicit argument wins over
    # anything inferred from the checkpoint's embedded training config.
    if normalization_mode is not None:
        OmegaConf.update(cfg, "model.input_normalization_mode",
                         str(normalization_mode))
    if memory_write_every is not None:
        OmegaConf.update(cfg, "model.memory_write_every", int(memory_write_every))
    if multimask is not None:
        OmegaConf.update(cfg, "model.sam2_model.multimask_output_in_sam",
                         bool(multimask))
        OmegaConf.update(cfg, "model.sam2_model.multimask_output_for_tracking",
                         bool(multimask))

    model = instantiate(cfg.model, _recursive_=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        hint = (
            "\nupstream's PromptEncoder keys present, the LearnedPrompt "
            "swap did not happen"
            if any("point_embeddings" in k for k in unexpected)
            else ""
        )
        raise RuntimeError(
            f"checkpoint does not match the assembled model: "
            f"{len(missing)} missing, {len(unexpected)} unexpected.\n"
            f"  missing:    {sorted(missing)[:8]}\n"
            f"  unexpected: {sorted(unexpected)[:8]}{hint}"
        )
    model.clear_inference_caches()
    if camera is not None:
        model.camera = camera
    model = model.to(device).eval()
    if components:
        model.set_components(**components)
    return model


def describe(model) -> str:
    """One-line summary, for logs that need to record what actually ran.

    Module-level form of `SAMVTP.describe()`, for callers that would rather pass
    the model than reach through it.
    """
    return model.describe()


def tune_for_inference(model=None, *, channels_last: bool = True,
                       cudnn_benchmark: bool = False) -> str:
    """Apply the deployment-only speedups that do not change the architecture.

    Explicit rather than an import side effect, and explicit rather than folded
    into build_model, because both of these move results slightly: a different
    convolution algorithm rounds differently. Do not turn them on for a run whose
    numbers have to reproduce an earlier one exactly -- in particular, not before
    tests/check_fullgraph.py.

    channels_last: NHWC for the whole model. Narrow convolutions are badly
        served by NCHW -- on an Orin Nano the refinement head's two 3x3
        convolutions took 9.9 ms in NCHW and 1.6 ms in NHWC -- and the memory
        encoder and the decoder's transposed convolutions pay the same tax.
        Measured 50.3 -> 47.4 ms per frame on the whole model.

    cudnn_benchmark: time the available convolution algorithms on first use and
        keep the fastest. OFF by default: it measured as exactly neutral on this
        model, and an unproven source of run-to-run variation is not worth
        taking for free.

    Returns a one-line description, worth logging next to the model banner.
    """
    import torch

    if not torch.cuda.is_available():
        return "tune_for_inference: no CUDA, nothing to set"
    notes = []
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        notes.append("cudnn.benchmark=True")
    if channels_last and model is not None:
        model.to(memory_format=torch.channels_last)
        notes.append("channels_last")
    elif channels_last:
        notes.append("channels_last SKIPPED (no model passed)")
    return f"tune_for_inference: {', '.join(notes) or 'nothing enabled'} (not bit-exact)"
