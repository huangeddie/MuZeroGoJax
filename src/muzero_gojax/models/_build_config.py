import chex


@chex.dataclass(frozen=True)
class ModelBuildConfig:
    """Build config for whole Go model."""
    board_size: int = -1
    hdim: int = -1
    embed_dim: int = -1
    dtype: str = None


@chex.dataclass(frozen=True)
class SubModelBuildConfig:
    """Build config for submodel."""
    name_key: str = None
    nlayers: int = -1
    model_build_config: ModelBuildConfig = ModelBuildConfig()
