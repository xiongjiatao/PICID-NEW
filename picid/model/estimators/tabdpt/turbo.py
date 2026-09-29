"""TabDPT-Turbo v1.2.0 adapter for its separately pinned environment."""

from importlib.metadata import version

from .wrapper import FitPredictTabDPTWrapper


class FitPredictTabDPT120Wrapper(FitPredictTabDPTWrapper):
    """Adapt the v1.2 inference API without changing its native context policy."""

    def __init__(self, *args, **kwargs):
        if version("tabdpt") != "1.2.0":
            raise RuntimeError("TabDPT-Turbo adapter requires tabdpt==1.2.0")
        super().__init__(*args, **kwargs)

    def _constructor_options(self, device, model_weight_path, compile):
        return dict(device=device, model_weight_path=model_weight_path,
                    compile=compile)

    def _prediction_options(self):
        return {**super()._prediction_options(), "batch_size": self.inf_batch_size}
