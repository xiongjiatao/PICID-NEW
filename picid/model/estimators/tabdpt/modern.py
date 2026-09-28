"""TabDPT 1.3.0 adapter; run only in its separately locked environment."""
from importlib.metadata import version

from .wrapper import FitPredictTabDPTWrapper


class FitPredictTabDPT130Wrapper(FitPredictTabDPTWrapper):
    def __init__(self, *args, context_size=2048, context_reduction="subsample", **kwargs):
        if version("tabdpt") != "1.3.0":
            raise RuntimeError("Modern baseline requires tabdpt==1.3.0")
        self.context_reduction = context_reduction
        super().__init__(*args, context_size=context_size, **kwargs)

    def _constructor_options(self, device, model_weight_path, compile):
        return dict(device=device, model_weight_path=model_weight_path,
                    compile=compile, context_reduction=self.context_reduction)

    def _prediction_options(self):
        return {**super()._prediction_options(), "batch_size": self.inf_batch_size}

    def predict_distribution(self, X):
        if self.task_type in self.classification_tasks:
            raise ValueError("Full distributions require a regression task")
        return self.backbone.predict(X.detach().cpu().numpy(), output_type="full",
                                     **self._prediction_options())
