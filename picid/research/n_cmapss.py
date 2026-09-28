"""Unit-isolated N-CMAPSS transforms used by the formal fit/predict protocol."""
import awkward as ak
import numpy as np

from picid.data.data_objects import NamedTransformInput
from picid.transforms.base.base_transform import RaggedTransform
from picid.transforms.base.multisource import NoFitPerSegmentMixin
from picid.transforms.base_transforms.subsample import WindowedAggregationTransform


class UnitWiseWindowedAggregationTransform(NoFitPerSegmentMixin, RaggedTransform):
    """Apply the established window aggregation independently to each engine.

    The multi-source datasource keeps each engine as one outer Awkward segment.
    Flattening those segments before a sliding window can mix adjacent engines;
    this adapter reuses the canonical dense operation per engine and returns a
    ragged engine axis for downstream temporal tabularization.
    """

    def __init__(
        self,
        window_size: int | str,
        step: int,
        agg: str | None = None,
        dim: int = 0,
        aggregation: str | None = None,
        **kwargs,
    ):
        super().__init__()
        self._dense = WindowedAggregationTransform(
            window_size=window_size,
            step=step,
            agg=agg,
            dim=dim,
            aggregation=aggregation,
            **kwargs,
        )

    def transform_data(self, data: NamedTransformInput, metadata: dict):
        values = {key: value for key, value in data.items()}
        ragged = [value for value in values.values() if isinstance(value, ak.Array)]
        if not ragged:
            raise TypeError("N-CMAPSS unit-wise aggregation requires grouped Awkward arrays")
        n_units = len(ragged[0])
        if any(len(value) != n_units for value in ragged):
            raise ValueError("Aligned input fields disagree on the number of engines")

        per_key: dict[str, list[np.ndarray]] = {key: [] for key in values}
        for unit_idx in range(n_units):
            unit_values = {}
            for key, value in values.items():
                if not isinstance(value, ak.Array):
                    raise TypeError(f"Expected unit-grouped Awkward input for {key!r}")
                unit_values[key] = ak.to_numpy(value[unit_idx])
            unit_data = NamedTransformInput(metadata=data.metadata, **unit_values)
            transformed = self._dense.transform_data(unit_data, metadata)
            for key in values:
                per_key[key].append(np.asarray(transformed[key]))

        result = {}
        for key, units in per_key.items():
            grouped = ak.Array(units)
            if units and units[0].ndim > 1:
                for axis in range(2, units[0].ndim + 1):
                    grouped = ak.to_regular(grouped, axis=axis)
            result[key] = grouped
        return NamedTransformInput(metadata=data.metadata, **result)


__all__ = ["UnitWiseWindowedAggregationTransform"]

class UnitIdentifierTransform(NoFitPerSegmentMixin, RaggedTransform):
    """Build an unambiguous, source plus engine ID per aggregated NC-P row."""

    def transform_data(self, data: NamedTransformInput, metadata: dict):
        unit = data["unit"]
        source = data["n_DS"]
        if not isinstance(unit, ak.Array) or not isinstance(source, ak.Array):
            raise TypeError("NC-P unit identifiers require grouped Awkward inputs")
        if len(unit) != len(source):
            raise ValueError("Datasource and engine identity groups are misaligned")
        unit_id = ak.concatenate(
            [ak.values_astype(source[..., np.newaxis], np.float32),
             ak.values_astype(unit[..., np.newaxis], np.float32)],
            axis=-1,
        )
        return NamedTransformInput(metadata=data.metadata, unit_id=unit_id)


__all__.append("UnitIdentifierTransform")
