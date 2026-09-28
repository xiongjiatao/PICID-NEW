import awkward as ak
import numpy as np
import torch

from picid.data.datasets.rul_context_dataset import RULContextBatchDataset


def test_row_aligned_composite_ids_follow_ragged_engine_windows():
    features = ak.to_regular(ak.Array([
        np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32),
    ]), axis=2)
    rul = ak.to_regular(ak.Array([
        np.asarray([[3.0], [2.0], [1.0]], dtype=np.float32),
        np.asarray([[3.0], [2.0], [1.0]], dtype=np.float32),
    ]), axis=2)
    unit_id = ak.to_regular(ak.Array([
        np.asarray([[1, 7], [1, 7], [1, 7]], dtype=np.int64),
        np.asarray([[4, 7], [4, 7], [4, 7]], dtype=np.int64),
    ]), axis=2)
    dataset = RULContextBatchDataset(
        data_dict={"features": features, "rul": rul, "unit_id": unit_id},
        task_type="rul",
        seq_len=2,
        label_len=0,
        pred_len=0,
        stride=1,
        get_unit_id=True,
        padding_left_flag=False,
        meta_data_dict={"current_data_split": "train"},
    )

    batch = dataset[[0, 3]]

    assert torch.equal(batch["unit_id"], torch.tensor([[1, 7], [4, 7]]))
    assert batch["features"].shape == (2, 2, 1)
