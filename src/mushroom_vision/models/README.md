# Mushroom detector checkpoint

- File: `best.pt` (22,539,498 bytes).
- Use: Ultralytics YOLO detection for the global and close-range simulated RGB-D views.
- Distribution: the maintainer confirmed redistribution permission on 2026-10-17; the checkpoint is included under AGPL-3.0-only. See the repository-root `LICENSE` and `THIRD_PARTY_NOTICES.md`.
- Installation: `mushroom_vision/setup.py` installs the checkpoint into `share/mushroom_vision/models/best.pt`; the top-level launch uses that path by default.
- An alternative compatible, authorized checkpoint may be supplied with the launch argument `model_path:=/absolute/path/to/best.pt`.

The training dataset, training script, split, exact training recipe and independent detection metrics are not included. The recorded project result is a fixed-scene simulated single-mushroom pick; it is not a general real-world detector accuracy evaluation.
