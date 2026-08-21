# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-08-21

### Added

- Self-contained MegaLoc implementation that loads from a bundled checkpoint,
  replacing the `torch.hub` dependency. The model is built bottom-up from the
  state dict with dimensions derived from the checkpoint.
- Support for registering multiple variants per model.
- Grouped `models` CLI subgroup for the model-listing commands.
- SVD-truncated MegaLoc checkpoints, with a generation script and release notes.
- `mask_token` parameter for DINOv2 to enable strict checkpoint loading.

### Changed

- `MegaLocModel.forward` raises `ValueError` on invalid input instead of
  asserting.
- Interpolated positional grid size mismatches raise `RuntimeError` instead of
  asserting.

## [0.2.0] - 2026-03-27

- Initial tagged release.

[0.3.0]: https://github.com/markvilar/viper/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/markvilar/viper/releases/tag/v0.2.0
