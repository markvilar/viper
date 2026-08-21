# Release notes — `megaloc-svd-truncated-v1.0`

SVD-truncated MegaLoc checkpoints for the self-contained `viper` implementation.
Each asset is a full MegaLoc `state_dict` whose final linear projection has been
truncated to a lower descriptor dimension via truncated SVD, keeping the DINOv2
backbone and SALAD aggregation intact. The files load unwrapped through
`viper.models.megaloc.build_megaloc_from_state_dict`.

## Assets

| Descriptor dim | Asset | sha256 |
|---|---|---|
| 256  | `megaloc-256d-svd-truncated-v1.0.pth`  | `d19c4feea4b810f5a10f2073c26afa9581453aa2f3c7e96bf41d684d32313730` |
| 512  | `megaloc-512d-svd-truncated-v1.0.pth`  | `b1f6cbbe0b6ea2d4226b77cdeb6f26655f49c4cfe7b16cc303c25160ebc7752a` |
| 1024 | `megaloc-1024d-svd-truncated-v1.0.pth` | `1e48423505cfad6c7a5244259df5eacd311e7b762d389ffad4fbf38f331818ca` |

## Provenance

Each checkpoint is derived from the pretrained 8448-D MegaLoc checkpoint by
truncated SVD of the final linear projection (`viper.model_decomposition`),
driven by the `viper forge adapt` CLI.

| | |
|---|---|
| Source | [`megaloc-8448d-v1` / `megaloc-8448d-v1.pth`](https://github.com/markvilar/viper/releases/tag/megaloc-8448d-v1) |
| Source sha256 | `5b0c5b817356bda618ccc6a4049dd49b52249183df76046ab1c8e7e03c5c2791` |
| Method | `svd` (truncated SVD of the final linear projection) |
| Revision | `1.0` (tracks the release tag and the `-v1.0` filename suffix) |
| Target dimensions | 256 / 512 / 1024 |

### Invocations

Produced in one pass on a CUDA host (the `megaloc` factory requires CUDA to load
the source model). Reproduce with:

```sh
scripts/publish_megaloc_svd_checkpoints.sh
```

which runs, per dimension:

```sh
viper forge adapt megaloc --method svd --dim 256  --revision 1.0
viper forge adapt megaloc --method svd --dim 512  --revision 1.0
viper forge adapt megaloc --method svd --dim 1024 --revision 1.0
```

## Verification

Each asset is a pure `state_dict` and round-trips through
`build_megaloc_from_state_dict` with `strict=True`, emitting unit-norm
descriptors at its target dimension.
