# Viper: A Common Image Embedder Interface for Visual Place Recognition

![ci](https://github.com/markvilar/viper/actions/workflows/ubuntu.yml/badge.svg)

This Python package provides a **unified** image embedder interface for visual place recognition (VPR), along with wrapper implementations of several state-of-the-art VPR models so they all expose the same API.
It also includes a lightweight registry mechanism that lets you register custom embedders and retrieve them by string key.

## Features

- Common `ImageEmbedder` protocol for VPR models (name, vector size, device, call semantics).
- Eight wrapper models adapting popular VPR architectures to this interface:
    - AnyLoc
    - CliqueMining
    - CosPlace
    - EigenPlaces
    - MegaLoc
    - MixVPR
    - NetVLAD
    - SALAD
- Simple registry (`register_embedder_factory` / `get_embedder_factory`) for instantiating embedders by key.

## Installation

The package is configured as a standard Python project via `pyproject.toml` and adds support for the `uv` package manager.

You can install it in editable mode for development:

```bash
uv sync
```

**Disclaimer:** AnyLoc, CliqueMining, MegaLoc, and SALAD require CUDA to run.

## Usage

### Loading a built-in embedder

The recommended way to construct models is through the embedder registry.

```python
import viper

factory: viper.ImageEmbedderFactory = viper.get_embedder_factory("salad")    # or "mixvpr", "netvlad", "eigenplaces", ...
embedder: viper.ImageEmbedder = factory()

print(embedder.name)                   # "salad"
print(embedder.vector_size)            # 8448
print(embedder.embedder_parameters)    # dict of parameters
print(embedder.device)                 # "cpu" or "cuda"
```

All embedders implement the `ImageEmbedder` protocol, which exposes:

- `name: str`
- `vector_size: int` (embedding dimension)
- `embedder_parameters: dict[str, Any]` (model-specific metadata such as backbone, descriptor size, etc.)
- `device: str` (e.g. `"cpu"` or `"cuda:0"`)
- `__call__(images: torch.Tensor) -> torch.Tensor` (batched embedding, `B x C x H x W -> B x E`)


### Embedding a batch of images

All wrappers expect a batch of images as a float tensor of shape `B x C x H x W` with values in `[0.0, 1.0]` (for NetVLAD this is explicitly asserted).

```python
import torch
import viper

factory: viper.ImageEmbedderFactory = viper.get_embedder_factory("netvlad")
embedder: viper.ImageEmbedder = factory()
images: torch.Tensor = torch.rand(8, 3, 480, 640)  # example batch, normalized to [0, 1]
embeddings: torch.Tensor = embedder(images)        # shape: (8, embedder.vectorsize)
```

Wrappers handle grayscale input by converting 1-channel batches to 3-channel RGB internally.
Some models also resize images so that height/width are multiples of 14, matching DINOv2 backbone constraints.

### Registering a custom model

You can register your own model as long as it fulfills the `ImageEmbedder` interface.

```python
import torch
from viper.registry import register_embedder_factory
from viper.types import ImageEmbedder

class MyEmbedder(torch.nn.Module):
    @property
    def name(self) -> str:
        return "myembedder"

    @property
    def vector_size(self) -> int:
        return 1024

    @property
    def embedder_parameters(self) -> dict:
        return {"backbone": "resnet50"}

    @property
    def device(self) -> str:
        return next(self.parameters()).device.type

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # return embeddings of shape B x 1024
        ...

# Factory that returns an ImageEmbedder instance
@register_embedder_factory(key="myembedder")
def load_myembedder() -> ImageEmbedder:
    model = MyEmbedder()
    return model

```

You can then retrieve a factory via `get_embedder_factory("myembedder")` like the built-in models.

## Included VPR models

Below is a brief description of each wrapped model and how it is loaded.


| Key | Wrapper class | Backbone / descriptor info | Loading method |
| :-- | :-- | :-- | :-- |
| `salad` | `SALADWrapper` | DINOv2 backbone, custom SALAD aggregator.[^2] | `torch.hub.load("serizba/salad", "dinov2_salad")`.[^2] |
| `mixvpr` | `MixVPRWrapper` | ResNet backbone, MixVPR aggregator, 512-D descriptors.[^3] | Custom implementation, weights from URL in `MIXVPRURL`.[^3] |
| `netvlad` | `NetVLAD` | VGG16 backbone, NetVLAD aggregation, 4k or 32k dims.[^4] | Weights loaded from official MATLAB `.mat` files via `wget` and `scipy.io`.[^4] |
| `eigenplaces` | `EigenPlacesWrapper` | ResNet50 backbone, 2048-D descriptors.[^5] | `torch.hub.load("gmberton/eigenplaces", "get_trained_model", ...)`.[^5] |
| `megaloc` | `MegaLocWrapper` | DINOv2 backbone, feature dim from `impl.feat_dim`.[^6] | `torch.hub.load("gmberton/MegaLoc", "get_trained_model", ...)` (CUDA-only).[^6] |
| `cosplace` | `CosPlaceWrapper` | ResNet50 backbone, 2048-D descriptors.[^7] | `torch.hub.load("gmberton/cosplace", "get_trained_model", ...)`.[^7] |
| `cliquemining` | `CliqueMiningWrapper` | DINOv2 backbone, SALAD-style aggregator.[^8] | SALAD from `torch.hub`, CliqueMining weights from `CLIQUEMINING_CHECKPOINT_URL` (CUDA-only).[^2][^8] |
| `anyloc` | `AnyLocWrapper` | DINOv2 backbone, VLAD aggregation.[^9] | `torch.hub.load("AnyLocDINO", "get_vlad_model", ...)` (CUDA-only).[^9] |

## Testing

The repository includes tests for the registry and embedder factories.[^10]

To run them:

```bash
uv run pytest
```


## Acknowledgements

This package reuses ideas, code, and checkpoints from several excellent VPR projects.
Please cite and credit the original works when using the corresponding models.

- SALAD: original repository at `https://github.com/serizba/salad`.
- MixVPR: paper “Feature Mixing for Visual Place Recognition”, and reference implementation at `https://github.com/amaralibey/MixVPR` (parts of which this code is based on).
- NetVLAD: paper “NetVLAD: CNN architecture for weakly supervised place recognition”, and code from `https://github.com/cvg/Hierarchical-Localization`.
- EigenPlaces: loaded from the `gmberton/eigenplaces` Torch Hub repository.
- MegaLoc: loaded from the `gmberton/MegaLoc` Torch Hub repository.
- CosPlace: loaded from the `gmberton/cosplace` Torch Hub repository.
- CliqueMining: original repository at `https://github.com/serizba/clique-mining/tree/main` and checkpoint hosted under your `vpr-model-zoo` release URL.
- AnyLoc: loaded from the `AnyLocDINO` Torch Hub entry, which wraps the AnyLoc model.
