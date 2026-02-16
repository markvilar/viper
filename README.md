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


## Testing

The repository includes tests for the registry and embedder factories.[^10]

To run them:

```bash
uv run pytest
```


## Acknowledgements

This package reuses ideas, code, and checkpoints from several excellent VPR projects.
Please cite and credit the original works when using the corresponding models.

- [AnyLoc](https://github.com/AnyLoc/DINO)
- [CliqueMining](https://github.com/serizba/clique-mining)
- [CosPlace](https://github.com/gmberton/CosPlace)
- [EigenPlaces](https://github.com/gmberton/EigenPlaces)
- [MegaLoc](https://github.com/gmberton/MegaLoc)
- [MixVPR](https://github.com/amaralibey/MixVPR)
- [NetVLAD](https://github.com/cvg/Hierarchical-Localization)
- [SALAD](https://github.com/serizba/salad)
