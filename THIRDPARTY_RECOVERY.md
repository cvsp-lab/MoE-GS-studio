# Third-Party Dependencies Recovery Guide

This project depends on several external open-source components that are not included in this repository.
Follow the instructions below to restore them before building or running the code.

## thirdparty/

### embedding (E-D3DGS)
```bash
# Source: https://github.com/JeongminB/E-D3DGS
git clone https://github.com/JeongminB/E-D3DGS.git /tmp/E-D3DGS
cp -r /tmp/E-D3DGS/scene /tmp/E-D3DGS/utils /tmp/E-D3DGS/arguments thirdparty/embedding/
```

## submodules/

### depth-diff-gaussian-rasterization
```bash
git clone https://github.com/ingra14m/depth-diff-gaussian-rasterization.git submodules/depth-diff-gaussian-rasterization
# Restore glm:
cd submodules/depth-diff-gaussian-rasterization/third_party
git clone https://github.com/g-truc/glm.git
```

### simple-knn
```bash
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git submodules/simple-knn
```

## lpipsPyTorch/
```bash
# Source: https://github.com/richzhang/PerceptualSimilarity
pip install lpips
# Or copy the lpipsPyTorch module from the PerceptualSimilarity repository.
```

## Building CUDA Extensions

After restoring all dependencies:
```bash
pip install submodules/depth-diff-gaussian-rasterization/
pip install submodules/simple-knn/
```
