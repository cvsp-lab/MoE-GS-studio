# MoE-GS Studio

**MoE-GS Studio** is a research hub for **Mixture-of-Experts (MoE) architectures for Dynamic Gaussian Splatting**.  
This repository organizes our research exploring how **expert specialization and routing mechanisms** can improve dynamic 3D scene reconstruction.

Our work investigates how multiple dynamic Gaussian splatting models can be combined through **Mixture-of-Experts frameworks** to better handle diverse motion patterns and scene dynamics.

---

### Research Index

- (NEW) [**MoDE (Mixture of Deformation Experts)**](#mode-mixture-of-deformation-experts) is our latest research project in the **MoE-GS Studio** series, extending the Mixture-of-Experts paradigm for **dynamic Gaussian splatting**.
- [**MoE-GS (ICLR 2026)**](https://github.com/cvsp-lab/MoE-GS) introduces a Mixture-of-Experts framework with a pixel-wise routing mechanism that adaptively selects and combines multiple dynamic Gaussian splatting models.
[`Code`](https://github.com/cvsp-lab/MoE-GS) [`Paper`](https://arxiv.org/abs/2510.19210) [`ProjectPage`](https://paper.pnu-cvsp.com/MoE-GS/)
<br>


# MoDE: Mixture of Deformation Experts

Instead of relying on a single deformation model, **MoDE introduces multiple deformation experts**, each specializing in different motion behaviors.  
A routing mechanism dynamically selects or combines experts depending on the spatial and temporal characteristics of the scene.

This design enables the model to better handle:

- complex object motion  
- non-rigid deformation  
- heterogeneous dynamic regions

MoDE builds upon insights from our earlier work [**MoE-GS**](#moe-gs-mixture-of-experts-for-dynamic-gaussian-splatting-iclr-2026), which demonstrated the benefits of combining multiple dynamic Gaussian splatting models.

<br>

### 1. Virtual Environment Setup

```bash
pip install -r requirements.txt
pip install -e submodules/diff-gaussian-rasterization/
pip install -e submodules/simple-knn/ 
```

---

### 2. Data Preprocessing

```bash
# automatically extract the frames and reorganize them
python script/pre_n3v.py --videopath <dataset>/<scene>

# downsample dense point clouds
python script/downsample_point.py \
	<location>/<scene>/colmap/dense/workspace/fused.ply \
	<location>/<scene>/points3D_downsample.ply
```

---

### 3. MoDE - 4DGaussians w/ E-D3DGS (Train & Render)

```bash
# 4D Gaussian + E-D3DGS (Neural 3D Video Dataset)
python train_emb.py \
	-s <N3V_DATASET_ROOT>/coffee_martini \
	--expname <SAVE_PATH> \
	--configs "arguments_MoDE/dynerf/config_rot_0/coffee_martini.py"

python render_emb.py --skip_test --skip_train \
	--model_path <SAVE_PATH> \
	--configs "arguments_MoDE/emb/config_0/coffee_martini.py" \
	--iteration 30000
```

---

### 4. MoDE - 4DGaussians w/ Grid4D (Train & Render)

```bash
# 4D Gaussian + Grid4D (Neural 3D Video Dataset)
python train_hash.py \
	-s <N3V_DATASET_ROOT>/coffee_martini \
	--expname <SAVE_PATH> \
	--configs "arguments_MoDE/dynerf/config_rot_0/coffee_martini.py"

python render_hash.py --skip_train --skip_test \
	--model_path <SAVE_PATH> \
	--configs "arguments_MoDE/hash/config_0/coffee_martini.py" \
	--iteration 30000
```