# MoE-GS Studio

**MoE-GS Studio** is a research hub for **Mixture-of-Experts (MoE) architectures for Dynamic Gaussian Splatting**.  
This repository organizes our research exploring how **expert specialization and routing mechanisms** can improve dynamic 3D scene reconstruction.

Our work investigates how multiple dynamic Gaussian splatting models can be combined through **Mixture-of-Experts frameworks** to better handle diverse motion patterns and scene dynamics.

---

# MoDE: Mixture of Deformation Experts (Latest Work)

**MoDE (Mixture of Deformation Experts)** is our latest research project in the **MoE-GS Studio** series, extending the Mixture-of-Experts paradigm for **dynamic Gaussian splatting**.

Instead of relying on a single deformation model, **MoDE introduces multiple deformation experts**, each specializing in different motion behaviors.  
A routing mechanism dynamically selects or combines experts depending on the spatial and temporal characteristics of the scene.

This design enables the model to better handle:

- complex object motion  
- non-rigid deformation  
- heterogeneous dynamic regions

MoDE builds upon insights from our earlier work **MoE-GS**, which demonstrated the benefits of combining multiple dynamic Gaussian splatting models.

🚧 **Code Release**  
The official implementation of **MoDE** will be released in this repository **after the paper is accepted**.

---

# Earlier Work in the Series

## MoE-GS: Mixture of Experts for Dynamic Gaussian Splatting

**MoE-GS** is the first work in our MoE-based dynamic Gaussian splatting research line.

It introduces a **Mixture-of-Experts framework** that adaptively combines multiple dynamic Gaussian splatting models.

Different models exhibit strengths under different conditions—for example:

- some handle **fast motion**
- others reconstruct **fine geometric details**

MoE-GS learns a **pixel-wise routing mechanism** that dynamically selects the most suitable expert during rendering.

Paper: **ICLR 2026**  
Project repository:  
https://github.com/cvsp-lab/MoE-GS
