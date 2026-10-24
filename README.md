# Polymer AI: Multi-Task Transformer for Polymer Property Prediction

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![RDKit](https://img.shields.io/badge/Chemistry-RDKit-green.svg)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ts3363/polymer-ai)

**Polymer AI** is a full-stack, deep learning platform engineered to predict macroscopic physicochemical properties of polymers directly from their microscopic repeating-unit SMILES representations. Combining an attention-based Transformer with 2D/3D topological chemoinformatics (RDKit) and a cyberpunk scientific web dashboard, Polymer AI accelerates polymer discovery and materials informatics.

---

## ⚡ 1-Click Live Deployment

Deploy your own live instance of Polymer AI in under 2 minutes:

| Platform | Deployment Type | Resources & Tier | One-Click Link |
| :--- | :--- | :--- | :--- |
| **Render** | Web Service (Automated) | Free Tier (512MB RAM) | [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ts3363/polymer-ai) |
| **Hugging Face Spaces** | Docker / Space | Free Tier (16 GB RAM, 2 vCPU) | [Create HF Space](https://huggingface.co/new-space) (Import `ts3363/polymer-ai`) |
| **Railway / Koyeb** | Docker / Container | Free / Standard | Connect GitHub repository `polymer-ai` |

---

## 🌟 Key Features

- **TransPolymer Multi-Modal Architecture:** Merges a 6-layer Multi-Head Self-Attention Transformer for 1D SMILES token sequences with a deep feed-forward encoder for 306 RDKit structural descriptors.
- **Cross-Attention Gated Fusion:** Dynamically weights sequence grammar and topological molecular descriptors through cross-attention and gating mechanisms.
- **5-in-1 Multi-Task Property Prediction:**
  - Glass Transition Temperature ($T_g$)
  - Fractional Free Volume (FFV)
  - Thermal Conductivity ($T_c$)
  - Polymer Density
  - Radius of Gyration ($R_g$)
- **Interactive Scientific Web Dashboard:**
  - **2D Topology Visualizer:** Real-time chemical structure vector rendering (SVG) powered by RDKit's Substructure Drawing Engine.
  - **Properties Radar Chart:** Interactive normalized multi-axis radar chart (via Chart.js) for evaluating property trade-offs.
  - **Scientific Rigor:** Mean Absolute Error (MAE) confidence intervals (e.g. `± 8.5 °C`), relative property gauges, and dim-decimal typography.
  - **Researcher Utilities:** One-click value clipboard copying, live input validation, and CSV dataset export.

---

## 🎯 Target Properties & Units

| Property | Symbol | Unit | Typical Range | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Glass Transition Temperature** | $T_g$ | °C | -100 to 400 | Temperature of transition from glassy to rubbery state |
| **Fractional Free Volume** | FFV | — | 0.10 to 0.50 | Unoccupied volume ratio, critical for gas membrane permeability |
| **Thermal Conductivity** | $T_c$ | W/(m·K) | 0.10 to 0.40 | Heat transfer rate through the bulk polymer matrix |
| **Polymer Density** | $\rho$ | g/cm³ | 0.80 to 2.00 | Bulk volumetric mass density of the amorphous chain network |
| **Radius of Gyration** | $R_g$ | Å | 5.0 to 40.0 | Root-mean-square distance of chain segments from mass center |

---

## 🏗️ Model Architecture (`TransPolymer`)

```
   Polymer SMILES String (e.g. *CC(*)c1ccccc1)
       │
       ├───► [Sub-structure Tokenizer] ───► Positional Encoding
       │                                            │
       │                                            ▼
       │                              Transformer Encoder (6 Layers, 8 Heads)
       │                                            │ (CLS Token)
       │                                            ▼
       │                                   [Cross-Attention] ◄───┐
       │                                            │            │
       └───► [RDKit Engine]                         ▼            │
             • 50 Molecular Descriptors    [Gated Fusion Layer]  │
             • 256-bit Morgan Fingerprints          │            │
                         │                          ▼            │
                         └──────────────► [RDKit MLP Encoder] ───┘
                                                    │
                                                    ▼
                                         [Multi-Task Output Heads]
                                                    │
                                         [Cross-Property Residual]
                                                    │
                                 ┌───────┬───────┼───────┬───────┐
                                 ▼       ▼       ▼       ▼       ▼
                                $T_g$   FFV     $T_c$  Density  $R_g$
```

### Technical Highlights
1. **Sequence Modeling:** Tokenizes SMILES sequences into chemical sub-units (up to length 128) and embeds them into a 256-dimensional space with sinusoidal positional encodings.
2. **Topological Featurization:** Extracts 50 physicochemical properties (e.g., MolWt, TPSA, Rotatable Bonds, Chi indices) combined with 256-bit Morgan Circular Fingerprints (radius=2) for a 306-dimensional feature vector.
3. **Multi-Head Cross-Attention:** Uses the SMILES CLS token as a query against RDKit descriptor latent vectors to ground sequence representations in topological chemistry.
4. **Multi-Task Heads with Cross-Property Residual:** Separate MLP prediction heads for each target property with an inter-property coupling matrix to model physical correlations (e.g., density vs. free volume).

---

## 📁 Repository Structure

```
polymer-ai/
├── backend.py                 # FastAPI backend & PyTorch inference pipeline
├── model2_transformer.pth     # Pretrained TransPolymer weights & scalers
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Git ignore rules for checkpoints/caches
├── static/
│   ├── index.html             # Analysis dashboard layout
│   ├── style.css              # Cyber-minimalist dark UI styles
│   └── script.js              # Frontend logic, charts, and API client
└── README.md                  # Project documentation
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.9+ (or conda environment)
- Git

### 2. Clone Repository
```bash
git clone https://github.com/ts3363/polymer-ai.git
cd polymer-ai
```

### 3. Setup Virtual Environment
```bash
# Using venv
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note:** For Conda users, RDKit can also be installed via `conda install -c conda-forge rdkit`.

### 5. Launch the Server
```bash
python backend.py
```
Or with Uvicorn directly:
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

The interactive dashboard will now be live at:
👉 **`http://localhost:8000`**

---

## 🔌 API Reference

### Predict Polymer Properties
`POST /predict`

**Request Body:**
```json
{
  "smiles": "*Oc1ccc(C(C)(C)c2ccc(Oc3ccc(C(=O)c4cccc(C(=O)c5ccc(*)cc5)c4)cc3)cc2)cc1"
}
```

**Response (`200 OK`):**
```json
{
  "predictions": {
    "Tg": 177.497,
    "FFV": 0.312,
    "Tc": 0.184,
    "Density": 1.156,
    "Rg": 14.821
  },
  "svg": "<svg xmlns='http://www.w3.org/2000/svg' ... </svg>"
}
```

---

## 🧪 Example SMILES for Testing

| Polymer Class | Example SMILES String |
| :--- | :--- |
| **Fluorinated Polyimide** | `*Oc1ccc(C=NN=Cc2ccc(Oc3ccc(C(c4ccc(*)cc4)(C(F)(F)F)C(F)(F)F)cc3)cc2)cc1` |
| **Polyetherketone (PEK)** | `*Oc1ccc(C(C)(C)c2ccc(Oc3ccc(C(=O)c4cccc(C(=O)c5ccc(*)cc5)c4)cc3)cc2)cc1` |
| **Polyimide Copolymer** | `*c1cccc(OCCCCCCCCOc2cccc(N3C(=O)c4ccc(-c5cccc6c5C(=O)N(*)C6=O)cc4C3=O)c2)c1` |
| **Polystyrene-derivative** | `*CC(*)c1ccccc1` |

---

## 📊 Scientific Insights & Findings

1. **Fractional Free Volume (FFV) Packing Ceiling:**
   Empirical testing revealed an $R^2$ ceiling around 0.848 for FFV when combining Morgan fingerprints with 2D RDKit descriptors. This finding aligns with polymer physics: FFV is fundamentally governed by 3D chain conformational packing and entanglements in the solid state, which 2D topological graph invariants only partially capture.
2. **Data Augmentation Efficacy:**
   By applying randomized SMILES canonicalization and string permutation during training, specialist models showed substantial validation $R^2$ jumps (e.g., $T_g$ specialist moving from 0.65 to >0.90), confirming that Transformer architectures benefit immensely from sequence diversity to capture backbone invariance.

---

## 🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
