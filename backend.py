import os
import math
import warnings
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['RDKIT_LOG_LEVEL'] = 'error'

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

# ====== MODEL CONSTANTS ======
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
MAX_LEN = 128
N_FP = 256
DESCRIPTOR_NAMES = [
    "MolWt","HeavyAtomMolWt","ExactMolWt","NumValenceElectrons",
    "FpDensityMorgan1","FpDensityMorgan2","FpDensityMorgan3",
    "MolLogP","MolMR","TPSA",
    "NumHAcceptors","NumHDonors","NumRotatableBonds","NumAromaticRings",
    "NumSaturatedRings","NumAliphaticRings","NumAromaticHeterocycles",
    "NumSaturatedHeterocycles","NumAliphaticHeterocycles",
    "RingCount","NumHeteroatoms","NumRadicalElectrons",
    "FractionCSP3","HeavyAtomCount","NHOHCount","NOCount",
    "MaxAbsEStateIndex","MinAbsEStateIndex","MaxEStateIndex","MinEStateIndex",
    "Chi0","Chi0n","Chi0v","Chi1","Chi1n","Chi1v","Chi2n","Chi2v",
    "Kappa1","Kappa2","Kappa3",
    "LabuteASA","PEOE_VSA1","PEOE_VSA2","PEOE_VSA3",
    "SMR_VSA1","SMR_VSA2","SlogP_VSA1","SlogP_VSA2","BalabanJ"
]
N_DESC = len(DESCRIPTOR_NAMES)
N_RDKIT = N_DESC + N_FP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Globals to be populated on startup
tok2id = {}
PAD_ID = CLS_ID = UNK_ID = 0
VOCAB_SIZE = 0
target_means = None
target_stds = None
scaler_X = None
feat_medians = None
model = None

# RDKit extractors
calc = MoleculeDescriptors.MolecularDescriptorCalculator(DESCRIPTOR_NAMES)
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_FP)

# ====== MODEL ARCHITECTURE ======
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransPolymerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, d_ff=1024, max_len=MAX_LEN, n_rdkit=N_RDKIT, n_targets=5, dropout=0.15):
        super().__init__()
        self.d_model = d_model
        # 1. Token Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len + 10, dropout)
        
        # 2. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
        
        # 3. RDKit Encoder
        self.rdkit_enc = nn.Sequential(
            nn.Linear(n_rdkit, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(256, d_model), nn.LayerNorm(d_model),
        )
        
        # 4. Fusion
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2), nn.LayerNorm(d_model*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )
        
        # 5. Heads
        def make_head():
            return nn.Sequential(
                nn.Linear(d_model, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout/2),
                nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
            )
        self.heads = nn.ModuleList([make_head() for _ in range(n_targets)])
        self.cross_prop = nn.Sequential(nn.Linear(n_targets, 32), nn.GELU(), nn.Linear(32, n_targets))

    def forward(self, tokens, rdkit_feats):
        pad_mask = (tokens == PAD_ID)
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        cls_out = x[:, 0, :]
        rdkit_lat = self.rdkit_enc(rdkit_feats)
        
        cls_q = cls_out.unsqueeze(1)
        rdkit_v = rdkit_lat.unsqueeze(1)
        cross, _ = self.cross_attn(cls_q, rdkit_v, rdkit_v)
        cross = self.cross_norm(cross.squeeze(1) + cls_out)
        
        concat = torch.cat([cross, rdkit_lat], dim=-1)
        gate = self.gate(concat)
        fused = gate * cross + (1 - gate) * rdkit_lat
        final = self.fusion_mlp(torch.cat([fused, cls_out], dim=-1))
        
        raw = torch.cat([head(final) for head in self.heads], dim=1)
        return raw + 0.05 * self.cross_prop(raw)

# ====== HELPER FUNCTIONS ======
def tokenize_smiles(smiles, max_len=MAX_LEN):
    tokens = [CLS_ID]
    i = 0
    s = smiles.strip()
    while i < len(s) and len(tokens) < max_len:
        matched = False
        for length in [6, 5, 4, 3, 2, 1]:
            chunk = s[i:i+length]
            if chunk in tok2id:
                tokens.append(tok2id[chunk])
                i += length
                matched = True
                break
        if not matched:
            tokens.append(UNK_ID)
            i += 1
    tokens = tokens[:max_len]
    pad_len = max_len - len(tokens)
    tokens = tokens + [PAD_ID] * pad_len
    return tokens

def smiles_to_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.full(N_RDKIT, np.nan, dtype=np.float32)
        desc = np.array(calc.CalcDescriptors(mol), dtype=np.float32)
        fp = np.array(morgan_gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
        return np.concatenate([desc, fp])
    except:
        return np.full(N_RDKIT, np.nan, dtype=np.float32)

def smiles_to_svg(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return ""
        # Create a transparent background SVG
        d2d = rdMolDraw2D.MolDraw2DSVG(400, 300)
        opts = d2d.drawOptions()
        opts.clearBackground = False
        opts.baseFontSize = 14
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        return d2d.GetDrawingText()
    except:
        return ""

# ====== FASTAPI SETUP ======
app = FastAPI(title="Polymer Prediction Backend")

@app.on_event("startup")
def load_model():
    global tok2id, PAD_ID, CLS_ID, UNK_ID, VOCAB_SIZE
    global target_means, target_stds, scaler_X, feat_medians, model
    
    checkpoint_path = "model2_transformer.pth"
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Model checkpoint not found at {checkpoint_path}")
        
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    tok2id = ckpt["tok2id"]
    PAD_ID, CLS_ID, UNK_ID = tok2id.get("<PAD>", 0), tok2id.get("<CLS>", 1), tok2id.get("<UNK>", 2)
    VOCAB_SIZE = len(tok2id)
    
    target_means = ckpt["target_means"]
    target_stds = ckpt["target_stds"]
    scaler_X = ckpt["scaler_X"]
    feat_medians = ckpt["feat_medians"]
    
    model = TransPolymerModel(vocab_size=VOCAB_SIZE)
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE)
    model.eval()

class PredictRequest(BaseModel):
    smiles: str

@app.post("/predict")
def predict(req: PredictRequest):
    smiles = req.smiles
    if not smiles:
        raise HTTPException(status_code=400, detail="Empty SMILES string")
        
    # Processing RDKit and tokenizing
    rdk = smiles_to_rdkit(smiles)
    for j in range(rdk.shape[0]):
        if np.isnan(rdk[j]):
            rdk[j] = feat_medians[j]
    
    try:
        rdk_scaled = scaler_X.transform(rdk.reshape(1, -1)).astype(np.float32)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scaler transform error: {e}")
        
    tokens = np.array([tokenize_smiles(smiles)], dtype=np.int32)
    
    with torch.no_grad():
        tok_t = torch.tensor(tokens, device=DEVICE)
        rdk_t = torch.tensor(rdk_scaled, device=DEVICE)
        preds_sc = model(tok_t, rdk_t).cpu().numpy()[0]
        
    preds_real = preds_sc * target_stds + target_means
    result = {target: float(preds_real[i]) for i, target in enumerate(TARGETS)}
    
    svg_str = smiles_to_svg(smiles)
    
    return {"predictions": result, "svg": svg_str}

# Mount static files folder to serve UI
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
