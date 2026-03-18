# AI PlantDocBot

Plant disease classification project using the PlantVillage dataset. Includes data-cleaning utilities, train/val split tooling, and Colab notebooks to train and compare CNN backbones (MobileNet and ResNet). Also includes a small symptom-text dataset generator for NLP experiments.

**Highlights**
- Clean and normalize PlantVillage class labels into a consistent schema
- Create stratified validation splits from a train folder
- Train and compare MobileNet and ResNet models in PyTorch
- Generate augmented symptom-text data from aligned symptom descriptions
- Store trained weights under `models/`

**Project Structure**
- `notebooks/` Colab notebooks for MobileNet, ResNet, and a comparison run
- `scripts/` Dataset utilities and symptom-text generator
- `data/` Dataset storage (ignored by git)
- `models/` Saved PyTorch checkpoints (`.pth`)
- `material/` Project report PDF
- `PlantVillage/` Raw dataset folder (ignored by git)

**Data Setup**
1. Place your PlantVillage images into `data/train` and `data/test` (optionally `data/val`).
2. Optional label normalization (creates `data_cleaned/`):
   ```bash
   python scripts/clean_dataset.py
   ```
3. Optional stratified validation split:
   ```bash
   python scripts/make_val_split.py --ratio 0.2 --seed 42
   ```

**Symptom Text Dataset (Optional)**
The generator reads `scripts/symptoms_aligned.csv` and writes `plant_disease_5000_dataset.csv`.
```bash
cd scripts
python generate_dataset.py
```

**Notebooks**
The notebooks in `notebooks/` are written for Google Colab and mount Google Drive. If you run locally, update dataset paths to match your local `data/` directory.

**Models**
Pretrained checkpoints are stored in `models/`:
- `plant_disease_mobilenet_v1.pth`
- `plant_disease_resnet18_v1.pth`
- `plant_disease_resnet50_v1.pth`

**Notes**
- Large datasets are excluded via `.gitignore`. Add your data locally in `data/`.
- If you want a pure local workflow, remove the Colab-specific Drive mount cells from the notebooks.
