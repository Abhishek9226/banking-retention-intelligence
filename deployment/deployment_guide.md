# 🚀 Deployment Guide — BankRetain IQ

## Option 1: Local Development

```bash
# 1. Clone and set up environment
git clone https://github.com/yourname/bankretain-iq.git
cd bankretain-iq
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your dataset
cp your_data.csv data/raw/churn_data.csv

# 4. Run full pipeline (generates features, trains models, saves artefacts)
python run_pipeline.py

# 5. Launch dashboard
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

---

## Option 2: Docker Container

```bash
# Build image
docker build -t bankretain-iq -f deployment/Dockerfile .

# Run container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  bankretain-iq

# Open: http://localhost:8501
```

---

## Option 3: Streamlit Cloud (Free)

1. Push project to GitHub (ensure `data/raw/churn_data.csv` is included or generate it in `run_pipeline.py`)
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Connect GitHub repo
5. Set **Main file path:** `dashboard/app.py`
6. Click **Deploy**

**Note:** Add `requirements.txt` to the repo root — Streamlit Cloud auto-installs it.

---

## Option 4: AWS EC2

```bash
# SSH into EC2 instance (Ubuntu 22.04+)
sudo apt update && sudo apt install python3-pip -y
pip3 install -r requirements.txt

# Run pipeline
python3 run_pipeline.py

# Launch with nohup (persist after SSH disconnect)
nohup streamlit run dashboard/app.py \
  --server.port 8501 \
  --server.address 0.0.0.0 &

# Open: http://<ec2-public-ip>:8501
# (Ensure Security Group allows TCP 8501)
```

---

## Environment Variables (Optional)

Create `.env` file for secrets:
```
STREAMLIT_SERVER_PORT=8501
MODEL_ARTIFACTS_PATH=models/artifacts
DATA_PATH=data/raw/churn_data.csv
```

---

## Directory Requirements

Before launching the dashboard, ensure:
- `data/raw/churn_data.csv` — raw dataset
- `data/processed/churn_features.parquet` — feature-engineered data (created by pipeline)
- `data/processed/churn_segmented.parquet` — segmented data (created by pipeline)

Run `python run_pipeline.py` once to generate all required files.
