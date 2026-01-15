# 🚗 Vahan-Insight: Indian EV Market Analytics Dashboard

An interactive data analytics platform analyzing India's electric vehicle revolution using the Kaggle Indian Vehicle Registration dataset.

## 📊 Live Dashboard

**Streamlit Dashboard:** Professional analytics interface with multi-select filters, geospatial maps, and FAME policy insights.

## ✨ Features

### 🔋 Advanced Analytics
- **3-Toggle Geospatial Map**: Absolute EV count, Adoption rate (%), and YoY growth
- **Multi-Select Filters**: Compare fuel types (EV vs Hybrid) and states
- **Public Mobility Analysis**: E-rickshaws, buses, and 3-wheelers vs personal EVs
- **FAME-II Policy Impact**: Data-backed insights on subsidy shift

### 📈 Key Insights
- **27x EV Growth** from 2020 to 2025
- **EV Market Share**: 0.6% → 7.8%
- **E-Rickshaw Saturation**: Growth plateaued at +3% after 2023
- **Bus Gap**: Major opportunity in public transport electrification

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Kaggle API credentials

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd Vahan_insights

# Install dependencies
pip install -r requirements.txt

# Set up Kaggle credentials
# Create .env file with KAGGLE_USERNAME and KAGGLE_KEY
# OR place kaggle.json in ~/.kaggle/

# Run data ingestion (one-time)
python ingest_data.py

# Run data cleaning
python data_cleaning.py

# Generate visualizations (optional)
python ev_analysis.py

# Launch dashboard
streamlit run dashboard_app.py
```

## 📁 Project Structure

```
Vahan_insights/
├── dashboard_app.py           # Streamlit dashboard
├── ingest_data.py             # Kaggle data download
├── data_cleaning.py           # Data processing pipeline
├── ev_analysis.py             # Static visualizations
├── requirements.txt           # Dependencies
├── config/
│   ├── rto_state_mapping.json
│   └── manufacturer_mapping.json
├── data/                      # CSV files (gitignored)
└── visualizations/            # Output charts
```

## 🛠️ Tech Stack

- **Backend**: Python, Pandas
- **Dashboard**: Streamlit
- **Visualization**: Plotly, Folium
- **Data Source**: Kaggle Indian Vehicle Registration Dataset

## 📋 Data Note

This dashboard analyzes a **statistical sample** of Indian vehicle registrations (500K records representing 2.8M vehicles). Absolute values are indicative. **Trends, growth rates, and comparative insights remain statistically valid**.

## 🎯 Roadmap

- [x] Phase 1: Data Engineering & Cleaning
- [x] Phase 2: EV Revolution Deep Dive
- [x] Phase 3: Interactive Dashboard
- [ ] Phase 4: Machine Learning (Forecasting, Clustering)
- [ ] Phase 5: Gen AI Intelligence Agent
- [ ] Phase 6: Deployment

## 📄 License

MIT License - see LICENSE file for details

## 👤 Author

Built with ❤️ for Indian EV market analytics
