# 🚗 Vahan-Insight: Indian EV Market Analytics

> **Interactive dashboard analyzing India's electric vehicle revolution using data-driven insights**

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)

## 📊 What We've Built

A comprehensive EV analytics platform featuring:

### ✨ Key Features
- **�️ Advanced Geospatial Map** - 3 toggle views: Absolute count, Adoption rate, YoY growth
- **🔀 Multi-Select Filters** - Compare fuel types (EV vs Hybrid) and states dynamically
- **🚌 Public Mobility Analysis** - E-rickshaws, buses, 3-wheelers vs personal EVs
- **📈 FAME-II Policy Insights** - Data-backed analysis of India's EV subsidy shift
- **📊 Interactive Charts** - Plotly visualizations with real-time filtering

### � Key Insights Discovered
- **27x EV Growth** - From 2020 to 2025
- **Market Share Jump** - 0.6% → 7.8% 
- **E-Rickshaw Saturation** - Growth plateaued at +3% after 2023
- **Bus Opportunity** - Massive gap in public transport electrification

## 🚀 Live Demo

**[View Dashboard](#)** _(Deploy to get link)_

## 🛠️ Tech Stack

```
Python + Pandas + Streamlit + Plotly + Folium
```

### Data Pipeline
1. **Ingestion** - Kaggle API automated download
2. **Cleaning** - RTO-to-state mapping, manufacturer normalization
3. **Analysis** - Segment classification, fuel categorization
4. **Visualization** - Interactive dashboard with filters

## 📁 Project Structure

```
Vahan_insights/
├── dashboard_app.py           # Main Streamlit dashboard
├── ingest_data.py             # Data download automation
├── data_cleaning.py           # ETL pipeline
├── ev_analysis.py             # Static visualizations
├── requirements.txt           # Dependencies
├── config/                    # Mapping files
│   ├── rto_state_mapping.json
│   └── manufacturer_mapping.json
└── data/                      # CSV files (gitignored)
```

## 🌐 Deployment (Streamlit Cloud)

### Prerequisites
1. **Kaggle Account** - Get API credentials from [kaggle.com/settings](https://www.kaggle.com/settings)
2. **GitHub Account** - Repository already set up!

### Deploy Steps

1. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/vahan-insight.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your GitHub repo: `vahan-insight`
   - Main file: `dashboard_app.py`
   - Click "Advanced settings" → "Secrets"
   - Add secrets:
     ```toml
     KAGGLE_USERNAME = "your_kaggle_username"
     KAGGLE_KEY = "your_kaggle_key"
     ```
   - Click "Deploy"!

3. **First Run** - The app will automatically:
   - Download data from Kaggle
   - Clean and process datasets
   - Launch the dashboard

## 🔒 Security

✅ **No credentials in code** - All API keys stored in Streamlit secrets  
✅ **Data files excluded** - `.gitignore` prevents CSV commits  
✅ **Public-ready** - Safe for public GitHub repository

## 📋 Data Note

This dashboard analyzes a **statistical sample** of 500K registration records (representing 2.8M vehicles). Absolute values are indicative. **Trends, growth rates, and comparative insights remain statistically valid** for analysis.

## 🎯 Phases Completed

- [x] **Phase 1** - Data Engineering & Cleaning
- [x] **Phase 2** - EV Revolution Deep Dive Analysis  
- [x] **Phase 3** - Interactive Dashboard with Advanced Filters
- [ ] **Phase 4** - Machine Learning (Forecasting, Clustering)
- [ ] **Phase 5** - Gen AI Intelligence Agent

## 📄 GitHub Description

```
Interactive EV analytics dashboard for India | 27x growth analysis | 
Streamlit + Plotly + Folium | FAME policy insights
```

## 👤 Author

Built for Indian EV market intelligence  
**Tech:** Python • Streamlit • Plotly • Folium

---

**⭐ Star this repo if you found it useful!**
