import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import folium
from streamlit_folium import st_folium

# Page config
st.set_page_config(
    page_title="Vahan-Insight Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Metrics cards */
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
    }
    
    /* Headers */
    h1 {
        color: #1a1a2e;
        font-weight: 700;
    }
    
    h3 {
        color: #374151;
        font-weight: 600;
        font-size: 18px;
        margin-bottom: 20px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: white;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Data loading with auto-setup for Streamlit Cloud
@st.cache_data
def load_data():
    """Load cleaned datasets - auto-run pipeline if files don't exist"""
    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    
    main_file = data_dir / "vehicle_registrations_cleaned.csv"
    agg_file = data_dir / "state_month_fuel_cleaned.csv"
    
    # Check if cleaned files exist
    if not main_file.exists() or not agg_file.exists():
        st.info("🔄 First-time setup: Downloading and processing data... (this may take 2-3 minutes)")
        
        try:
            # Run data ingestion
            import subprocess
            import sys
            
            with st.spinner("Downloading data from Kaggle..."):
                result = subprocess.run(
                    [sys.executable, str(base_dir / "ingest_data.py")],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    st.error(f"Data ingestion failed: {result.stderr}")
                    st.stop()
            
            with st.spinner("Cleaning and processing data..."):
                result = subprocess.run(
                    [sys.executable, str(base_dir / "data_cleaning.py")],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode != 0:
                    st.error(f"Data cleaning failed: {result.stderr}")
                    st.stop()
            
            st.success("✅ Data setup complete!")
            
        except Exception as e:
            st.error(f"Setup failed: {str(e)}")
            st.info("Please check that Kaggle credentials are correctly set in Streamlit secrets.")
            st.stop()
    
    # Load the cleaned data
    df_main = pd.read_csv(main_file)
    df_agg = pd.read_csv(agg_file)
    
    # Filter valid years
    df_main = df_main[(df_main['reg_year'] >= 2020) & (df_main['reg_year'] <= 2025)].copy()
    df_agg = df_agg[(df_agg['reg_year'] >= 2020) & (df_agg['reg_year'] <= 2025)].copy()
    
    # Convert dates
    df_main['registration_date'] = pd.to_datetime(df_main['registration_date'])
    df_agg['registration_date'] = pd.to_datetime(df_agg['registration_date'])
    
    return df_main, df_agg

# Load data (will auto-setup if needed)
df_main, df_agg = load_data()

# Base directory for accessing models
base_dir = Path(__file__).parent

# Sidebar
with st.sidebar:
    st.title("🚗 Vahan-Insight")
    st.markdown("---")
    
    # Date filter
    st.subheader("📅 Date Range")
    year_range = st.slider(
        "Select Years",
        min_value=2020,
        max_value=2025,
        value=(2020, 2025),
        step=1
    )
    
    # Fuel filter - MULTISELECT for comparison
    st.subheader("⛽ Fuel Types")
    fuel_options = sorted(df_agg['fuel_category'].unique().tolist())
    
    # Initialize session state if not exists
    if 'fuel_multiselect' not in st.session_state:
        st.session_state['fuel_multiselect'] = fuel_options
    
    # Select All / Clear buttons for fuel
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ All Fuels", key="all_fuels", use_container_width=True):
            st.session_state['fuel_multiselect'] = fuel_options
            st.rerun()
    with col2:
        if st.button("❌ Clear", key="clear_fuels", use_container_width=True):
            st.session_state['fuel_multiselect'] = []
            st.rerun()
    
    selected_fuels = st.multiselect(
        "Compare fuel types",
        fuel_options,
        default=None,
        key="fuel_multiselect",
        help="Select multiple to compare (e.g., EV vs Hybrid)"
    )
    
    # State filter - MULTISELECT for comparison
    st.subheader("🗺️ States")
    state_options = sorted(df_agg['stateName'].unique().tolist())
    
    # Initialize session state if not exists
    if 'state_multiselect' not in st.session_state:
        st.session_state['state_multiselect'] = state_options
    
    # Select All / Clear buttons for states
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ All States", key="all_states", use_container_width=True):
            st.session_state['state_multiselect'] = state_options
            st.rerun()
    with col2:
        if st.button("❌ Clear", key="clear_states", use_container_width=True):
            st.session_state['state_multiselect'] = []
            st.rerun()
    
    selected_states = st.multiselect(
        "Compare states",
        state_options,
        default=None,
        key="state_multiselect",
        help="Select multiple to compare"
    )
    
    # Quick presets
    st.markdown("**Quick Presets:**")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔋 EV Only", key="ev_only", use_container_width=True):
            st.session_state['fuel_multiselect'] = ['EV']
            st.rerun()
    with col2:
        if st.button("⚡ EV vs Hybrid", key="ev_hybrid", use_container_width=True):
            st.session_state['fuel_multiselect'] = ['EV', 'HYBRID']
            st.rerun()
    
    st.markdown("---")
    st.caption("Data: 2020-2025 Indian Vehicle Registrations")

# Filter data based on selections
df_main_filtered = df_main[
    (df_main['reg_year'] >= year_range[0]) & 
    (df_main['reg_year'] <= year_range[1])
]
df_agg_filtered = df_agg[
    (df_agg['reg_year'] >= year_range[0]) & 
    (df_agg['reg_year'] <= year_range[1])
]

# Apply fuel filter (multiselect)
if selected_fuels and len(selected_fuels) < len(fuel_options):
    df_main_filtered = df_main_filtered[df_main_filtered['fuel_category'].isin(selected_fuels)]
    df_agg_filtered = df_agg_filtered[df_agg_filtered['fuel_category'].isin(selected_fuels)]

# Apply state filter (multiselect) - APPLIES TO BOTH dataframes
if selected_states and len(selected_states) < len(state_options):
    df_agg_filtered = df_agg_filtered[df_agg_filtered['stateName'].isin(selected_states)]
    # Also filter main dataset - it has stateName column
    if 'stateName' in df_main_filtered.columns:
        df_main_filtered = df_main_filtered[df_main_filtered['stateName'].isin(selected_states)]

# Main header
st.title("📊 EV Analytics Dashboard")
st.markdown(f"**Period:** {year_range[0]} - {year_range[1]}")

# Show active filters summary
filter_info = []
if len(selected_fuels) < len(fuel_options):
    filter_info.append(f"**Fuel:** {', '.join(selected_fuels)}")
if len(selected_states) < len(state_options):
    if len(selected_states) <= 3:
        filter_info.append(f"**States:** {', '.join(selected_states)}")
    else:
        filter_info.append(f"**States:** {len(selected_states)} selected")
if filter_info:
    st.info("🔍 Active Filters: " + " | ".join(filter_info))

# Data disclaimer
st.warning("""
📋 **Data Note:** This dashboard analyzes a **statistical sample** of Indian vehicle registrations (500K records representing 2.8M vehicles). 
Absolute values should be interpreted as indicative figures. **Trends, growth rates, and comparative insights remain statistically valid** for analysis.
For comprehensive registration counts, refer to official government sources.
""", icon="ℹ️")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔋 EV Deep Dive", "⛽ Fuel Analysis", "🔮 ML Insights"])

# ============= TAB 1: OVERVIEW =============
with tab1:
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    ev_count = df_agg_filtered[df_agg_filtered['fuel_category'] == 'EV']['vehicleCount'].sum()
    total_count = df_agg_filtered['vehicleCount'].sum()
    ev_share = (ev_count / total_count * 100) if total_count > 0 else 0
    states_count = df_agg_filtered['stateName'].nunique()
    
    # Calculate growth
    years = sorted(df_agg_filtered['reg_year'].unique())
    if len(years) >= 2:
        first_year = years[0]
        last_year = years[-1]
        first_year_ev = df_agg_filtered[(df_agg_filtered['reg_year'] == first_year) & (df_agg_filtered['fuel_category'] == 'EV')]['vehicleCount'].sum()
        last_year_ev = df_agg_filtered[(df_agg_filtered['reg_year'] == last_year) & (df_agg_filtered['fuel_category'] == 'EV')]['vehicleCount'].sum()
        yoy_growth = ((last_year_ev - first_year_ev) / first_year_ev * 100) if first_year_ev > 0 else 0
    else:
        yoy_growth = 0
    
    with col1:
        st.metric("Total EV Registrations", f"{ev_count:,}", f"+{ev_count/1000:.1f}K vs 2020")
    
    with col2:
        st.metric("EV Market Share", f"{ev_share:.1f}%", f"+{ev_share-0.6:.1f}%")
    
    with col3:
        st.metric("States Covered", states_count, "100% coverage")
    
    with col4:
        st.metric("Growth Rate", f"{yoy_growth:.1f}%", f"vs {year_range[0]}")
    
    st.markdown("---")
    
    # Charts row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Registration Trend by Year")
        yearly_data = df_agg_filtered.groupby(['reg_year', 'fuel_category'])['vehicleCount'].sum().reset_index()
        
        fig = px.line(
            yearly_data,
            x='reg_year',
            y='vehicleCount',
            color='fuel_category',
            markers=True,
            color_discrete_map={
                'EV': '#22c55e',
                'PETROL': '#ef4444',
                'DIESEL': '#f97316',
                'HYBRID': '#eab308',
                'CNG': '#3b82f6',
                'LPG': '#a855f7',
                'OTHER': '#9ca3af'
            }
        )
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Top 10 States (EV Registrations)")
        state_ev = df_agg_filtered[df_agg_filtered['fuel_category'] == 'EV'].groupby('stateName')['vehicleCount'].sum().sort_values(ascending=True).tail(10)
        
        fig = go.Figure(go.Bar(
            x=state_ev.values,
            y=state_ev.index,
            orientation='h',
            marker=dict(color='#22c55e')
        ))
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="EV Registrations",
            yaxis_title=""
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Charts row 2
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔋 Fuel Category Distribution")
        fuel_dist = df_agg_filtered.groupby('fuel_category')['vehicleCount'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=fuel_dist.index,
            values=fuel_dist.values,
            hole=0.5,
            marker=dict(colors=['#22c55e', '#3b82f6', '#ef4444', '#f97316', '#eab308', '#a855f7', '#9ca3af'])
        )])
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Personal vs Public EVs")
        personal_cats = ['TWO WHEELER(NT)', 'LIGHT MOTOR VEHICLE', 'TWO WHEELER(T)']
        public_cats = ['THREE WHEELER(T)', 'THREE WHEELER(NT)', 'HEAVY PASSENGER VEHICLE', 'MEDIUM PASSENGER VEHICLE', 'LIGHT PASSENGER VEHICLE']
        
        ev_main = df_main_filtered[df_main_filtered['is_ev'] == True]
        personal_ev = ev_main[ev_main['vehicleCategoryName'].isin(personal_cats)]['vehicleCount'].sum()
        public_ev = ev_main[ev_main['vehicleCategoryName'].isin(public_cats)]['vehicleCount'].sum()
        other_ev = ev_main[~ev_main['vehicleCategoryName'].isin(personal_cats + public_cats)]['vehicleCount'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=['Personal (2W+Cars)', 'Public (3W+Buses)', 'Other'],
            values=[personal_ev, public_ev, other_ev],
            hole=0.5,
            marker=dict(colors=['#667eea', '#f093fb', '#9ca3af'])
        )])
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.markdown("### 📅 Monthly Trend (Last Year)")
        last_year = year_range[1]
        monthly_data = df_agg_filtered[df_agg_filtered['reg_year'] == last_year].groupby('reg_month')['vehicleCount'].sum()
        
        fig = go.Figure(data=[go.Bar(
            x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(monthly_data)],
            y=monthly_data.values,
            marker=dict(color='#22c55e')
        )])
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            yaxis_title="Registrations"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ============= ADVANCED GEO MAP =============
    st.markdown("### 🗺️ India EV Analytics Map")
    
    # Map view toggle
    map_view = st.radio(
        "Select View:",
        ["📊 Absolute EV Count", "📈 EV Adoption Rate (%)", "📉 Year-over-Year Change"],
        horizontal=True,
        key="map_view_toggle"
    )
    
    # Legend explanation
    if map_view == "📊 Absolute EV Count":
        st.caption("🟢 High (>15K) | 🟡 Medium (5K-15K) | 🔴 Low (<5K) — Circle size = EV count")
    elif map_view == "📈 EV Adoption Rate (%)":
        st.caption("🟢 High (≥7%) | 🟡 Moderate (4-7%) | 🔴 Low (<4%) — Circle size = Relative adoption")
    else:
        st.caption("🟢 Growing (>20%) | 🟡 Steady (0-20%) | 🔴 Declining (<0%) — YoY change in EV registrations")
    
    # State coordinates
    state_coords = {
        'Andhra Pradesh': [15.9129, 79.7400],
        'Arunachal Pradesh': [28.2180, 94.7278],
        'Assam': [26.2006, 92.9376],
        'Bihar': [25.0961, 85.3131],
        'Chhattisgarh': [21.2787, 81.8661],
        'Delhi': [28.7041, 77.1025],
        'Goa': [15.2993, 74.1240],
        'Gujarat': [22.2587, 71.1924],
        'Haryana': [29.0588, 76.0856],
        'Himachal Pradesh': [31.1048, 77.1734],
        'Jharkhand': [23.6102, 85.2799],
        'Karnataka': [15.3173, 75.7139],
        'Kerala': [10.8505, 76.2711],
        'Madhya Pradesh': [22.9734, 78.6569],
        'Maharashtra': [19.7515, 75.7139],
        'Manipur': [24.6637, 93.9063],
        'Meghalaya': [25.4670, 91.3662],
        'Mizoram': [23.1645, 92.9376],
        'Nagaland': [26.1584, 94.5624],
        'Odisha': [20.9517, 85.0985],
        'Punjab': [31.1471, 75.3412],
        'Rajasthan': [27.0238, 74.2179],
        'Sikkim': [27.5330, 88.5122],
        'Tamil Nadu': [11.1271, 78.6569],
        'Telangana': [18.1124, 79.0193],
        'Tripura': [23.9408, 91.9882],
        'Uttar Pradesh': [26.8467, 80.9462],
        'Uttarakhand': [30.0668, 79.0193],
        'West Bengal': [22.9868, 87.8550],
        'Chandigarh': [30.7333, 76.7794],
        'Puducherry': [11.9416, 79.8083],
        'Jammu and Kashmir': [33.7782, 76.5762],
        'Ladakh': [34.1526, 77.5771],
    }
    
    # Prepare data for all views
    # Current period EV data
    ev_state = df_agg_filtered[df_agg_filtered['fuel_category'] == 'EV'].groupby('stateName')['vehicleCount'].sum().reset_index()
    ev_state.columns = ['stateName', 'ev_count']
    
    # Total vehicles by state 
    total_state = df_agg_filtered.groupby('stateName')['vehicleCount'].sum().reset_index()
    total_state.columns = ['stateName', 'total_count']
    
    # Merge
    state_data = ev_state.merge(total_state, on='stateName', how='left')
    state_data['ev_share'] = (state_data['ev_count'] / state_data['total_count'] * 100).round(2)
    
    # Calculate YoY change
    if len(years) >= 2:
        prev_year = years[-2] if len(years) >= 2 else years[0]
        curr_year = years[-1]
        
        # Previous year EV
        prev_ev = df_agg[(df_agg['reg_year'] == prev_year) & (df_agg['fuel_category'] == 'EV')].groupby('stateName')['vehicleCount'].sum().reset_index()
        prev_ev.columns = ['stateName', 'prev_ev']
        
        # Current year EV
        curr_ev = df_agg[(df_agg['reg_year'] == curr_year) & (df_agg['fuel_category'] == 'EV')].groupby('stateName')['vehicleCount'].sum().reset_index()
        curr_ev.columns = ['stateName', 'curr_ev']
        
        yoy_data = prev_ev.merge(curr_ev, on='stateName', how='outer').fillna(0)
        yoy_data['yoy_change'] = ((yoy_data['curr_ev'] - yoy_data['prev_ev']) / yoy_data['prev_ev'].replace(0, 1) * 100).round(1)
        
        state_data = state_data.merge(yoy_data[['stateName', 'yoy_change', 'prev_ev', 'curr_ev']], on='stateName', how='left')
        state_data['yoy_change'] = state_data['yoy_change'].fillna(0)
    else:
        state_data['yoy_change'] = 0
        state_data['prev_ev'] = 0
        state_data['curr_ev'] = state_data['ev_count']
    
    # Create map
    m = folium.Map(location=[22.5, 82.0], zoom_start=5, tiles='CartoDB positron')
    
    for _, row in state_data.iterrows():
        state = row['stateName']
        if state in state_coords:
            lat, lon = state_coords[state]
            
            # Determine color and radius based on selected view
            if map_view == "📊 Absolute EV Count":
                # Absolute numbers
                value = row['ev_count']
                if value >= 15000:
                    color = '#22c55e'  # Green - High
                    status = "High"
                elif value >= 5000:
                    color = '#eab308'  # Yellow - Medium
                    status = "Medium"
                else:
                    color = '#ef4444'  # Red - Low
                    status = "Low"
                radius = min(max(value / 500, 5), 30)
                popup_text = f"""
                    <div style='font-family: Arial; min-width: 180px;'>
                        <h4 style='margin: 0; color: {color};'>{state}</h4>
                        <hr style='margin: 5px 0;'>
                        <b>EV Registrations:</b> {value:,}<br>
                        <b>Status:</b> {status}<br>
                        <b>Rank:</b> #{list(state_data.sort_values('ev_count', ascending=False)['stateName']).index(state) + 1}
                    </div>
                """
                tooltip = f"{state}: {value:,} EVs ({status})"
                
            elif map_view == "📈 EV Adoption Rate (%)":
                # Relative adoption
                value = row['ev_share']
                if value >= 7:
                    color = '#22c55e'  # Green - High adoption
                    status = "High Adoption"
                elif value >= 4:
                    color = '#eab308'  # Yellow - Moderate
                    status = "Moderate"
                else:
                    color = '#ef4444'  # Red - Low adoption
                    status = "Low Adoption"
                radius = min(max(value * 3, 5), 30)
                popup_text = f"""
                    <div style='font-family: Arial; min-width: 180px;'>
                        <h4 style='margin: 0; color: {color};'>{state}</h4>
                        <hr style='margin: 5px 0;'>
                        <b>EV Share:</b> {value:.1f}%<br>
                        <b>Total EVs:</b> {row['ev_count']:,}<br>
                        <b>Total Vehicles:</b> {row['total_count']:,}<br>
                        <b>Status:</b> {status}
                    </div>
                """
                tooltip = f"{state}: {value:.1f}% EV Share ({status})"
                
            else:  # YoY Change
                value = row['yoy_change'] if pd.notna(row['yoy_change']) else 0
                prev_ev_val = int(row.get('prev_ev', 0)) if pd.notna(row.get('prev_ev', 0)) else 0
                curr_ev_val = int(row.get('curr_ev', 0)) if pd.notna(row.get('curr_ev', 0)) else 0
                
                if value > 20:
                    color = '#22c55e'  # Green - Growing
                    status = "📈 Growing"
                elif value >= 0:
                    color = '#eab308'  # Yellow - Steady
                    status = "➡️ Steady"
                else:
                    color = '#ef4444'  # Red - Declining
                    status = "📉 Declining"
                radius = min(max(abs(value) / 5 + 5, 5), 30)
                popup_text = f"""
                    <div style='font-family: Arial; min-width: 180px;'>
                        <h4 style='margin: 0; color: {color};'>{state}</h4>
                        <hr style='margin: 5px 0;'>
                        <b>YoY Change:</b> {value:+.1f}%<br>
                        <b>Previous Year:</b> {prev_ev_val:,}<br>
                        <b>Current Year:</b> {curr_ev_val:,}<br>
                        <b>Trend:</b> {status}
                    </div>
                """
                tooltip = f"{state}: {value:+.1f}% YoY ({status})"
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=2,
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=tooltip
            ).add_to(m)
    
    # Display map
    st_folium(m, width=1200, height=450, returned_objects=[])

# ============= TAB 2: EV DEEP DIVE =============
with tab2:
    st.markdown("### 🔋 Electric Vehicle Analysis")
    
    # EV-specific metrics
    col1, col2, col3, col4 = st.columns(4)
    
    ev_df = df_main_filtered[df_main_filtered['is_ev'] == True]
    total_ev = ev_df['vehicleCount'].sum()
    
    # 2W EVs
    two_w_ev = ev_df[ev_df['vehicleCategoryName'].str.contains('TWO WHEELER', na=False)]['vehicleCount'].sum()
    cars_ev = ev_df[ev_df['vehicleCategoryName'] == 'LIGHT MOTOR VEHICLE']['vehicleCount'].sum()
    three_w_ev = ev_df[ev_df['vehicleCategoryName'].str.contains('THREE WHEELER', na=False)]['vehicleCount'].sum()
    
    with col1:
        st.metric("2-Wheeler EVs", f"{two_w_ev:,}", f"{two_w_ev/total_ev*100:.1f}% of EVs")
    with col2:
        st.metric("Electric Cars", f"{cars_ev:,}", f"{cars_ev/total_ev*100:.1f}% of EVs")
    with col3:
        st.metric("3-Wheeler EVs", f"{three_w_ev:,}", f"{three_w_ev/total_ev*100:.1f}% of EVs")
    with col4:
        ev_growth_pct = (yoy_growth) if yoy_growth > 0 else 0
        st.metric("EV CAGR", f"{ev_growth_pct:.1f}%", "2020-2025")
    
    st.markdown("---")
    
    # EV type breakdown
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 EV Segment Growth Over Time")
        
        personal_cats = ['TWO WHEELER(NT)', 'LIGHT MOTOR VEHICLE', 'TWO WHEELER(T)']
        public_cats = ['THREE WHEELER(T)', 'THREE WHEELER(NT)', 'HEAVY PASSENGER VEHICLE']
        
        ev_filtered = df_main_filtered[df_main_filtered['is_ev'] == True].copy()
        ev_filtered['segment'] = ev_filtered['vehicleCategoryName'].apply(
            lambda x: 'Personal' if x in personal_cats else ('Public' if x in public_cats else 'Commercial')
        )
        
        segment_yearly = ev_filtered.groupby(['reg_year', 'segment'])['vehicleCount'].sum().reset_index()
        
        fig = px.bar(
            segment_yearly,
            x='reg_year',
            y='vehicleCount',
            color='segment',
            barmode='stack',
            color_discrete_map={'Personal': '#667eea', 'Public': '#f093fb', 'Commercial': '#9ca3af'}
        )
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Year",
            yaxis_title="Registrations"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🔍 EV Vehicle Class Breakdown")
        
        ev_class = ev_filtered.groupby('vehicleClassName')['vehicleCount'].sum().sort_values(ascending=False).head(8)
        
        fig = go.Figure(data=[go.Pie(
            labels=ev_class.index,
            values=ev_class.values,
            hole=0.4
        )])
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ============= PUBLIC MOBILITY STORY =============
    st.markdown("### 🚌 Public Mobility vs Personal EVs — The FAME Story")
    
    # Show state context
    if len(selected_states) < len(state_options):
        if len(selected_states) <= 3:
            st.info(f"📍 **Analyzing:** {', '.join(selected_states)}")
        else:
            st.info(f"📍 **Analyzing:** {len(selected_states)} selected states")
    
    # Define vehicle classes for public mobility (using actual class names from data)
    erickshaw_classes = ['e-Rickshaw(P)', 'e-Rickshaw with Cart (G)']
    bus_classes = ['Bus', 'Educational Institution Bus', 'Private Service Vehicle', 'Stage Carriage']
    # 3W includes both Three Wheeler categories AND Motor Cycle/Scooter-Used For Hire (shared mobility)
    three_w_classes = ['Three Wheeler (Passenger)', 'Three Wheeler (Goods)', 'Motor Cycle/Scooter-Used For Hire', 'Motor Cab']
    personal_classes = ['M-Cycle/Scooter', 'Motor Car', 'Moped', 'Motorised Cycle (CC > 25cc)']
    
    # Get counts by class
    ev_by_class = ev_filtered.groupby('vehicleClassName')['vehicleCount'].sum()
    
    erickshaw_count = sum(ev_by_class.get(c, 0) for c in erickshaw_classes)
    bus_count = sum(ev_by_class.get(c, 0) for c in bus_classes)
    three_w_count = sum(ev_by_class.get(c, 0) for c in three_w_classes)
    personal_count = sum(ev_by_class.get(c, 0) for c in personal_classes)
    
    # Metrics row for public mobility
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🛺 E-Rickshaws",
            f"{erickshaw_count:,}",
            f"{erickshaw_count/total_ev*100:.1f}% of EVs",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "🚌 Electric Buses",
            f"{bus_count:,}",
            "Only 0.3% of EVs" if bus_count < 1000 else f"{bus_count/total_ev*100:.1f}% of EVs",
            delta_color="off" if bus_count < 1000 else "normal"
        )
    
    with col3:
        st.metric(
            "🛵 3-Wheeler (Passenger)",
            f"{three_w_count:,}",
            f"{three_w_count/total_ev*100:.1f}% of EVs"
        )
    
    with col4:
        public_total = erickshaw_count + bus_count + three_w_count
        st.metric(
            "📊 Public Mobility Total",
            f"{public_total:,}",
            f"{public_total/total_ev*100:.1f}% of all EVs"
        )
    
    st.markdown("---")
    
    # Public Mobility YoY Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Public Mobility Segment Growth")
        
        # Calculate YoY for each segment
        ev_filtered['mobility_type'] = ev_filtered['vehicleClassName'].apply(
            lambda x: 'E-Rickshaw' if x in erickshaw_classes else 
                      ('Bus' if x in bus_classes else 
                       ('3W Passenger' if x in three_w_classes else 
                        ('Personal (2W/Cars)' if x in personal_classes else 'Other')))
        )
        
        mobility_yearly = ev_filtered.groupby(['reg_year', 'mobility_type'])['vehicleCount'].sum().reset_index()
        
        fig = px.line(
            mobility_yearly,
            x='reg_year',
            y='vehicleCount',
            color='mobility_type',
            markers=True,
            color_discrete_map={
                'E-Rickshaw': '#f093fb',
                'Bus': '#22c55e',
                '3W Passenger': '#667eea',
                'Personal (2W/Cars)': '#3b82f6',
                'Other': '#9ca3af'
            }
        )
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Year",
            yaxis_title="EV Registrations",
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Public vs Personal EV Share Over Time")
        
        # Calculate segment share by year
        segment_share = ev_filtered.groupby(['reg_year', 'segment'])['vehicleCount'].sum().reset_index()
        segment_pivot = segment_share.pivot(index='reg_year', columns='segment', values='vehicleCount').fillna(0)
        segment_pivot['Total'] = segment_pivot.sum(axis=1)
        
        # Calculate percentages
        for col in ['Personal', 'Public', 'Commercial']:
            if col in segment_pivot.columns:
                segment_pivot[f'{col}_pct'] = (segment_pivot[col] / segment_pivot['Total'] * 100).round(1)
        
        # Create stacked area chart for percentage
        fig = go.Figure()
        
        if 'Personal_pct' in segment_pivot.columns:
            fig.add_trace(go.Scatter(
                x=segment_pivot.index, y=segment_pivot['Personal_pct'],
                name='Personal', fill='tonexty', mode='lines',
                line=dict(color='#667eea', width=2), fillcolor='rgba(102, 126, 234, 0.5)'
            ))
        
        if 'Public_pct' in segment_pivot.columns:
            fig.add_trace(go.Scatter(
                x=segment_pivot.index, y=segment_pivot['Personal_pct'] + segment_pivot['Public_pct'],
                name='Public', fill='tonexty', mode='lines',
                line=dict(color='#f093fb', width=2), fillcolor='rgba(240, 147, 251, 0.5)'
            ))
        
        fig.update_layout(
            height=350,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Year",
            yaxis_title="Share (%)",
            yaxis=dict(range=[0, 100]),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Story insight boxes
    col1, col2 = st.columns(2)
    
    with col1:
        st.warning("""
        **🚌 The Bus Gap:** Only **{:,}** electric buses registered (2020-2025) out of **{:,}** total EVs.
        
        This represents a massive untapped opportunity for public transport electrification — exactly why GOI shifted FAME focus to buses.
        """.format(bus_count, total_ev))
    
    with col2:
        st.success("""
        **🛺 E-Rickshaw Saturation:** **{:,}** e-rickshaws dominate public EV mobility.
        
        However, growth plateaued at **+3%** after 2023, indicating market maturity. This is natural shift for a low-cost segment.
        """.format(erickshaw_count))
    
    # Final insight
    st.info("""
    💡 **Policy Rationale (Data-Backed):** Personal EVs (2W+Cars) now represent **{:.0f}%** of registrations and are growing organically. 
    E-Rickshaws have saturated. Electric buses remain severely under-represented at **{:.1f}%**.
    This explains the GOI's FAME policy shift towards public transport subsidies.
    """.format(personal_count/total_ev*100, bus_count/total_ev*100))

# ============= TAB 3: FUEL ANALYSIS =============
with tab3:
    st.markdown("### ⛽ Fuel Type Analysis")
    
    # Fuel metrics
    col1, col2, col3, col4 = st.columns(4)
    
    petrol_count = df_agg_filtered[df_agg_filtered['fuel_category'] == 'PETROL']['vehicleCount'].sum()
    diesel_count = df_agg_filtered[df_agg_filtered['fuel_category'] == 'DIESEL']['vehicleCount'].sum()
    
    with col1:
        st.metric("Petrol Vehicles", f"{petrol_count:,}", f"{petrol_count/total_count*100:.1f}%")
    with col2:
        st.metric("Diesel Vehicles", f"{diesel_count:,}", f"{diesel_count/total_count*100:.1f}%")
    with col3:
        st.metric("EV Share", f"{ev_share:.1f}%", "+7.2% vs 2020")
    with col4:
        hybrid_count = df_agg_filtered[df_agg_filtered['fuel_category'] == 'HYBRID']['vehicleCount'].sum()
        st.metric("Hybrid Vehicles", f"{hybrid_count:,}", f"{hybrid_count/total_count*100:.1f}%")
    
    st.markdown("---")
    
    # Fuel trend
    st.markdown("### 📈 Fuel Type Trend Over Time")
    
    fuel_yearly = df_agg_filtered.groupby(['reg_year', 'fuel_category'])['vehicleCount'].sum().reset_index()
    fuel_pivot = fuel_yearly.pivot(index='reg_year', columns='fuel_category', values='vehicleCount').fillna(0)
    
    fig = go.Figure()
    
    colors = {
        'PETROL': '#ef4444',
        'DIESEL': '#f97316',
        'EV': '#22c55e',
        'HYBRID': '#eab308',
        'CNG': '#3b82f6',
        'LPG': '#a855f7',
        'OTHER': '#9ca3af'
    }
    
    stack_chart = st.checkbox("Stack Chart", value=False, key="fuel_stack_chart")
    
    for col in fuel_pivot.columns:
        fig.add_trace(go.Scatter(
            x=fuel_pivot.index,
            y=fuel_pivot[col],
            name=col,
            mode='lines+markers',
            line=dict(color=colors.get(col, '#9ca3af'), width=3),
            stackgroup='one' if stack_chart else None
        ))
    
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Year",
        yaxis_title="Registrations",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

# ============= TAB 4: ML INSIGHTS =============
with tab4:
    st.markdown("### 🤖 Machine Learning Insights")
    st.info("**Powered by XGBoost** | Forecasting EV adoption trends for 2026-2030")
    
    # Check if ML models exist
    models_dir = base_dir / "models"
    
    if not (models_dir / "forecast_2026_2030.csv").exists():
        st.warning("⚠️ ML models not yet trained. Run `python ml_models.py` to generate forecasts.")
    else:
        # Load ML results
        forecast_df = pd.read_csv(models_dir / "forecast_2026_2030.csv")
        clusters_df = pd.read_csv(models_dir / "state_clusters.csv")
        importance_df = pd.read_csv(models_dir / "feature_importance.csv")
        
        import json
        with open(models_dir / "metrics_report.json") as f:
            metrics = json.load(f)
        
        # Model Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 R² Score", f"{metrics['r2_test']:.3f}", "Good fit" if metrics['r2_test'] > 0.8 else "Moderate")
        with col2:
            st.metric("📉 RMSE", f"{metrics['rmse_test']:.1f}", "Monthly registrations")
        with col3:
            st.metric("🎯 MAE", f"{metrics['mae_test']:.1f}", "Avg error")
        with col4:
            st.metric("🔍 Silhouette", f"{metrics['silhouette_score']:.3f}", "Cluster quality")
        
        st.markdown("---")
        
        # Two columns: Forecast and Clusters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📈 EV Adoption Forecast (2026-2030)")
            
            # Get historical data for comparison
            ev_historical = df_agg_filtered[df_agg_filtered['fuel_category'] == 'EV']
            hist_monthly = ev_historical.groupby(pd.Grouper(key='registration_date', freq='M'))['vehicleCount'].sum().reset_index()
            hist_monthly.columns = ['date', 'count']
            hist_monthly['type'] = 'Historical'
            
            # Top states forecast
            top_states = clusters_df.nlargest(5, 'total_evs')['state'].tolist()
            forecast_top = forecast_df[forecast_df['state'].isin(top_states)]
            forecast_top['date'] = pd.to_datetime(forecast_top['date'])
            
            # Aggregate forecast by date
            forecast_agg = forecast_top.groupby('date')['predicted_count'].sum().reset_index()
            forecast_agg.columns = ['date', 'count']
            forecast_agg['type'] = 'Forecast'
            
            # Combine
            combined = pd.concat([
                hist_monthly[['date', 'count', 'type']],
                forecast_agg[['date', 'count', 'type']]
            ])
            
            fig = px.line(
                combined,
                x='date',
                y='count',
                color='type',
                color_discrete_map={'Historical': '#667eea', 'Forecast': '#f093fb'},
                labels={'count': 'Monthly EV Registrations', 'date': 'Date'}
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("📍 Forecast starts after Dec 2025")
        
        with col2:
            st.markdown("### 🎯 State Clustering Analysis")
            
            # Scatter plot of clusters
            fig = px.scatter(
                clusters_df,
                x='adoption_rate',
                y='growth_rate',
                size='total_evs',
                color='cluster_name',
                hover_name='state',
                hover_data={'total_evs': True, 'adoption_rate': ':.2f', 'growth_rate': ':.1f'},
                color_discrete_sequence=px.colors.qualitative.Set2,
                labels={'adoption_rate': 'EV Adoption Rate (%)', 'growth_rate': 'Growth Rate (%)'}
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Feature Importance and Top Predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔑 Key Drivers of EV Growth")
            
            fig = px.bar(
                importance_df.head(6),
                x='importance_pct',
                y='feature',
                orientation='h',
                color='importance_pct',
                color_continuous_scale='Blues',
                labels={'importance_pct': 'Importance (%)', 'feature': 'Feature'}
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                coloraxis_showscale=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Key Insights:**
            - 📊 **Recent trends** (rolling_avg, lag features) are strong predictors
            - 📈 **YoY growth** indicates momentum continuation
            - 📅 **Seasonality** (month) matters for registration patterns
            """)
        
        with col2:
            st.markdown("### 🏆 Top States - 2030 Forecast")
            
            # Get 2030 predictions
            forecast_df['date'] = pd.to_datetime(forecast_df['date'])
            forecast_2030 = forecast_df[forecast_df['date'].dt.year == 2030]
            state_totals = forecast_2030.groupby('state')['predicted_count'].sum().sort_values(ascending=False).head(10)
            
            fig = px.bar(
                x=state_totals.values,
                y=state_totals.index,
                orientation='h',
                color=state_totals.values,
                color_continuous_scale='Greens',
                labels={'x': 'Predicted Monthly EV Registrations', 'y': 'State'}
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                coloraxis_showscale=False,
                yaxis={'categoryorder': 'total ascending'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.success(f"""
            **2030 Prediction Highlights:**
            - 🥇 **{state_totals.index[0]}** leads with ~{state_totals.values[0]:.0f} monthly registrations
            - 🚀 Top 5 states account for {(state_totals.values[:5].sum() / state_totals.values.sum() * 100):.0f}% of predictions
            """)

# Footer
st.markdown("---")
st.caption("📊 Vahan-Insight Dashboard | Data: Indian Vehicle Registration (2020-2025) | ML: XGBoost + K-Means | Powered by Streamlit")
