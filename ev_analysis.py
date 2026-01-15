#!/usr/bin/env python3
"""
Vahan-Insight: EV Revolution Analysis & Visualization
=======================================================
Creates stunning interactive visualizations for the Indian EV market analysis.

Features:
- EV vs ICE growth trend analysis
- State-wise EV adoption heat map
- FAME-II subsidy correlation timeline
- Segment analysis (Personal vs Public EVs)

Usage:
    python ev_analysis.py

Author: Vahan-Insight Team
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VIZ_DIR = BASE_DIR / "visualizations"
VIZ_DIR.mkdir(exist_ok=True)

# Color palette - Premium dark theme inspired
COLORS = {
    'ev': '#00D4AA',           # Vibrant teal for EV
    'petrol': '#FF6B6B',       # Coral red for petrol
    'diesel': '#4ECDC4',       # Light teal for diesel
    'hybrid': '#FFE66D',       # Yellow for hybrid
    'personal': '#667EEA',     # Purple for personal
    'public': '#F093FB',       # Pink for public
    'background': '#0F0F23',   # Dark background
    'card': '#1A1A2E',         # Card background
    'text': '#EAEAEA',         # Light text
    'grid': '#2D2D44',         # Grid lines
    'accent': '#00D4AA',       # Accent color
}

# FAME-II Policy Timeline
FAME_EVENTS = [
    {'date': '2019-04-01', 'event': 'FAME-II Launch', 'subsidy': '₹10,000/kWh', 'color': '#00D4AA'},
    {'date': '2021-06-11', 'event': 'Subsidy Boost (2W)', 'subsidy': '₹15,000/kWh', 'color': '#667EEA'},
    {'date': '2023-06-01', 'event': 'Subsidy Reduced', 'subsidy': '₹10,000/kWh', 'color': '#FF6B6B'},
    {'date': '2024-03-31', 'event': 'FAME-II Extended', 'subsidy': 'Extended', 'color': '#FFE66D'},
]

# State coordinates for map
STATE_COORDS = {
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


def load_data():
    """Load cleaned datasets."""
    df_main = pd.read_csv(DATA_DIR / "vehicle_registrations_cleaned.csv")
    df_agg = pd.read_csv(DATA_DIR / "state_month_fuel_cleaned.csv")
    
    # Filter to valid years (2020-2025)
    df_main = df_main[(df_main['reg_year'] >= 2020) & (df_main['reg_year'] <= 2025)]
    df_agg = df_agg[(df_agg['reg_year'] >= 2020) & (df_agg['reg_year'] <= 2025)]
    
    return df_main, df_agg


def create_ev_growth_trend(df_agg):
    """Create EV vs ICE growth trend visualization."""
    logger.info("📊 Creating EV vs ICE growth trend chart...")
    
    # Aggregate by year and fuel category
    yearly = df_agg.groupby(['reg_year', 'fuel_category'])['vehicleCount'].sum().reset_index()
    
    # Pivot for easier plotting
    pivot = yearly.pivot(index='reg_year', columns='fuel_category', values='vehicleCount').fillna(0)
    
    # Calculate EV share
    pivot['Total'] = pivot.sum(axis=1)
    pivot['EV_Share'] = (pivot.get('EV', 0) / pivot['Total'] * 100).round(2)
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('📈 Fuel Type Registrations by Year', '🔋 EV Market Share Growth'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Stacked area chart for fuel types
    for fuel, color in [('PETROL', COLORS['petrol']), ('DIESEL', COLORS['diesel']), 
                         ('EV', COLORS['ev']), ('HYBRID', COLORS['hybrid'])]:
        if fuel in pivot.columns:
            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[fuel],
                    name=fuel,
                    mode='lines+markers',
                    line=dict(width=3, color=color),
                    marker=dict(size=10, symbol='circle'),
                    fill='tonexty' if fuel != 'PETROL' else 'tozeroy',
                    hovertemplate=f'{fuel}: %{{y:,.0f}}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # EV Share line chart
    fig.add_trace(
        go.Scatter(
            x=pivot.index,
            y=pivot['EV_Share'],
            name='EV Share %',
            mode='lines+markers+text',
            line=dict(width=4, color=COLORS['accent']),
            marker=dict(size=14, symbol='diamond', color=COLORS['accent']),
            text=[f'{v:.1f}%' for v in pivot['EV_Share']],
            textposition='top center',
            textfont=dict(size=14, color=COLORS['accent']),
        ),
        row=2, col=1
    )
    
    # Add FAME-II event markers
    for event in FAME_EVENTS:
        event_year = int(event['date'][:4])
        if event_year in pivot.index:
            fig.add_vline(
                x=event_year,
                line_dash="dash",
                line_color=event['color'],
                annotation_text=event['event'],
                annotation_position="top",
                row=1, col=1
            )
    
    # Update layout for dark theme
    fig.update_layout(
        title=dict(
            text='🚗 Indian EV Revolution: Growth Trajectory (2020-2025)',
            font=dict(size=24, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text'], size=12),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(26,26,46,0.8)'
        ),
        hovermode='x unified',
        height=800
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridcolor=COLORS['grid'],
        tickfont=dict(size=12),
        title_text='Year'
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS['grid'],
        tickfont=dict(size=12),
        title_text='Registrations',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text='EV Share (%)',
        range=[0, 10],
        row=2, col=1
    )
    
    # Save
    output_path = VIZ_DIR / "ev_growth_trend.html"
    fig.write_html(output_path, include_plotlyjs='cdn')
    logger.info(f"✓ Saved: {output_path}")
    
    return fig


def create_segment_analysis(df_main):
    """Create Personal vs Public EV segment analysis."""
    logger.info("📊 Creating EV segment analysis chart...")
    
    # Define categories
    personal_cats = ['TWO WHEELER(NT)', 'LIGHT MOTOR VEHICLE', 'TWO WHEELER(T)']
    public_cats = ['THREE WHEELER(T)', 'THREE WHEELER(NT)', 'HEAVY PASSENGER VEHICLE', 
                   'MEDIUM PASSENGER VEHICLE', 'LIGHT PASSENGER VEHICLE']
    
    # Filter EVs
    ev_df = df_main[df_main['is_ev'] == True].copy()
    
    # Categorize
    def categorize_segment(cat):
        if cat in personal_cats:
            return 'Personal (2W + Cars)'
        elif cat in public_cats:
            return 'Public (3W + Buses)'
        else:
            return 'Commercial'
    
    ev_df['segment'] = ev_df['vehicleCategoryName'].apply(categorize_segment)
    
    # Aggregate
    segment_yearly = ev_df.groupby(['reg_year', 'segment'])['vehicleCount'].sum().reset_index()
    
    # Create stacked bar chart
    fig = px.bar(
        segment_yearly,
        x='reg_year',
        y='vehicleCount',
        color='segment',
        barmode='stack',
        color_discrete_map={
            'Personal (2W + Cars)': COLORS['personal'],
            'Public (3W + Buses)': COLORS['public'],
            'Commercial': '#95a5a6'
        },
        labels={'vehicleCount': 'EV Registrations', 'reg_year': 'Year', 'segment': 'Segment'},
    )
    
    # Calculate and add share percentages
    pivot = segment_yearly.pivot(index='reg_year', columns='segment', values='vehicleCount').fillna(0)
    pivot['Total'] = pivot.sum(axis=1)
    
    annotations = []
    for year in pivot.index:
        total = pivot.loc[year, 'Total']
        if 'Personal (2W + Cars)' in pivot.columns:
            personal_share = pivot.loc[year, 'Personal (2W + Cars)'] / total * 100
            annotations.append(dict(
                x=year,
                y=total + 1000,
                text=f'Personal: {personal_share:.0f}%',
                showarrow=False,
                font=dict(size=11, color=COLORS['personal'])
            ))
    
    fig.update_layout(
        title=dict(
            text='🎯 FAME Policy Shift Analysis: Personal vs Public EVs',
            font=dict(size=22, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        annotations=annotations,
        height=600
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['grid'])
    
    # Save
    output_path = VIZ_DIR / "ev_segment_analysis.html"
    fig.write_html(output_path, include_plotlyjs='cdn')
    logger.info(f"✓ Saved: {output_path}")
    
    return fig


def create_state_heatmap(df_agg):
    """Create interactive state-wise EV adoption heat map."""
    logger.info("🗺️ Creating India EV adoption heat map...")
    
    # Aggregate EV registrations by state
    ev_state = df_agg[df_agg['fuel_category'] == 'EV'].groupby('stateName')['vehicleCount'].sum().reset_index()
    ev_state.columns = ['state', 'ev_count']
    
    # Calculate EV share per state
    total_state = df_agg.groupby('stateName')['vehicleCount'].sum().reset_index()
    total_state.columns = ['state', 'total_count']
    
    state_data = ev_state.merge(total_state, on='state')
    state_data['ev_share'] = (state_data['ev_count'] / state_data['total_count'] * 100).round(2)
    state_data = state_data.sort_values('ev_count', ascending=False)
    
    # Create Folium map centered on India
    india_map = folium.Map(
        location=[20.5937, 78.9629],
        zoom_start=5,
        tiles='CartoDB dark_matter'
    )
    
    # Add markers for each state
    for _, row in state_data.iterrows():
        state = row['state']
        if state in STATE_COORDS:
            lat, lon = STATE_COORDS[state]
            
            # Size based on EV count (scaled)
            radius = min(max(row['ev_count'] / 500, 5), 40)
            
            # Color based on EV share
            if row['ev_share'] >= 7:
                color = '#00D4AA'  # High adoption
            elif row['ev_share'] >= 4:
                color = '#667EEA'  # Medium adoption
            else:
                color = '#FF6B6B'  # Low adoption
            
            # Create popup content
            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="color: {color}; margin-bottom: 10px;">{state}</h4>
                <p><b>EV Registrations:</b> {row['ev_count']:,}</p>
                <p><b>EV Share:</b> {row['ev_share']:.1f}%</p>
                <p><b>Rank:</b> #{state_data[state_data['state']==state].index[0] + 1}</p>
            </div>
            """
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{state}: {row['ev_count']:,} EVs ({row['ev_share']:.1f}%)"
            ).add_to(india_map)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background-color: rgba(15,15,35,0.9); padding: 15px; border-radius: 10px;
                font-family: Arial; color: white;">
        <h4 style="margin: 0 0 10px 0;">🔋 EV Adoption Level</h4>
        <p><span style="color: #00D4AA;">●</span> High (≥7% share)</p>
        <p><span style="color: #667EEA;">●</span> Medium (4-7% share)</p>
        <p><span style="color: #FF6B6B;">●</span> Low (<4% share)</p>
        <p style="font-size: 10px; color: #888;">Circle size = EV count</p>
    </div>
    '''
    india_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Add title
    title_html = '''
    <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                z-index: 1000; background-color: rgba(15,15,35,0.9); padding: 15px 30px; 
                border-radius: 10px; font-family: Arial; color: white;">
        <h2 style="margin: 0; color: #00D4AA;">🇮🇳 India EV Adoption Heat Map (2020-2025)</h2>
    </div>
    '''
    india_map.get_root().html.add_child(folium.Element(title_html))
    
    # Save
    output_path = VIZ_DIR / "india_ev_heatmap.html"
    india_map.save(str(output_path))
    logger.info(f"✓ Saved: {output_path}")
    
    return india_map, state_data


def create_fame_correlation(df_agg):
    """Create FAME-II subsidy correlation timeline."""
    logger.info("📊 Creating FAME-II correlation timeline...")
    
    # Monthly EV data
    df_agg['registration_date'] = pd.to_datetime(df_agg['registration_date'])
    
    ev_monthly = df_agg[df_agg['fuel_category'] == 'EV'].groupby(
        df_agg['registration_date'].dt.to_period('M')
    )['vehicleCount'].sum().reset_index()
    ev_monthly['registration_date'] = ev_monthly['registration_date'].astype(str)
    ev_monthly['date'] = pd.to_datetime(ev_monthly['registration_date'])
    
    # Create figure
    fig = go.Figure()
    
    # EV registration line
    fig.add_trace(go.Scatter(
        x=ev_monthly['date'],
        y=ev_monthly['vehicleCount'],
        mode='lines+markers',
        name='EV Registrations',
        line=dict(color=COLORS['ev'], width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(0, 212, 170, 0.2)'
    ))
    
    # Add FAME-II event markers
    for event in FAME_EVENTS:
        event_date = pd.to_datetime(event['date'])
        fig.add_vline(
            x=event_date,
            line_width=3,
            line_dash="dash",
            line_color=event['color'],
        )
        fig.add_annotation(
            x=event_date,
            y=ev_monthly['vehicleCount'].max() * 0.9,
            text=f"<b>{event['event']}</b><br>{event['subsidy']}",
            showarrow=True,
            arrowhead=2,
            arrowcolor=event['color'],
            font=dict(size=11, color=event['color']),
            bgcolor='rgba(26,26,46,0.8)',
            bordercolor=event['color'],
            borderwidth=1
        )
    
    fig.update_layout(
        title=dict(
            text='📅 FAME-II Policy Impact on EV Registrations',
            font=dict(size=22, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis_title='Date',
        yaxis_title='Monthly EV Registrations',
        hovermode='x unified',
        height=600
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['grid'])
    
    # Save
    output_path = VIZ_DIR / "fame2_correlation.html"
    fig.write_html(output_path, include_plotlyjs='cdn')
    logger.info(f"✓ Saved: {output_path}")
    
    return fig


def create_top_states_chart(state_data):
    """Create top 10 EV adopter states bar chart."""
    logger.info("📊 Creating top states chart...")
    
    top10 = state_data.head(10).copy()
    
    fig = go.Figure()
    
    # EV count bars
    fig.add_trace(go.Bar(
        y=top10['state'],
        x=top10['ev_count'],
        name='EV Registrations',
        orientation='h',
        marker=dict(
            color=top10['ev_share'],
            colorscale=[[0, COLORS['petrol']], [0.5, COLORS['hybrid']], [1, COLORS['ev']]],
            showscale=True,
            colorbar=dict(title='EV Share %')
        ),
        text=[f"{v:,} ({s:.1f}%)" for v, s in zip(top10['ev_count'], top10['ev_share'])],
        textposition='outside',
        textfont=dict(color=COLORS['text'])
    ))
    
    fig.update_layout(
        title=dict(
            text='🏆 Top 10 EV Adopter States in India',
            font=dict(size=22, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        plot_bgcolor=COLORS['card'],
        font=dict(color=COLORS['text']),
        xaxis_title='EV Registrations',
        yaxis=dict(autorange='reversed'),
        height=500,
        margin=dict(l=150)
    )
    
    fig.update_xaxes(showgrid=True, gridcolor=COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridcolor=COLORS['grid'])
    
    # Save
    output_path = VIZ_DIR / "top_ev_states.html"
    fig.write_html(output_path, include_plotlyjs='cdn')
    logger.info(f"✓ Saved: {output_path}")
    
    return fig


def create_vehicle_class_breakdown(df_main):
    """Create EV breakdown by vehicle class (sunburst chart)."""
    logger.info("📊 Creating vehicle class breakdown...")
    
    ev_df = df_main[df_main['is_ev'] == True].copy()
    
    # Aggregate by vehicle class
    class_data = ev_df.groupby('vehicleClassName')['vehicleCount'].sum().reset_index()
    class_data = class_data.sort_values('vehicleCount', ascending=False).head(10)
    
    # Create donut chart
    fig = go.Figure(data=[go.Pie(
        labels=class_data['vehicleClassName'],
        values=class_data['vehicleCount'],
        hole=0.5,
        marker=dict(
            colors=px.colors.sequential.Tealgrn,
            line=dict(color=COLORS['background'], width=2)
        ),
        textinfo='label+percent',
        textfont=dict(size=11, color=COLORS['text']),
        hovertemplate='%{label}<br>%{value:,} registrations<br>%{percent}<extra></extra>'
    )])
    
    # Add center annotation
    fig.add_annotation(
        text=f"<b>{ev_df['vehicleCount'].sum():,}</b><br>Total EVs",
        x=0.5, y=0.5,
        font=dict(size=20, color=COLORS['ev']),
        showarrow=False
    )
    
    fig.update_layout(
        title=dict(
            text='🔍 EV Registration Breakdown by Vehicle Class',
            font=dict(size=22, color=COLORS['text']),
            x=0.5
        ),
        paper_bgcolor=COLORS['background'],
        font=dict(color=COLORS['text']),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.05
        ),
        height=600
    )
    
    # Save
    output_path = VIZ_DIR / "ev_vehicle_breakdown.html"
    fig.write_html(output_path, include_plotlyjs='cdn')
    logger.info(f"✓ Saved: {output_path}")
    
    return fig


def main():
    """Run all visualizations."""
    logger.info("=" * 60)
    logger.info("📊 Vahan-Insight: EV Revolution Visualization")
    logger.info("=" * 60 + "\n")
    
    # Load data
    logger.info("Loading cleaned data...")
    df_main, df_agg = load_data()
    logger.info(f"✓ Loaded {len(df_main):,} main records, {len(df_agg):,} aggregated records\n")
    
    # Create visualizations
    logger.info("Creating visualizations...\n")
    
    create_ev_growth_trend(df_agg)
    create_segment_analysis(df_main)
    india_map, state_data = create_state_heatmap(df_agg)
    create_top_states_chart(state_data)
    create_fame_correlation(df_agg)
    create_vehicle_class_breakdown(df_main)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ All visualizations created successfully!")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {VIZ_DIR}")
    logger.info("Open the HTML files in a browser to view interactive charts.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
