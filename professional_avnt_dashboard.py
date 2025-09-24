#!/usr/bin/env python3
"""
🎯 DASHBOARD PROFESSIONNEL AVEC AVNT
Design moderne sombre mais coloré, animations subtiles, données complètes
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import time
import json
import requests
from datetime import datetime, timedelta
import logging
import threading
import os
import sys

# Imports des moteurs
from technical_analysis_engine import technical_engine

# Import Support/Résistance Daily
try:
    from daily_sr_calculator import get_complete_daily_sr_levels
    SR_DAILY_AVAILABLE = True
    print("✅ S/R Daily disponible")
except ImportError:
    SR_DAILY_AVAILABLE = False
    print("⚠️ S/R Daily indisponible")

# Configuration Streamlit
st.set_page_config(
    page_title="🎯 Professional AVNT Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS PROFESSIONNEL - Moderne Sombre mais Coloré avec Animations Subtiles
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #1a1d29;
        --bg-card: #242938;
        --bg-hover: #2a2f3f;
        
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-yellow: #f59e0b;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        
        --border: #334155;
        --border-hover: #475569;
        
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    /* Background principal */
    .main .block-container {
        background: var(--bg-primary);
        padding: 1.5rem;
        max-width: 100%;
    }
    
    /* Header professionnel */
    .main-header {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-card) 100%);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple), var(--accent-cyan));
        animation: headerLine 3s ease-in-out infinite;
    }
    
    @keyframes headerLine {
        0% { left: -100%; }
        50% { left: 100%; }
        100% { left: -100%; }
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        color: var(--text-primary);
        margin: 0;
        animation: fadeIn 0.8s ease-out;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        color: var(--text-secondary);
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Cards professionnelles */
    .pro-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
        position: relative;
    }
    
    .pro-card:hover {
        background: var(--bg-hover);
        border-color: var(--border-hover);
        box-shadow: var(--shadow);
        transform: translateY(-1px);
    }
    
    /* Métriques */
    .metric-card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
        text-align: center;
    }
    
    .metric-card:hover {
        background: var(--bg-hover);
        border-color: var(--border-hover);
        transform: translateY(-1px);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 1.5rem;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Prix principal */
    .price-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 0.5rem;
        animation: priceUpdate 0.3s ease-out;
    }
    
    @keyframes priceUpdate {
        0% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    /* Signaux avec couleurs subtiles */
    .signal-bullish {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
        border-left: 3px solid var(--accent-green);
    }
    
    .signal-bearish {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
        border-left: 3px solid var(--accent-red);
    }
    
    .signal-neutral {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1), rgba(139, 92, 246, 0.05));
        border-left: 3px solid var(--accent-purple);
    }
    
    /* Support/Résistance */
    .support-item {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-green);
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .support-item:hover {
        background: var(--bg-hover);
        border-color: var(--accent-green);
    }
    
    .resistance-item {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent-red);
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
    }
    
    .resistance-item:hover {
        background: var(--bg-hover);
        border-color: var(--accent-red);
    }
    
    .sr-price {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--text-primary);
    }
    
    .sr-distance {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    .sr-description {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.25rem;
        font-style: italic;
    }
    
    /* Daily levels */
    .daily-level {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem;
        margin: 0.5rem;
        transition: all 0.2s ease;
    }
    
    .daily-level:hover {
        background: var(--bg-hover);
        border-color: var(--border-hover);
    }
    
    .daily-level-title {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.875rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .daily-level-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 1rem;
        color: var(--text-primary);
    }
    
    /* Crypto grid */
    .crypto-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .crypto-item {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1.25rem;
        transition: all 0.2s ease;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .crypto-item:hover {
        background: var(--bg-hover);
        border-color: var(--border-hover);
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }
    
    .crypto-symbol {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--text-primary);
        margin-bottom: 0.5rem;
    }
    
    .crypto-price {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
    }
    
    .crypto-signal {
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        font-weight: 500;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
    }
    
    /* Couleurs de signaux */
    .signal-bullish-text { color: var(--accent-green); }
    .signal-bearish-text { color: var(--accent-red); }
    .signal-neutral-text { color: var(--accent-purple); }
    
    .bg-bullish { background: rgba(16, 185, 129, 0.1); }
    .bg-bearish { background: rgba(239, 68, 68, 0.1); }
    .bg-neutral { background: rgba(139, 92, 246, 0.1); }
    
    /* Indicateurs techniques */
    .indicator-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .indicator-item {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    .indicator-item:hover {
        background: var(--bg-hover);
        border-color: var(--border-hover);
    }
    
    .indicator-value {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 1.1rem;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .indicator-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--bg-primary) !important;
        border-right: 1px solid var(--border);
    }
    
    /* Boutons */
    .stButton > button {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--bg-hover);
        border-color: var(--accent-blue);
        color: var(--accent-blue);
        transform: translateY(-1px);
    }
    
    /* Progress bars */
    .stProgress .st-bo {
        background: var(--accent-blue);
        border-radius: 4px;
        animation: progressFill 0.3s ease-out;
    }
    
    @keyframes progressFill {
        from { width: 0; }
        to { width: var(--progress-width, 50%); }
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        color: var(--text-primary);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        color: var(--text-primary);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border);
        position: relative;
    }
    
    .section-header::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 0;
        width: 50px;
        height: 2px;
        background: var(--accent-blue);
        animation: sectionLine 0.5s ease-out;
    }
    
    @keyframes sectionLine {
        from { width: 0; }
        to { width: 50px; }
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .crypto-grid {
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        }
        
        .indicator-grid {
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        }
    }
</style>
""", unsafe_allow_html=True)

# ========== FONCTIONS UTILITAIRES ==========

def safe_rerun():
    """Rerun compatible toutes versions"""
    try:
        st.rerun()
    except:
        try:
            st.experimental_rerun()
        except:
            pass

def format_large_number(num):
    """Formater grands nombres"""
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return f"{num:.2f}"

def format_price(price, decimals=2):
    """Formater prix avec bon nombre de décimales"""
    if price < 1:
        return f"${price:.4f}"
    elif price < 100:
        return f"${price:.2f}"
    else:
        return f"${price:,.0f}"

def get_signal_color_class(signal):
    """Retourner classe CSS selon signal"""
    if signal == 'BULLISH':
        return 'signal-bullish-text'
    elif signal == 'BEARISH':
        return 'signal-bearish-text'
    else:
        return 'signal-neutral-text'

def get_signal_bg_class(signal):
    """Retourner classe background selon signal"""
    if signal == 'BULLISH':
        return 'bg-bullish'
    elif signal == 'BEARISH':
        return 'bg-bearish'
    else:
        return 'bg-neutral'

# ========== FONCTIONS DE DONNÉES AVEC AVNT ==========

@st.cache_data(ttl=300)
def get_live_signals(symbol: str):
    """Signaux techniques temps réel"""
    try:
        signals = technical_engine.generate_signals(symbol)
        return signals
    except Exception as e:
        st.error(f"Erreur signaux {symbol}: {e}")
        return None

@st.cache_data(ttl=300)
def get_sr_daily_levels(symbol: str):
    """Support/Résistance daily complets"""
    if not SR_DAILY_AVAILABLE:
        return {'supports': [], 'resistances': [], 'daily_levels': {}, 'current_price': 0}
    
    try:
        return get_complete_daily_sr_levels(symbol)
    except Exception as e:
        st.error(f"Erreur S/R daily {symbol}: {e}")
        return {'supports': [], 'resistances': [], 'daily_levels': {}, 'current_price': 0}

@st.cache_data(ttl=600)
def get_defi_prices():
    """Prix DeFi temps réel"""
    defi_tokens = {
        'UNI': 'uniswap', 'AAVE': 'aave', 'COMP': 'compound-governance-token',
        'SUSHI': 'sushi', 'CRV': 'curve-dao-token', 'YFI': 'yearn-finance',
        'SNX': 'synthetix-network-token', 'MKR': 'maker', 'CAKE': 'pancakeswap-token'
    }
    
    try:
        ids = ','.join(defi_tokens.values())
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies=usd&include_24hr_change=true&include_market_cap=true"
        
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            defi_data = {}
            for symbol, coin_id in defi_tokens.items():
                if coin_id in data:
                    defi_data[symbol] = {
                        'price': data[coin_id]['usd'],
                        'change_24h': data[coin_id].get('usd_24h_change', 0),
                        'market_cap': data[coin_id].get('usd_market_cap', 0)
                    }
            
            return defi_data
        else:
            return get_synthetic_defi_prices()
            
    except:
        return get_synthetic_defi_prices()

def get_synthetic_defi_prices():
    """Prix DeFi synthétiques"""
    import random
    
    base_prices = {
        'UNI': 7.50, 'AAVE': 150.00, 'COMP': 45.00, 'SUSHI': 1.20,
        'CRV': 0.85, 'YFI': 8500.00, 'SNX': 2.80, 'MKR': 1200.00, 'CAKE': 2.50
    }
    
    defi_data = {}
    for symbol, base_price in base_prices.items():
        variation = random.uniform(-0.05, 0.05)
        current_price = base_price * (1 + variation)
        change_24h = random.uniform(-15, 15)
        market_cap = current_price * random.uniform(100000000, 10000000000)
        
        defi_data[symbol] = {
            'price': round(current_price, 4),
            'change_24h': round(change_24h, 2),
            'market_cap': round(market_cap, 0)
        }
    
    return defi_data

def get_market_summary():
    """Résumé marché global AVEC AVNT"""
    try:
        # AVNT inclus comme token normal
        major_coins = ['BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'AVNT']
        
        bullish_count = 0
        bearish_count = 0
        total_strength = 0
        
        for coin in major_coins:
            try:
                signals = get_live_signals(coin)
                if signals:
                    if signals['overall_signal'] == 'BULLISH':
                        bullish_count += 1
                    elif signals['overall_signal'] == 'BEARISH':
                        bearish_count += 1
                    
                    total_strength += signals['strength']
            except:
                continue
        
        avg_strength = total_strength / len(major_coins) if major_coins else 50
        
        if bullish_count > bearish_count:
            market_sentiment = 'BULLISH'
        elif bearish_count > bullish_count:
            market_sentiment = 'BEARISH'
        else:
            market_sentiment = 'MIXED'
        
        return {
            'sentiment': market_sentiment,
            'bullish_coins': bullish_count,
            'bearish_coins': bearish_count,
            'neutral_coins': len(major_coins) - bullish_count - bearish_count,
            'average_strength': avg_strength
        }
    except:
        return {
            'sentiment': 'UNKNOWN',
            'bullish_coins': 0,
            'bearish_coins': 0,
            'neutral_coins': 0,
            'average_strength': 50
        }

# ========== PAGES PROFESSIONNELLES ==========

def render_main_dashboard():
    """Dashboard principal professionnel"""
    
    # Header professionnel
    st.markdown('''
    <div class="main-header">
        <h1>📊 Professional Trading Dashboard</h1>
        <p>Analyse technique avancée avec données temps réel</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Contrôles en ligne propre
    col1, col2, col3, col4 = st.columns([1, 1, 2, 1])
    with col1:
        auto_refresh = st.checkbox("⚡ Auto-refresh (30s)")
    with col2:
        if st.button("🔄 Actualiser", key="refresh_main"):
            st.cache_data.clear()
            safe_rerun()
    with col3:
        # AVNT comme token normal dans la liste
        selected_crypto = st.selectbox("🎯 Crypto Principal", 
                                     ["BTC", "ETH", "XRP", "SOL", "ADA", "AVNT"],
                                     index=0)
    with col4:
        show_advanced = st.checkbox("📊 Mode Avancé", value=True)
    
    if auto_refresh:
        time.sleep(30)
        safe_rerun()
    
    # === SECTION PRIX PRINCIPAL ===
    signals = get_live_signals(selected_crypto)
    sr_data = get_sr_daily_levels(selected_crypto)
    
    if signals:
        # Prix principal avec métriques
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
        with col1:
            st.markdown(f'''
            <div class="pro-card">
                <div class="price-display">{format_price(signals['price'])}</div>
                <div class="metric-label">💰 {selected_crypto} Prix Temps Réel</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            signal_class = get_signal_color_class(signals['overall_signal'])
            signal_bg = get_signal_bg_class(signals['overall_signal'])
            signal_emoji = "🚀" if signals['overall_signal'] == 'BULLISH' else "🐻" if signals['overall_signal'] == 'BEARISH' else "🔄"
            
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value {signal_class}">{signal_emoji} {signals["overall_signal"]}</div>
                <div class="metric-label">Signal Global</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value">{signals['strength']:.0f}%</div>
                <div class="metric-label">💪 Force Signal</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            if 'change_24h' in signals:
                change_color = 'signal-bullish-text' if signals['change_24h'] > 0 else 'signal-bearish-text'
                change_sign = '+' if signals['change_24h'] > 0 else ''
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-value {change_color}">{change_sign}{signals['change_24h']:.2f}%</div>
                    <div class="metric-label">📈 Change 24h</div>
                </div>
                ''', unsafe_allow_html=True)
    
    # Section Signaux Techniques supprimée comme demandé
    
    # === SECTION SUPPORT/RÉSISTANCE COMPLÈTE ===
    if sr_data['current_price'] > 0:
        st.markdown('<div class="section-header">📊 Support & Résistance Daily Complets</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🟢 SUPPORTS")
            
            if sr_data['supports']:
                for i, support in enumerate(sr_data['supports'][:5]):  # Top 5 supports
                    distance = ((signals['price'] - support['price']) / signals['price']) * 100
                    strength_stars = "⭐" * min(5, int(support['strength'] * 5))
                    
                    # Nom descriptif comme dans la page dédiée
                    support_name = support.get('description', support.get('type', 'Support technique'))
                    
                    st.markdown(f'''
                    <div class="support-item">
                        <div class="sr-price">Support {i+1} - {support_name}: {format_price(support['price'])}</div>
                        <div class="sr-distance">📏 Distance: -{distance:.2f}%</div>
                        <div class="sr-description">💪 Force: {strength_stars} ({support['strength']:.1f})</div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("Aucun support détecté")
        
        with col2:
            st.markdown("### 🔴 RÉSISTANCES")
            
            if sr_data['resistances']:
                for i, resistance in enumerate(sr_data['resistances'][:5]):  # Top 5 résistances
                    distance = ((resistance['price'] - signals['price']) / signals['price']) * 100
                    strength_stars = "⭐" * min(5, int(resistance['strength'] * 5))
                    
                    # Nom descriptif comme dans la page dédiée
                    resistance_name = resistance.get('description', resistance.get('type', 'Résistance technique'))
                    
                    st.markdown(f'''
                    <div class="resistance-item">
                        <div class="sr-price">Résistance {i+1} - {resistance_name}: {format_price(resistance['price'])}</div>
                        <div class="sr-distance">📏 Distance: +{distance:.2f}%</div>
                        <div class="sr-description">💪 Force: {strength_stars} ({resistance['strength']:.1f})</div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("Aucune résistance détectée")
        
        # === NIVEAUX DAILY DÉTAILLÉS ===
        if sr_data.get('daily_levels'):
            daily = sr_data['daily_levels']
            
            st.markdown("### 📅 Niveaux Daily Détaillés")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f'''
                <div class="daily-level">
                    <div class="daily-level-title">📅 Daily 24h</div>
                    <div class="daily-level-value">High: {format_price(daily.get('daily_high_24h', 0))}</div>
                    <div class="daily-level-value">Low: {format_price(daily.get('daily_low_24h', 0))}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown(f'''
                <div class="daily-level">
                    <div class="daily-level-title">📆 Weekly</div>
                    <div class="daily-level-value">High: {format_price(daily.get('weekly_high', 0))}</div>
                    <div class="daily-level-value">Low: {format_price(daily.get('weekly_low', 0))}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown(f'''
                <div class="daily-level">
                    <div class="daily-level-title">🎯 Pivots</div>
                    <div class="daily-level-value">24h: {format_price(daily.get('pivot_24h', 0))}</div>
                    <div class="daily-level-value">7d: {format_price(daily.get('pivot_7d', 0))}</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                if signals:
                    daily_high = daily.get('daily_high_24h', signals['price'])
                    daily_low = daily.get('daily_low_24h', signals['price'])
                    
                    if daily_high != daily_low:
                        daily_position = ((signals['price'] - daily_low) / (daily_high - daily_low)) * 100
                        position_color = 'signal-bullish-text' if daily_position > 70 else 'signal-bearish-text' if daily_position < 30 else 'signal-neutral-text'
                        
                        st.markdown(f'''
                        <div class="daily-level">
                            <div class="daily-level-title">📊 Position Range</div>
                            <div class="daily-level-value {position_color}">{daily_position:.1f}%</div>
                            <div class="sr-description">Position dans la range daily</div>
                        </div>
                        ''', unsafe_allow_html=True)
    
    # === SECTION MARCHÉ GLOBAL ===
    st.markdown('<div class="section-header">🌍 Vue d\'Ensemble du Marché</div>', unsafe_allow_html=True)
    market_summary = get_market_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sentiment_color = get_signal_color_class(market_summary['sentiment'])
        sentiment_emoji = "🚀" if market_summary['sentiment'] == 'BULLISH' else "🐻" if market_summary['sentiment'] == 'BEARISH' else "🔄"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value {sentiment_color}">{sentiment_emoji}</div>
            <div class="metric-label">Sentiment Global</div>
            <div class="metric-value {sentiment_color}">{market_summary["sentiment"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value signal-bullish-text">{market_summary['bullish_coins']}</div>
            <div class="metric-label">📈 Cryptos Bullish</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value signal-bearish-text">{market_summary['bearish_coins']}</div>
            <div class="metric-label">📉 Cryptos Bearish</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{market_summary['average_strength']:.0f}%</div>
            <div class="metric-label">⚡ Force Moyenne</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # === SECTION PORTFOLIO MULTI-CRYPTOS ===
    st.markdown('<div class="section-header">💎 Portfolio Multi-Cryptos</div>', unsafe_allow_html=True)
    
    # AVNT traité comme token normal
    all_cryptos = ["BTC", "ETH", "XRP", "SOL", "ADA", "AVNT"]
    other_cryptos = [c for c in all_cryptos if c != selected_crypto]
    
    st.markdown('<div class="crypto-grid">', unsafe_allow_html=True)
    
    for crypto in other_cryptos:
        try:
            crypto_signals = get_live_signals(crypto)
            if crypto_signals:
                signal_color = get_signal_color_class(crypto_signals['overall_signal'])
                signal_bg = get_signal_bg_class(crypto_signals['overall_signal'])
                signal_emoji = "🚀" if crypto_signals['overall_signal'] == 'BULLISH' else "🐻" if crypto_signals['overall_signal'] == 'BEARISH' else "🔄"
                
                st.markdown(f'''
                <div class="crypto-item">
                    <div class="crypto-symbol">{crypto}</div>
                    <div class="crypto-price {signal_color}">{format_price(crypto_signals['price'])}</div>
                    <div class="crypto-signal {signal_bg} {signal_color}">
                        {signal_emoji} {crypto_signals['overall_signal']} ({crypto_signals['strength']:.0f}%)
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # S/R pour mode avancé
                if show_advanced:
                    crypto_sr = get_sr_daily_levels(crypto)
                    if crypto_sr['supports'] or crypto_sr['resistances']:
                        support_info = f"S: {format_price(crypto_sr['supports'][0]['price'])}" if crypto_sr['supports'] else ""
                        resistance_info = f"R: {format_price(crypto_sr['resistances'][0]['price'])}" if crypto_sr['resistances'] else ""
                        st.caption(f"🟢 {support_info} 🔴 {resistance_info}")
        except:
            st.markdown(f'''
            <div class="crypto-item">
                <div class="crypto-symbol">{crypto}</div>
                <div style="color: var(--text-muted);">⚠️ Données indisponibles</div>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_live_signals():
    """Page signaux live"""
    
    st.markdown('''
    <div class="main-header">
        <h1>🎯 Signaux Live Multi-Cryptos</h1>
        <p>Analyse technique temps réel pour tous les tokens</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Contrôles
    col1, col2, col3 = st.columns(3)
    with col1:
        auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", key="signals_refresh")
    with col2:
        if st.button("🔄 Actualiser Signaux", key="refresh_signals"):
            st.cache_data.clear()
    with col3:
        show_details = st.checkbox("📊 Détails Avancés", value=True)
    
    if auto_refresh:
        time.sleep(30)
        safe_rerun()
    
    # Vue d'ensemble marché
    market_summary = get_market_summary()
    
    st.markdown('<div class="section-header">🌍 Vue d\'Ensemble du Marché</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    sentiment_color = get_signal_color_class(market_summary['sentiment'])
    sentiment_emoji = "🚀" if market_summary['sentiment'] == 'BULLISH' else "🐻" if market_summary['sentiment'] == 'BEARISH' else "🔄"
    
    with col1:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value {sentiment_color}">{sentiment_emoji}</div>
            <div class="metric-label">Sentiment Global</div>
            <div class="metric-value {sentiment_color}">{market_summary["sentiment"]}</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value signal-bullish-text">{market_summary['bullish_coins']}</div>
            <div class="metric-label">📈 Cryptos Bullish</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value signal-bearish-text">{market_summary['bearish_coins']}</div>
            <div class="metric-label">📉 Cryptos Bearish</div>
        </div>
        ''', unsafe_allow_html=True)
    
    with col4:
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{market_summary['average_strength']:.0f}%</div>
            <div class="metric-label">⚡ Force Moyenne</div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Signaux par crypto - AVNT traité normalement
    st.markdown('<div class="section-header">🎯 Signaux Multi-Cryptos</div>', unsafe_allow_html=True)
    
    major_coins = ['BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'AVNT', 'DOT']
    
    for coin in major_coins:
        try:
            signals = get_live_signals(coin)
            
            if signals:
                signal_class = f"signal-{signals['overall_signal'].lower()}"
                signal_emoji = "🚀" if signals['overall_signal'] == 'BULLISH' else "🐻" if signals['overall_signal'] == 'BEARISH' else "🔄"
                
                with st.expander(f"{signal_emoji} {coin} - {signals['overall_signal']} ({signals['strength']:.0f}%)", expanded=False):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Prix et signal principal
                        signal_color = get_signal_color_class(signals['overall_signal'])
                        
                        st.markdown(f'''
                        <div class="pro-card {signal_class}">
                            <div class="price-display">{format_price(signals["price"])}</div>
                            <div class="metric-label">💰 Prix {coin}</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        # Progress bar pour la force
                        st.progress(signals['strength'] / 100)
                        
                        # Trend
                        if 'trend' in signals:
                            trend_emoji = "📈" if signals['trend'] == 'UPTREND' else "📉" if signals['trend'] == 'DOWNTREND' else "➡️"
                            trend_color = get_signal_color_class(signals['overall_signal'])
                            st.markdown(f'''
                            <div style="text-align: center; color: var(--text-secondary); font-family: 'Inter', sans-serif; font-weight: 500;">
                                {trend_emoji} Trend: {signals['trend']}
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    with col2:
                        # Indicateurs techniques
                        st.markdown("**📊 Indicateurs:**")
                        
                        rsi_color = get_signal_color_class('BULLISH' if 30 <= signals['rsi'] <= 70 else 'BEARISH')
                        st.markdown(f'''
                        <div class="indicator-item">
                            <div class="indicator-value {rsi_color}">{signals['rsi']:.1f}</div>
                            <div class="indicator-label">RSI</div>
                        </div>
                        ''', unsafe_allow_html=True)
                        
                        if 'macd' in signals:
                            macd_val = signals['macd']['macd']
                            macd_color = get_signal_color_class('BULLISH' if macd_val > 0 else 'BEARISH')
                            st.markdown(f'''
                            <div class="indicator-item">
                                <div class="indicator-value {macd_color}">{macd_val:.3f}</div>
                                <div class="indicator-label">MACD</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        if 'bollinger' in signals:
                            bb = signals['bollinger']
                            current_price = signals['price']
                            bb_position = "Haute" if current_price > bb['upper'] else "Basse" if current_price < bb['lower'] else "Moyenne"
                            bb_color = get_signal_color_class('BEARISH' if bb_position == "Haute" else 'BULLISH' if bb_position == "Basse" else 'NEUTRAL')
                            st.markdown(f'''
                            <div style="text-align: center; color: var(--text-secondary); font-family: 'Inter', sans-serif; font-weight: 500;">
                                📊 Bollinger: {bb_position}
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    with col3:
                        # Support/Résistance
                        st.markdown("**📊 S/R Niveaux:**")
                        
                        if 'support' in signals:
                            st.markdown(f'''
                            <div class="support-item">
                                <div class="sr-price">🟢 Support: {format_price(signals['support'])}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        if 'resistance' in signals:
                            st.markdown(f'''
                            <div class="resistance-item">
                                <div class="sr-price">🔴 Résistance: {format_price(signals['resistance'])}</div>
                            </div>
                            ''', unsafe_allow_html=True)
                        
                        if 'volume_trend' in signals:
                            volume_emoji = "📊" if signals['volume_trend'] == 'INCREASING' else "📉"
                            volume_color = get_signal_color_class('BULLISH' if signals['volume_trend'] == 'INCREASING' else 'BEARISH')
                            st.markdown(f'''
                            <div style="text-align: center; color: var(--text-secondary); font-family: 'Inter', sans-serif; font-weight: 500;">
                                {volume_emoji} Volume: {signals['volume_trend']}
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    # Détails avancés
                    if show_details:
                        st.markdown("**🔍 Analyse Détaillée:**")
                        
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            if 'moving_averages' in signals:
                                ma = signals['moving_averages']
                                st.markdown(f'''
                                <div class="daily-level">
                                    <div class="daily-level-title">Moyennes Mobiles:</div>
                                    <div class="daily-level-value">SMA 20: {format_price(ma.get('sma_20', 0))}</div>
                                    <div class="daily-level-value">SMA 50: {format_price(ma.get('sma_50', 0))}</div>
                                    <div class="daily-level-value">EMA 12: {format_price(ma.get('ema_12', 0))}</div>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        with detail_col2:
                            if 'stochastic' in signals:
                                stoch = signals['stochastic']
                                st.markdown(f'''
                                <div class="daily-level">
                                    <div class="daily-level-title">Stochastique:</div>
                                    <div class="daily-level-value">%K: {stoch.get('k', 0):.1f}</div>
                                    <div class="daily-level-value">%D: {stoch.get('d', 0):.1f}</div>
                                </div>
                                ''', unsafe_allow_html=True)
                            
                            # Timestamp
                            if 'timestamp' in signals:
                                st.markdown(f'''
                                <div style="text-align: center; font-size: 0.875rem; color: var(--text-muted);">
                                    ⏰ Mis à jour: {signals['timestamp'].strftime('%H:%M:%S')}
                                </div>
                                ''', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"❌ Erreur {coin}: {e}")

def render_sr_complete_page():
    """Page Support/Résistance avec design comme l'ancien dashboard"""
    
    st.title("📊 Supports & Résistances Daily")
    st.markdown("**Niveaux basés sur Daily High/Low + Pivots + Fibonacci + Analytics**")
    
    # Contrôles
    col1, col2 = st.columns(2)
    with col1:
        selected_crypto = st.selectbox("🎯 Crypto", ["BTC", "ETH", "XRP", "SOL", "ADA", "AVNT"])
    with col2:
        if st.button("🔄 Actualiser S/R"):
            st.cache_data.clear()
    
    # Récupérer données
    signals = get_live_signals(selected_crypto)
    sr_data = get_sr_daily_levels(selected_crypto)
    
    if sr_data['current_price'] > 0 and signals:
        current_price = signals['price']
        
        # Prix principal
        st.metric(f"💰 Prix {selected_crypto}", f"${current_price:,.2f}")
        
        # S/R en colonnes comme l'ancien
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🟢 SUPPORTS")
            
            if sr_data['supports']:
                for i, support in enumerate(sr_data['supports']):
                    distance = ((current_price - support['price']) / current_price) * 100
                    strength_stars = "⭐" * min(5, int(support['strength'] * 5))
                    
                    # Style simplifié comme l'ancien
                    st.markdown(f"""
                    **Support {i+1}** - {support.get('description', support.get('type', 'Support technique'))}
                    - 💰 ${support['price']:,.2f}
                    - 📏 -{distance:.2f}%
                    - 💪 {strength_stars}
                    """)
            else:
                st.info("Aucun support détecté pour ce token")
        
        with col2:
            st.subheader("🔴 RÉSISTANCES")
            
            if sr_data['resistances']:
                for i, resistance in enumerate(sr_data['resistances']):
                    distance = ((resistance['price'] - current_price) / current_price) * 100
                    strength_stars = "⭐" * min(5, int(resistance['strength'] * 5))
                    
                    # Style simplifié comme l'ancien
                    st.markdown(f"""
                    **Résistance {i+1}** - {resistance.get('description', resistance.get('type', 'Résistance technique'))}
                    - 💰 ${resistance['price']:,.2f}
                    - 📏 +{distance:.2f}%
                    - 💪 {strength_stars}
                    """)
            else:
                st.info("Aucune résistance détectée pour ce token")
        
        # Niveaux Daily détaillés comme l'ancien
        if sr_data.get('daily_levels'):
            daily = sr_data['daily_levels']
            
            st.subheader("📅 Niveaux Daily Détaillés")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📅 Daily 24h**")
                st.metric("High", f"${daily.get('daily_high_24h', 0):,.2f}")
                st.metric("Low", f"${daily.get('daily_low_24h', 0):,.2f}")
            
            with col2:
                st.markdown("**📆 Weekly**")
                st.metric("High", f"${daily.get('weekly_high', 0):,.2f}")
                st.metric("Low", f"${daily.get('weekly_low', 0):,.2f}")
            
            with col3:
                st.markdown("**🎯 Pivots**")
                st.metric("24h", f"${daily.get('pivot_24h', 0):,.2f}")
                st.metric("7d", f"${daily.get('pivot_7d', 0):,.2f}")
                
                # Position dans range
                daily_high = daily.get('daily_high_24h', current_price)
                daily_low = daily.get('daily_low_24h', current_price)
                
                if daily_high != daily_low:
                    daily_position = ((current_price - daily_low) / (daily_high - daily_low)) * 100
                    st.metric("Position Daily", f"{daily_position:.1f}%")
    else:
        st.error("❌ Impossible de charger les données S/R pour ce token")

def render_defi_page():
    """Page DeFi professionnelle"""
    
    st.markdown('''
    <div class="main-header">
        <h1>🏦 Écosystème DeFi</h1>
        <p>Prix temps réel des tokens DeFi principaux</p>
    </div>
    ''', unsafe_allow_html=True)
    
    if st.button("🔄 Actualiser DeFi"):
        st.cache_data.clear()
    
    defi_data = get_defi_prices()
    
    if defi_data:
        st.markdown('<div class="section-header">💎 Tokens DeFi Principaux</div>', unsafe_allow_html=True)
        
        # Grid DeFi professionnelle
        st.markdown('<div class="crypto-grid">', unsafe_allow_html=True)
        
        for symbol, data in defi_data.items():
            price = data['price']
            change = data['change_24h']
            market_cap = data.get('market_cap', 0)
            
            emoji = "🚀" if change > 5 else "📈" if change > 0 else "📉" if change < -5 else "➡️"
            change_color = 'signal-bullish-text' if change > 0 else 'signal-bearish-text'
            change_sign = '+' if change > 0 else ''
            
            st.markdown(f'''
            <div class="crypto-item">
                <div class="crypto-symbol">{symbol}</div>
                <div class="crypto-price">{format_price(price)}</div>
                <div class="crypto-signal {change_color}">
                    {emoji} {change_sign}{change:.2f}%
                </div>
                <div class="sr-description">MC: {format_large_number(market_cap)}</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Tableau détaillé
        st.markdown('<div class="section-header">📊 Détails DeFi</div>', unsafe_allow_html=True)
        
        defi_df = pd.DataFrame.from_dict(defi_data, orient='index')
        defi_df['Prix'] = defi_df['price'].apply(lambda x: format_price(x))
        defi_df['Change 24h'] = defi_df['change_24h'].apply(lambda x: f"{x:+.2f}%")
        defi_df['Market Cap'] = defi_df['market_cap'].apply(lambda x: format_large_number(x))
        
        st.dataframe(defi_df[['Prix', 'Change 24h', 'Market Cap']], use_container_width=True)
    else:
        st.error("❌ Impossible de charger les prix DeFi")

def main():
    """Application principale professionnelle"""
    
    # Sidebar professionnelle
    with st.sidebar:
        st.markdown('''
        <div class="main-header">
            <h2>📊 Professional Dashboard</h2>
            <p>Navigation</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Navigation principale
        page = st.selectbox(
            "📍 Choisir une page:",
            [
                "🏠 Dashboard Principal",
                "🎯 Signaux Live",
                "📊 Supports & Résistances",
                "🏦 Écosystème DeFi"
            ]
        )
        
        st.markdown("---")
        
        # Status système
        st.markdown("**📊 Status Système:**")
        st.markdown('''
        <div class="metric-card">
            <div class="metric-label">✅ Signaux temps réel</div>
        </div>
        ''', unsafe_allow_html=True)
        
        if SR_DAILY_AVAILABLE:
            st.markdown('''
            <div class="metric-card">
                <div class="metric-label">✅ S/R Daily actifs</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('''
        <div class="metric-card">
            <div class="metric-label">✅ AVNT intégré</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Infos temps réel
        current_time = datetime.now()
        st.markdown(f'''
        <div class="daily-level">
            <div class="daily-level-title">⏰ Temps Réel</div>
            <div class="daily-level-value">{current_time.strftime('%H:%M:%S')}</div>
            <div class="sr-description">{current_time.strftime('%Y-%m-%d')}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Test rapide AVNT
        st.markdown("---")
        st.markdown("**🧪 Test AVNT:**")
        if st.button("🧪 Test Signaux AVNT"):
            try:
                avnt_signals = get_live_signals('AVNT')
                if avnt_signals:
                    st.success(f"✅ AVNT: {format_price(avnt_signals['price'])}")
                    st.info(f"🎯 {avnt_signals['overall_signal']} ({avnt_signals['strength']:.0f}%)")
                else:
                    st.error("❌ Erreur AVNT")
            except Exception as e:
                st.error(f"❌ {e}")
        
        # Liens rapides
        st.markdown("---")
        st.markdown("**🔗 Actions Rapides:**")
        if st.button("🗑️ Vider Cache"):
            st.cache_data.clear()
            st.success("Cache vidé!")
    
    # Routing des pages
    if page == "🏠 Dashboard Principal":
        render_main_dashboard()
    elif page == "🎯 Signaux Live":
        render_live_signals()
    elif page == "📊 Supports & Résistances":
        render_sr_complete_page()
    elif page == "🏦 Écosystème DeFi":
        render_defi_page()

if __name__ == "__main__":
    main()
