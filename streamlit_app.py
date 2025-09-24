#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌐 DASHBOARD TRADING - VERSION WEB PUBLIQUE
📊 Accessible depuis n'importe où dans le monde
🚀 Optimisé pour Streamlit Cloud
"""

import streamlit as st
import sys
import os

# Configuration page - DOIT ÊTRE EN PREMIER
st.set_page_config(
    page_title="💎 Trading Dashboard Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Marquer comme Streamlit pour éviter conflits d'encodage
sys._called_from_streamlit = True

# Message de bienvenue
st.markdown("""
<div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 20px;">
    <h1 style="color: white; margin: 0;">🌐 Dashboard Trading Pro</h1>
    <p style="color: white; margin: 10px 0 0 0;">Accessible depuis partout dans le monde !</p>
</div>
""", unsafe_allow_html=True)

# Import du dashboard principal
try:
    # Essayer d'importer le dashboard existant
    exec(open('CUSTOM_TRADING_DASHBOARD.py', encoding='utf-8').read())
except FileNotFoundError:
      st.error("❌ Fichier CUSTOM_TRADING_DASHBOARD.py introuvable")
    st.info("📋 Assurez-vous que tous les fichiers sont présents dans le repository GitHub")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement du dashboard: {e}")
    st.info("🔧 Vérifiez la configuration et les dépendances")
    
    # Afficher un dashboard de base en cas d'erreur
    st.markdown("## 📊 Dashboard de Base")
    st.success("✅ L'application fonctionne ! Le dashboard principal sera chargé une fois tous les fichiers uploadés.")

