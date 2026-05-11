"""Agent - abre el agente externo en una nueva pestana."""
import streamlit as st
from theme import page_header

AGENT_URL = "http://10.132.12.178:7860/"


def render():
    page_header(
        eyebrow="AI - Agent",
        title="Agent Kim",
        subtitle="Agente conversacional de inteligencia artificial para tu piso de planta.",
    )

    st.markdown(
        f"""
        <div style='
            display:flex;flex-direction:column;align-items:center;
            justify-content:center;padding:80px 20px;
            background:#f8fafc;border:2px dashed #cbd5e1;
            border-radius:16px;margin-top:10px;
        '>
            <div style='font-size:4rem;margin-bottom:16px;'>&#x1F916;</div>
            <div style='font-size:1.2rem;font-weight:700;color:#0f172a;margin-bottom:8px;'>
                Agent Kim
            </div>
            <div style='font-size:0.88rem;color:#64748b;text-align:center;
                        max-width:340px;margin-bottom:28px;'>
                El agente se ejecuta en un servicio externo.
                Haz clic en el boton para abrirlo en una nueva pestana.
            </div>
            <a href="{AGENT_URL}" target="_blank" rel="noopener noreferrer"
               style='
                   display:inline-flex;align-items:center;gap:10px;
                   background:linear-gradient(135deg,#4f46e5,#7c3aed);
                   color:#fff;font-weight:700;font-size:1rem;
                   padding:14px 32px;border-radius:12px;
                   text-decoration:none;
                   box-shadow:0 4px 14px rgba(79,70,229,.35);
               '>
                &#x1F680;&nbsp; Abrir Agent Kim
            </a>
            <div style='margin-top:16px;font-size:0.75rem;color:#94a3b8;'>
                {AGENT_URL}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )