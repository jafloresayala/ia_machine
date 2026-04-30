"""
Tema visual centralizado — estilo Databricks Mosaic AI / Agent Bricks.
Diseño moderno con sidebar oscuro, contenido claro, gradientes y cards.
"""
import streamlit as st

# Paleta inspirada en Mosaic AI / Agent Bricks
COLORS = {
    "bg":          "#f6f7fb",
    "surface":     "#ffffff",
    "sidebar":     "#0b1020",
    "sidebar_2":   "#111735",
    "ink":         "#0f172a",
    "ink_muted":   "#64748b",
    "border":      "#e6e8ef",
    "primary":     "#4f46e5",  # indigo
    "primary_2":   "#7c3aed",  # violet
    "accent":      "#06b6d4",  # cyan
    "success":     "#10b981",
    "warning":     "#f59e0b",
    "danger":      "#ef4444",
    "chart_1":     "#4f46e5",
    "chart_2":     "#06b6d4",
    "chart_3":     "#10b981",
    "chart_4":     "#f59e0b",
    "chart_5":     "#ef4444",
    "chart_6":     "#8b5cf6",
}


def inject_global_css():
    """Inyecta el CSS global de la aplicación."""
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

      :root {{
        --bg:        {COLORS['bg']};
        --surface:   {COLORS['surface']};
        --sidebar:   {COLORS['sidebar']};
        --sidebar-2: {COLORS['sidebar_2']};
        --ink:       {COLORS['ink']};
        --ink-muted: {COLORS['ink_muted']};
        --border:    {COLORS['border']};
        --primary:   {COLORS['primary']};
        --primary-2: {COLORS['primary_2']};
        --accent:    {COLORS['accent']};
        --success:   {COLORS['success']};
        --warning:   {COLORS['warning']};
        --danger:    {COLORS['danger']};
      }}

      html, body, [class*="css"] {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }}

      .stApp {{
        background: var(--bg);
      }}

      /* =========== SIDEBAR =========== */
      section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--sidebar) 0%, var(--sidebar-2) 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
      }}
      section[data-testid="stSidebar"] * {{
        color: #e2e8f0;
      }}
      section[data-testid="stSidebar"] .stMarkdown p,
      section[data-testid="stSidebar"] .stMarkdown span {{
        color: #cbd5e1;
      }}

      /* Sidebar brand */
      .sb-brand {{
        display: flex; align-items: center; gap: 10px;
        padding: 4px 6px 14px;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 10px;
      }}
      .sb-logo {{
        width: 38px; height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, #4f46e5 0%, #06b6d4 100%);
        display: flex; align-items: center; justify-content: center;
        font-size: 18px; color: #fff;
        box-shadow: 0 6px 18px rgba(79,70,229,0.35);
      }}
      .sb-brand-name {{
        font-weight: 700; font-size: 0.98rem; letter-spacing: -0.01em;
        color: #f8fafc; line-height: 1.1;
      }}
      .sb-brand-sub {{
        font-size: 0.7rem; color: #94a3b8; font-weight: 500;
        letter-spacing: 0.04em; text-transform: uppercase;
      }}

      /* Sidebar section label */
      .sb-section {{
        font-size: 0.68rem; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
        color: #64748b !important;
        padding: 14px 6px 6px;
      }}

      /* Sidebar nav buttons */
      section[data-testid="stSidebar"] .stButton > button {{
        background: transparent !important;
        color: #cbd5e1 !important;
        border: 1px solid transparent !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.5rem 0.8rem !important;
        min-height: 0 !important;
        font-size: 0.9rem !important;
        transition: all .15s ease;
      }}
      section[data-testid="stSidebar"] .stButton > button:hover {{
        background: rgba(255,255,255,0.05) !important;
        color: #fff !important;
        border-color: rgba(255,255,255,0.08) !important;
      }}
      section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, rgba(79,70,229,0.25), rgba(6,182,212,0.15)) !important;
        color: #fff !important;
        border: 1px solid rgba(79,70,229,0.45) !important;
        box-shadow: 0 4px 16px rgba(79,70,229,0.15);
      }}

      /* Sidebar status pill */
      .sb-status {{
        display:flex; align-items:center; gap:8px;
        padding: 8px 10px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        font-size: 0.78rem;
        margin-top: 6px;
      }}
      .dot {{
        width: 8px; height: 8px; border-radius: 50%;
        display: inline-block;
      }}
      .dot-ok   {{ background: #10b981; box-shadow: 0 0 8px #10b981; }}
      .dot-warn {{ background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }}
      .dot-err  {{ background: #ef4444; box-shadow: 0 0 8px #ef4444; }}

      /* =========== MAIN CONTENT =========== */
      .block-container {{
        max-width: 1480px !important;
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
      }}

      /* Page header */
      .page-head {{
        display: flex; align-items: flex-start; justify-content: space-between;
        gap: 1rem;
        padding: 0.6rem 0 1rem;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.4rem;
      }}
      .page-head-left h1 {{
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: var(--ink) !important;
        letter-spacing: -0.02em !important;
        margin: 0 0 0.25rem 0 !important;
        line-height: 1.15 !important;
      }}
      .page-head-left .sub {{
        font-size: 0.92rem;
        color: var(--ink-muted);
        font-weight: 400;
        max-width: 720px;
      }}
      .page-head-left .eyebrow {{
        display: inline-block;
        font-size: 0.68rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--primary);
        margin-bottom: 0.4rem;
      }}

      /* Cards */
      .card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.03);
      }}
      .card-hover:hover {{
        border-color: var(--primary) !important;
        box-shadow: 0 10px 24px rgba(79,70,229,0.08) !important;
        transform: translateY(-1px);
        transition: all .15s ease;
      }}
      .card-title {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--ink-muted);
        font-weight: 700;
        margin-bottom: 0.4rem;
      }}
      .card-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.02em;
      }}
      .card-delta {{
        font-size: 0.82rem;
        font-weight: 500;
        color: var(--ink-muted);
        margin-top: 0.2rem;
      }}

      /* Native metric tweak */
      [data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.75rem 0.95rem !important;
      }}
      [data-testid="stMetricLabel"] {{ font-size: 0.72rem !important; color: var(--ink-muted) !important; font-weight:600 !important; letter-spacing: 0.05em; text-transform: uppercase; }}
      [data-testid="stMetricValue"] {{ font-size: 1.35rem !important; font-weight: 700 !important; color: var(--ink) !important; }}

      /* Bordered container */
      [data-testid="stVerticalBlockBorderWrapper"] > div {{
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        background: var(--surface);
      }}

      /* Buttons (main) */
      .main .stButton > button {{
        border-radius: 10px !important;
        font-weight: 500 !important;
        border: 1px solid var(--border) !important;
        background: #fff !important;
        color: var(--ink) !important;
        transition: all .15s ease !important;
      }}
      .main .stButton > button:hover {{
        border-color: var(--primary) !important;
        color: var(--primary) !important;
        box-shadow: 0 4px 12px rgba(79,70,229,0.1) !important;
      }}
      .main .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(79,70,229,0.25) !important;
      }}
      .main .stButton > button[kind="primary"]:hover {{
        filter: brightness(1.08);
        transform: translateY(-1px);
      }}

      /* Inputs */
      .stTextInput input, .stTextArea textarea,
      .stSelectbox div[data-baseweb="select"] > div,
      .stNumberInput input, .stDateInput input, .stTimeInput input {{
        border-radius: 10px !important;
        border: 1px solid var(--border) !important;
        background: #fff !important;
      }}
      .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
      }}

      /* Tabs */
      .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid var(--border);
      }}
      .stTabs [data-baseweb="tab"] {{
        border-radius: 10px 10px 0 0 !important;
        font-weight: 500 !important;
        padding: 0.55rem 1rem !important;
        color: var(--ink-muted) !important;
      }}
      .stTabs [aria-selected="true"] {{
        color: var(--primary) !important;
        background: rgba(79,70,229,0.06) !important;
      }}

      /* Pills badge */
      .pill {{
        display: inline-flex; align-items: center; gap: 6px;
        font-size: 0.72rem; font-weight: 600;
        padding: 0.22rem 0.65rem; border-radius: 999px;
        letter-spacing: 0.02em;
      }}
      .pill-primary {{ background: rgba(79,70,229,0.08); color: var(--primary); border:1px solid rgba(79,70,229,0.2); }}
      .pill-accent  {{ background: rgba(6,182,212,0.08); color: var(--accent); border:1px solid rgba(6,182,212,0.2); }}
      .pill-ok      {{ background: #ecfdf5; color: #047857; border:1px solid #a7f3d0; }}
      .pill-warn    {{ background: #fffbeb; color: #b45309; border:1px solid #fde68a; }}
      .pill-err     {{ background: #fef2f2; color: #b91c1c; border:1px solid #fecaca; }}
      .pill-neutral {{ background: #f1f5f9; color: #334155; border:1px solid #e2e8f0; }}

      /* Hero */
      .hero {{
        background: linear-gradient(135deg, #0b1020 0%, #1e1b4b 45%, #312e81 100%);
        color: #fff;
        border-radius: 20px;
        padding: 2rem 2.2rem;
        position: relative; overflow: hidden;
        margin-bottom: 1.5rem;
      }}
      .hero::after {{
        content: "";
        position: absolute; right: -80px; top: -80px;
        width: 320px; height: 320px; border-radius: 50%;
        background: radial-gradient(circle, rgba(6,182,212,0.35) 0%, transparent 70%);
        filter: blur(10px);
      }}
      .hero h1 {{
        font-size: 1.9rem !important; font-weight: 800 !important;
        letter-spacing: -0.02em !important; margin: 0 0 0.35rem !important;
      }}
      .hero p {{
        color: #cbd5e1; max-width: 640px; margin: 0;
      }}
      .hero-chips {{
        display: flex; gap: 8px; margin-top: 1rem; flex-wrap: wrap;
      }}
      .hero-chip {{
        background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15);
        color: #f1f5f9; padding: 0.3rem 0.8rem; border-radius: 999px;
        font-size: 0.78rem; font-weight: 500;
        backdrop-filter: blur(6px);
      }}

      /* Mono / code style */
      .mono {{
        font-family: 'JetBrains Mono', ui-monospace, Menlo, monospace;
        font-size: 0.82rem; color: var(--ink);
      }}

      /* Divider soft */
      hr {{ border-color: var(--border) !important; margin: 0.9rem 0 !important; }}

      /* Hide default header */
      header[data-testid="stHeader"] {{ background: transparent; }}

      /* Expander */
      details > summary {{
        font-weight: 600 !important;
      }}

      /* Chat bubbles */
      .user-row {{ display:flex; justify-content:flex-end; margin: 0.5rem 0 0.2rem; }}
      .user-bubble {{
        background: linear-gradient(135deg, var(--primary), var(--primary-2));
        color: #fff;
        border-radius: 18px 18px 4px 18px;
        padding: 0.65rem 1rem;
        font-size: 0.92rem; line-height: 1.55; max-width: 72%;
        box-shadow: 0 4px 12px rgba(79,70,229,0.2);
      }}
      .ai-row {{
        display:flex; align-items:flex-start; gap: 10px;
        margin: 0.4rem 0 0.8rem;
      }}
      .ai-avatar {{
        width: 30px; height: 30px; border-radius: 9px;
        background: linear-gradient(135deg, #06b6d4, #4f46e5);
        color:#fff; display:flex; align-items:center; justify-content:center;
        font-size:13px; font-weight:700; flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(6,182,212,0.3);
      }}
      .ai-text {{ font-size: 0.93rem; color: var(--ink); line-height: 1.65; padding-top: 3px; }}

      /* Footer */
      .app-footer {{
        padding: 1.2rem 0 0.4rem;
        color: var(--ink-muted);
        font-size: 0.75rem;
        text-align: center;
        border-top: 1px solid var(--border);
        margin-top: 2rem;
      }}

      /* =========== LOADING OVERLAY =========== */
      @keyframes mip-spin {{
        0%   {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
      }}
      @keyframes mip-pulse {{
        0%, 100% {{ opacity: 1; }}
        50%       {{ opacity: 0.4; }}
      }}
      @keyframes mip-bar {{
        0%   {{ left: -40%; width: 40%; }}
        50%  {{ left: 20%;  width: 60%; }}
        100% {{ left: 110%; width: 40%; }}
      }}

      /* Full-section loader card */
      .mip-loader {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 1.1rem;
        padding: 3.5rem 2rem;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: var(--surface);
        box-shadow: 0 4px 24px rgba(15,23,42,0.05);
        min-height: 220px;
      }}
      .mip-loader-ring {{
        width: 52px; height: 52px;
        border: 4px solid rgba(79,70,229,0.12);
        border-top: 4px solid var(--primary);
        border-radius: 50%;
        animation: mip-spin 0.9s linear infinite;
      }}
      .mip-loader-label {{
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--ink-muted);
        letter-spacing: 0.01em;
        animation: mip-pulse 1.6s ease-in-out infinite;
      }}
      .mip-loader-sub {{
        font-size: 0.78rem;
        color: #94a3b8;
      }}

      /* Progress bar (top of section) */
      .mip-progress-bar {{
        position: relative;
        height: 3px;
        width: 100%;
        background: rgba(79,70,229,0.08);
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 1.2rem;
      }}
      .mip-progress-bar::after {{
        content: "";
        position: absolute;
        top: 0; height: 100%;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 999px;
        animation: mip-bar 1.4s ease-in-out infinite;
      }}

      /* Skeleton shimmer */
      @keyframes mip-shimmer {{
        0%   {{ background-position: -400px 0; }}
        100% {{ background-position: 400px 0; }}
      }}
      .mip-skeleton {{
        background: linear-gradient(90deg, #e2e8f0 25%, #f1f5f9 50%, #e2e8f0 75%);
        background-size: 800px 100%;
        animation: mip-shimmer 1.4s infinite;
        border-radius: 8px;
        height: 14px;
        width: 100%;
        margin: 6px 0;
      }}
      .mip-skeleton-wide  {{ width: 80%; }}
      .mip-skeleton-short {{ width: 45%; }}
    </style>
    """, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str = "", right_html: str = ""):
    """Renderiza un encabezado consistente para cada página."""
    st.markdown(f"""
    <div class="page-head">
      <div class="page-head-left">
        <div class="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <div class="sub">{subtitle}</div>
      </div>
      <div>{right_html}</div>
    </div>
    """, unsafe_allow_html=True)


def show_loader(label: str = "Cargando…", sub: str = "", progress_bar: bool = True):
    """Muestra un loader de sección grande con animación mientras Streamlit carga datos."""
    bar_html = '<div class="mip-progress-bar"></div>' if progress_bar else ""
    sub_html = f'<div class="mip-loader-sub">{sub}</div>' if sub else ""
    st.markdown(f"""
    {bar_html}
    <div class="mip-loader">
      <div class="mip-loader-ring"></div>
      <div class="mip-loader-label">{label}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


def metric_card(title: str, value: str, delta: str = "", color: str = "primary"):
    """Card tipo métrica con estilo Databricks."""
    pill = f'<span class="pill pill-{color}">{delta}</span>' if delta else ""
    st.markdown(f"""
    <div class="card card-hover">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="card-delta">{pill}</div>
    </div>
    """, unsafe_allow_html=True)
