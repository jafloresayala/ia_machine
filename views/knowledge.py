"""Knowledge Base — catálogo curado de parámetros de máquinas SMT."""
import streamlit as st

from theme import page_header, COLORS
from knowledge_base import PARAMETER_KNOWLEDGE, guess_category, get_category_info
from machine_registry import discover_machines


CATEGORY_LABELS = {
    "paste_printer":  "🖨️ Paste Printer",
    "pick_and_place": "🤖 Pick & Place",
    "reflow_oven":    "🔥 Reflow Oven",
    "spi":            "🔍 SPI",
    "aoi":            "👁️ AOI",
    "general":        "⚙️ General",
}


def render():
    page_header(
        eyebrow="Library · Knowledge",
        title="Knowledge Base",
        subtitle="Catálogo curado de parámetros típicos SMT — qué significan, rangos esperados y alertas.",
    )

    categories = list(PARAMETER_KNOWLEDGE.keys())

    # Buscador opcional: si el usuario eligió una máquina antes, sugerir esa categoría
    sel_machine = st.session_state.get("selected_machine_name") or st.session_state.get("analytics_machine")
    default_cat = guess_category(sel_machine) if sel_machine else "paste_printer"
    if default_cat not in categories:
        default_cat = categories[0]

    tabs = st.tabs([CATEGORY_LABELS.get(c, c) for c in categories])

    for tab, cat_key in zip(tabs, categories):
        with tab:
            info = get_category_info(cat_key)
            st.markdown(f"**{info['description']}**")

            # Sugerir máquinas que coinciden
            df = st.session_state.get("machines_df")
            if df is not None and not df.empty and cat_key != "general":
                kws = info["machine_keywords"]
                mask = df["machine_name"].str.lower().apply(lambda n: any(kw in n for kw in kws))
                matched = df[mask]
                if not matched.empty:
                    st.caption(f"🔗 {len(matched)} máquinas en tu planta coinciden con esta categoría")
                    with st.expander("Ver máquinas"):
                        st.dataframe(
                            matched[["line_name", "machine_name"]].reset_index(drop=True),
                            use_container_width=True, hide_index=True, height=200,
                        )

            st.markdown("---")
            params = info.get("parameters", {})
            cols = st.columns(2)
            for i, (pname, pdata) in enumerate(params.items()):
                with cols[i % 2]:
                    with st.container(border=True):
                        st.markdown(f"#### 🔹 {pname}")
                        st.markdown(
                            f"<span class='pill pill-neutral'>Unidad: {pdata.get('uom','—')}</span>&nbsp;"
                            f"<span class='pill pill-primary'>Rango típico: {pdata.get('typical_range','—')}</span>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(f"**¿Por qué importa?**")
                        st.markdown(pdata.get("why_it_matters", "—"))
                        st.markdown(f"**🚨 Alertas:**")
                        st.markdown(f"<div style='background:#fff7ed;border-left:3px solid #f59e0b;padding:0.5rem 0.75rem;border-radius:6px;color:#92400e;'>{pdata.get('alerts','—')}</div>",
                                    unsafe_allow_html=True)
