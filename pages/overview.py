"""Overview / narrative landing page."""
from __future__ import annotations

import dash
from dash import dcc, html

from core import config


dash.register_page(__name__, path="/", name="Overview")

intro_md = f"""
### Project Goals

1. **Understand** Chicago's air-quality sensor coverage using open data (`{config.DATASET_ID}`).
2. **Explain** modeling choices (interpolation, network dynamics, forecasting).
3. **Share** interactive visuals that decision makers can explore without the notebook.

The site is organized into dedicated pages:
- **Data Explorer**: Station uptime, coverage stats, and exploratory metrics.
- **Network Analysis**: Graph-based views of spatial relationships and temporal dynamics.
- **Interpolation Studio**: Compare IDW vs. Kriging surfaces and prep for future STGNN work.

Later phases will add STGNN fine-tuning and socio-economic comparisons.
"""

layout = html.Div(
    [
        dcc.Markdown(intro_md, className="prose"),
        html.H3("Workflow Highlights"),
        html.Ul(
            [
                html.Li("Paginated Socrata ingestion — no more 5k row cap."),
                html.Li("Modular Python packages for data intake, network analysis, and interpolation."),
                html.Li("Dash multi-page UI ready for deployment."),
            ]
        ),
    ],
    className="page overview-page",
)
