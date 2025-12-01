# app_dashboard.py
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# App multipágina
app = dash.Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Proyecto de Título – Segmentación de Clientes"
)

# Layout general con contenedor glass
app.layout = html.Div(
    className="app-background",
    children=[
        html.Div(
            className="app-shell",
            children=[
                # Barra superior
                html.Div(
                    className="app-header",
                    children=[
                        html.Div(
                            [
                                html.Span("PT", className="logo-badge"),
                                html.Span(
                                    "Proyecto de Título – Segmentación de Clientes y Patrones de Compra",
                                    className="app-title"
                                ),
                            ],
                            className="app-header-left",
                        ),
                        html.Div(
                            [
                                dcc.Link("Inicio", href="/", className="nav-link"),
                                dcc.Link("K-Means", href="/kmeans", className="nav-link"),
                                dcc.Link("Apriori", href="/apriori", className="nav-link"),
                                dcc.Link("Cruce", href="/cruce", className="nav-link"),
                            ],
                            className="app-header-right",
                        )
                    ],
                ),

                # Contenido de páginas
                html.Div(dash.page_container, className="page-container"),
            ],
        )
    ],
)

if __name__ == "__main__":
    app.run(debug=True)
