"""Hugging Face Space entrypoint --- imports the Gradio demo from
``gradio_app.py`` and launches it.

HF Spaces reads ``app_file: app.py`` from the README YAML frontmatter
and runs ``python app.py`` at boot. Keeping this file tiny means the
entrypoint contract is stable; UI changes happen in ``gradio_app.py``.
"""

from gradio_app import demo

# Theme on launch() (Gradio 6.0 deprecation: theme moved off Blocks()).
# Lazy import keeps app.py importable without gradio when a tool only
# wants to introspect the module.
try:
    import gradio as gr
    _theme = gr.themes.Soft()
except Exception:
    _theme = None


if __name__ == "__main__":
    # HF Spaces runs this. server_name=0.0.0.0 is required so the
    # outer Node proxy can reach the Python worker. The launch options
    # match what gradio_app.py does for local development; HF ignores
    # any port we set and proxies its own.
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=_theme,
    )
