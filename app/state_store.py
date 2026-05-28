# Holds a reference to the original FastAPI app.state
# Set once at startup, read anywhere — avoids Gradio wrapper confusion

_app_state = None

def set_state(state):
    global _app_state
    _app_state = state

def get_state():
    if _app_state is None:
        raise RuntimeError("App state not initialised yet.")
    return _app_state