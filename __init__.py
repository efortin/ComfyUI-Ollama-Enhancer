try:
    # ComfyUI context (when loaded as a package inside custom_nodes)
    from .nodes.ollama_enhancer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except ImportError:
    # For development (launch as root package)
    from nodes.ollama_enhancer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
