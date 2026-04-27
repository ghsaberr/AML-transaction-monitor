# src/api/routers/__init__.py
"""
API routers for AML service.

Routers are imported directly in main.py to support optional dependencies.
This structure avoids circular imports and allows graceful handling of 
optional LLM-based features that may not be installed.
"""