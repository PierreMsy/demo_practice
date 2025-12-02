import sys
from datetime import datetime
from pathlib import Path

# Add src/ to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

project = "Toy Binary Classification"
author = "Your Name"
year = datetime.now().year
copyright = f"{year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"

autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
