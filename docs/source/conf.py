# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "ommx-fixstars-amplify-adapter"
copyright = "2024, Jij Inc."
author = "Jij Inc."

version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_fontawesome",
    "autoapi.extension",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_show_sourcelink = False

# -- AutoAPI settings --------------------------------------------------------
# https://sphinx-autoapi.readthedocs.io/en/latest/reference/config.html#event-autoapi-skip-member

autoapi_dirs = ["../../ommx_fixstars_amplify_adapter"]
autoapi_member_order = "groupwise"
