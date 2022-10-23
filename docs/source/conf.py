# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#


import os
import inspect


def abspath(rel):
    """
    Take paths relative to the current file and
    convert them to absolute paths.
    Parameters
    ------------
    rel : str
      Relative path, IE '../stuff'
    Returns
    -------------
    abspath : str
      Absolute path, IE '/home/user/stuff'
    """

    # current working directory
    cwd = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))

    path = os.path.join(cwd, rel)
    print(path)

    return os.path.abspath(path)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'welleng'
copyright = '2022, Jonny Corcutt'
author = 'Jonny Corcutt'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon'
]

myst_all_links_external = True
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# The master toctree document.
master_doc = 'index'

# grab from trimesh without installing
exec(open(abspath('../../welleng/version.py')).read())
version = __version__
# The full version, including alpha/beta/rc tags.
release = __version__

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# options for rtd-theme
html_theme_options = {
    'analytics_id': 'UA-186225449-3',
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    # toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# custom css
html_css_files = ['custom.css']

html_context = {
    "display_github": True,
    "github_user": "jonnymaserati",
    "github_repo": "welleng",
    "github_version": "main",
    "conf_py_path": "/docs/"
}

# Output file base name for HTML help builder.
htmlhelp_basename = 'wellengdoc'

# -- Extensions configuration ----------------------------------
autodoc_default_options = {
    'autosummary': True,
    'special-members': '__init__',
}
