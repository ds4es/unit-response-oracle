Initiliaze a Sphinx documentation and install a custom theme
```bash
source ./env/bin/activate
# Install a custom theme and an utility to easier sphinx documentation preview 
pip install sphinx-rtd-theme sphinx-serve
cd ./doc
# Create the Sphinx project skeleton
sphinx-quickstart
vi conf.py
```
Edit `conf.py` and replace 

```python
extensions = ['defaulttheme']
```
by

```python
import sphinx_rtd_theme

extensions = [
    "sphinx_rtd_theme",
]

html_theme = "sphinx_rtd_theme"
```

Rebuild your documentation
```
make html
```
Serve Sphinx locally
```
sphinx-serve
```