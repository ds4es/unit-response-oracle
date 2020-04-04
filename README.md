# Data Science Project Template

## Folder structure
- `model`: Folder for storing binary (json or other format) file for local use.
- `data`: Folder for storing subset data for experiments. It includes both raw data and processed data for temporary use.
- `doc`:
- `model`: Folder for storing binary (json or other format) file for local use.
- `notebook`: Storing all notebooks includeing EDA and modeling stage.
- `reference`:
- `report`:
- `src`: Stores source code (python, R etc) which serves multiple scenarios. During data exploration and model training, we have to transform data for particular purpose. We have to use same code to transfer data during online prediction as well. So it better separates code from notebook such that it serves different purpose.


```
.
├── .env               <- Store your secrets and config variables in a special file
├── .gitignore         <- Avoids uploading data, credentials, 
|                         outputs, system files etc
├── env                <- Will contain the Python executable files and installed libraries for your virtualenv environment
├── requirements.txt   <- Install the environment dependencies with: `pip install -r requirements.txt`
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── doc                <- Space for Sphinx documentation
├── models              <- Trained and serialized models, model predictions, or model summaries
├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
│   |                     the creator's initials, and a short `-` delimited description, e.g.
│   |                     `1.0-jqp-initial-data-exploration`.
│   ├── eda            <- Placholder for describing exploratory data analysis
│   ├── evaluation     <- Placholder for describing model evaluation
│   ├── modeling       <- Placholder for describing how the model is built
│   └── poc            <- Describing Proof-of-Concept
├── references          <- Data dictionaries, manuals, etc.
├── reports             <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   ├── build_features.py
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
└── tox.ini               <- Automate testing, cf. https://tox.readthedocs.io/en/latest/.
```
## Prerequisites
### Linux
For local developement we advice the use of [Anaconda 3.x](https://www.anaconda.com/distribution/) (or [Miniconda 3.x](https://docs.conda.io/en/latest/miniconda.html)) installed under your `/Home/username` directory keeping you far away from unintentional troubles messing up your Python OS depedencies, and `virtualenv` to encapsulate the Python package dependencies relying to this project in its own directory.

Anaconda install
```
sudo dnf install wget # replace the dnf keyword by the one suiting your Linux distribution
cd ~/Downloads 
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Install `virtualenv`
```
conda update conda
conda update python
conda install pip
# Check that pip is relying to the right Anaconda installation
which pip 
pip install virtualenv
```

## Initial setup
```
git clone https://github.com/benjamin-berhault/esuro
cd ./esuro
```
Create en isolated Python environment in ./env
```
virtualenv env
```
Enable the virtual python environment
```
source ./env/bin/activate
```
Setup a local environment
```
pip install -r requirements.txt
```
Download the raw data
```
make raw-data
```

## Track, manage and share models with MLFlow and Neptune
`neptune-mflow` integrates `mlflow` with `Neptune` to let you get the best of both worlds: tracking and reproducibility of `mlflow` with organization and collaboration of `Neptune`.

#### Install MLFlow and Neptune 
__Reference:__ [neptune-mlflow GitHub repo](https://github.com/neptune-ai/neptune-mlflow)

_`mlflow` is a dependency of `neptune-mlflow` so do not worry if you need or not `mlflow`._

```
pip install neptune-mlflow
```

#### Linux configuration
Login to Neptune retrieve your API token and declare it through `~/.bashrc`
```bash
vi ~/.bashrc
```
Add this line with your Neptune API token  
```bash
export NEPTUNE_API_TOKEN="your_long_api_token"
```

Reload `~/.bashrc`
```
source ~/.bashrc
```

### Usage

```
mkdir project_repo
cd project_repo
vi example.py
```

Add MLFlow functions in your code to log metrics, parameters and artifacts

```python
# example.py
import os
from mlflow import log_metric, log_param, log_artifact

if __name__ == "__main__":
    # Log a parameter (key-value pair)
    log_param("param1", 5)

    # Log a metric; metrics can be updated throughout the run
    log_metric("foo", 1)
    log_metric("foo", 2)
    log_metric("foo", 3)

    # Log an artifact (output file)
    with open("output.txt", "w") as f:
        f.write("Hello world!")
    log_artifact("output.txt")
```
Run your code
```
cd project_repo
python example.py
```
By now the tracking API should have write data into files into your local ./mlruns directory.

#### Monitor your experiments with MLFlow locally
To launch the MLFlow web interface execute
```
cd project_repo
mlflow ui
```
and view it at http://localhost:5000.

#### Share your experiments on Neptune
Retrieve the related `PROJECT_NAME` on Neptune you want to collaborate or create a new one: https://ui.neptune.ai

Publish and share your experiments on Neptune

```bash
cd project_repo
neptune mlflow --project USER_NAME/PROJECT_NAME
```

## References
* [Complete Data Science Project Template with Mlflow for Non-Dummies](https://towardsdatascience.com/complete-data-science-project-template-with-mlflow-for-non-dummies-d082165559eb)
* [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
* [Manage your Data Science project structure in early stage](https://towardsdatascience.com/manage-your-data-science-project-structure-in-early-stage-95f91d4d0600)
* [GitHub repo of an example data science project using Mlflow](https://gitlab.com/jan-teichmann/ml-flow-ds-project)
* [A logical, reasonably standardized, but flexible project structure for doing and sharing data science work](https://drivendata.github.io/cookiecutter-data-science/#data-is-immutable) & [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/)
* [How to Structure a Python-Based Data Science Project](https://medium.com/swlh/how-to-structure-a-python-based-data-science-project-a-short-tutorial-for-beginners-7e00bff14f56)