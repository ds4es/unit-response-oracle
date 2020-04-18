# Emergency services unit response oracle

The response time is one of the most important factors for emergency services because their ability to save lives and rescue people depends on it. A non-optimal choice of an emergency vehicle for a rescue request may lengthen the arrival time of the rescuers and impact the future of the victim. This choice is therefore highly critical for emergency services and directly rely on their ability to predict precisely the arrival time of the different units available.


This project aims to predict the response time of the appliances of an emergency service and will only be made possible through the code sharing from Wenqi Shu-Quartier-dit-Maire (wshuquar - rank 1 on the leaderboard), Antoine Moulin (amoulin 0.3213), [Julien Jerphanion & Edwige Cyffers](https://gitlab.com/jjerphan/challenge-data-paris-fire-brigade) (edwige & jjerphan), Wassim Bouaziz & Elliot Vincent (elliot.vincent & wesbz), [Quentin Gallouedec](https://github.com/quenting44/predicting_response_times) (Quenting44), [Laurent Deborde](https://github.com/ljmdeb/Pompiers) (Ljmdeb), François Paupier (popszer), Léo Andéol (leoandeol).

Thanks to all of them very much for the work carried out and shared.

For details on the project structure please refer to: https://ds4es.org/docs/ds_project_template.html

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
git clone https://github.com/ds4es/esuro
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