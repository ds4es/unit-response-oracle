# Unit Response Oracle

The response time is one of the most important factors for emergency services because their ability to save lives and rescue people depends on it. A non-optimal choice of an emergency vehicle for a rescue request may lengthen the arrival time of the rescuers and impact the future of the victim. This choice is therefore highly critical for emergency services and directly rely on their ability to predict precisely the arrival time of the different units available.

## Objective and Thanks
This project aims to predict the response time of the appliances of an emergency service and is ONLY made possible through the code sharing from Wenqi Shu-Quartier-dit-Maire _(wshuquar - rank 2 on the leaderboard)_, Antoine Moulin _(amoulin)_, [Julien Jerphanion & Edwige Cyffers](https://gitlab.com/jjerphan/challenge-data-paris-fire-brigade) _(edwige & jjerphan)_, Wassim Bouaziz & Elliot Vincent _(elliot.vincent & wesbz)_, [Quentin Gallouedec](https://github.com/quenting44/predicting_response_times) _(Quenting44)_, [Laurent Deborde](https://github.com/ljmdeb/Pompiers) _(Ljmdeb)_, François Paupier _(popszer)_, Léo Andéol _(leoandeol)_.

Thanks to all of them very much for the work carried out and shared.

## Performance

With the current stage of the model and the Paris Fire Brigade data, we reach the following performances:

| Metric                                      | Score                     |
| ------------------------------------------- |:-------------------------:|
| Delta selection-presentation R² score       | 0.3519                    |
| RMSLE (Root mean squared logarithmic error) | 0.2409                    |
| Median error                                | 46.8585 seconds           |
| Mean error                                  | 79.8842 seconds           |

For instance we have only used the fantastic work of Wenqi Shu-Quartier-dit-Maire but we are eager to also exploit the work of the other participants quoted just above resealing wonderful ideas.

## Usage

### Prerequisites

Download the repo
```
git clone https://github.com/ds4es/unit-response-oracle
cd ./unit-response-oracle
```
Load your Python environment (eg. `conda activate my_env`).

Install dependencies
```bash
pip install -r requirements.txt
```

Add the root project path to your `PYTHONPATH` variable:
```
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Commands

* `python src/data/paris_fire_brigade_raw_data_to_processed.py`: to turn Paris Fire Brigade data challenge raw data into preprocessed data
* `python src/models/train_linear.py`: to train an intermediate linear model
* `python src/models/train_lgbm.py`: to train the global model
* `python src/models/predict.py`: to make a prediction
* `python src/utils/prediction_evaluation.py`: to evaluate a prediction *(the default execution will need a `raw/paris_fire_brigade_2018/y_test.zip` file)*
* `python src/train_predict_evaluate.py`: to train a model, predict and evaluate the result of the prediction *(the default execution will need a `raw/paris_fire_brigade_2018/y_test.zip` file)*

To remove assert and __debug__-dependent statements add the `-O` flag when running one of those scripts, e.g.:
```
python -O src/train_predict_evaluate.py
```
*NB: Take a look at the src/config.py constant variables*

To start the jupyter notebook in the dedicated folder
```
jupyter notebook --notebook-dir=notebooks
```

#### Test the consumption of the model through an API

In a first terminal, start the API:
```
python api/api_for_paris_fire_brigade_data.py
```

In a second terminal, launch a random query:
```
python api/make_me_a_query_for_paris_fire_brigade_2018_data.py
```

### Input data description (for the Paris Fire Brigade dataset under the `data/processed/paris_fire_brigade_2018` folder)

* **[ID]** `emergency vehicle selection`: identifier of the selection instance of an emergency vehicle for an intervention
* Intervention
  * `intervention`: identifier of the intervention
  * `alert reason category` (category): alert reason category
  * `alert reason` (category): alert reason
* Address
  * `intervention on public roads` (boolean): 1 when it concerns an intervention on public roads, 0 otherwise
  * `floor` (int): floor of the intervention
  * `location of the event` (category): qualifies the location of the emergency request, for example: entrance hall, boiler room, motorway, etc.
  * `longitude intervention`: (float): approximate longitude of the intervention address
  * `latitude intervention` (float): approximate latitude of the intervention address
* Emergency vehicle 
  * `emergency vehicle`: identifier of the emergency vehicle 
  * `emergency vehicle type` (category): type of the emergency vehicle
  * `rescue center` (category): identifier of the rescue center to which belong the vehicle (parking spot of the emergency vehicle)
* `selection time` (int): selection time of the emergency vehicle (seconds since the 1 January 1970)
* `delta status preceding selection-selection` (int): number of seconds before the vehicle was selected when its previous status was entered
* `departed from its rescue center` (boolean) : 1 when the vehicle departed from its rescue center (emergency vehicle parking spot), 0 otherwise
* `longitude before departure` (float): longitude of the position of the vehicle preceding his departure
* `latitude before departure` (float): latitude of the position of the vehicle preceding his departure
* `delta position gps previous departure-departure` (int): number of seconds before the selection of the vehicle where its GPS position was recorded (when not parked at its emergency center)
* `routing engine estimated distance` (float): distance (in meters) calculated by the routing engine route service
* `routing engine estimated duration` (float): transit delay (in seconds) calculated by the routing engine route service
* `routing engine estimated distance from last observed GPS position` (float): distance (in meters) calculated by the routing engine route service from last observed GPS position
* `routing engine estimated duration from last observed GPS position` (float): transit delay (in seconds) calculated by the routing engine route service from last observed GPS position
* `time elapsed between selection and last observed GPS position` (float): in seconds
* `updated routing engine estimated duration` (float): time elapsed (in seconds) between selection and last observed GPS position + routing engine estimated duration from last observed GPS position
* `time day` (float): <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\text{'selection&space;time'}\mod(3600*24)}{3600*24}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\text{'selection&space;time'}\mod(3600*24)}{3600*24}"/></a>
* `time week` (float): <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\text{'selection&space;time'}\mod(3600*24*7)}{3600*24*7}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\text{'selection&space;time'}\mod(3600*24*7)}{3600*24*7}"/></a>
* `time year` (float): <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\text{'selection&space;time'}\mod(3600*24*7*365)}{3600*24*7*365}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\text{'selection&space;time'}\mod(3600*24*7*365)}{3600*24*7*365}"/></a>
* `departure center` (int): Departure parking place
* `GPS tracks count` (int): Number of GPS traces observed
* `estimated speed` (float): Estimated speed (in meters/second) due to last observed position <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\text{'routing&space;engine&space;estimated&space;distance'}-\text{'routing&space;engine&space;estimated&space;distance&space;from&space;last&space;observed&space;GPS&space;position'}}{\text{'time&space;elapsed&space;between&space;selection&space;and&space;last&space;observed&space;GPS&space;position'}}&space;\text{meters/second}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\text{'routing&space;engine&space;estimated&space;distance'}-\text{'routing&space;engine&space;estimated&space;distance&space;from&space;last&space;observed&space;GPS&space;position'}}{\text{'time&space;elapsed&space;between&space;selection&space;and&space;last&space;observed&space;GPS&space;position'}}&space;\text{meters/second}" /></a>
* `estimated duration from speed` (float): Estimated `delta selection-presentation` (in seconds) based on `estimated speed`<a href="https://www.codecogs.com/eqnedit.php?latex=\text{'time&space;elapsed&space;between&space;selection&space;and&space;last&space;observed&space;GPS&space;position'}&plus;\frac{\text{routing&space;engine&space;estimated&space;distance&space;from&space;last&space;observed&space;GPS&space;position'}}{\text{'estimated&space;speed'}}&space;\text{seconds}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{'time&space;elapsed&space;between&space;selection&space;and&space;last&space;observed&space;GPS&space;position'}&plus;\frac{\text{routing&space;engine&space;estimated&space;distance&space;from&space;last&space;observed&space;GPS&space;position'}}{\text{'estimated&space;speed'}}&space;\text{seconds}" /></a>
* `estimated time factor` (float): <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\text{'routing&space;engine&space;estimated&space;duration'}-\text{routing&space;engine&space;estimated&space;distance&space;from&space;last&space;observed&space;GPS&space;position'}}{\text{'time&space;elapsed&space;between&space;selection&space;and&space;last&space;observed&space;GPS&space;position'}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\text{'routing&space;engine&space;estimated&space;duration'}-\text{routing&space;engine&space;estimated&space;distance&space;from&space;last&space;observed&space;GPS&space;position'}}{\text{'time&space;elapsed&space;between&space;selection&space;and&space;last&space;observed&space;GPS&space;position'}}" /></a>
* `estimated duration from time`(float): Estimated `delta selection-presentation` (in seconds) based on `estimated time factor` <a href="https://www.codecogs.com/eqnedit.php?latex=\text{'time&space;elapsed&space;between&space;selection&space;and&space;last&space;observed&space;GPS&space;position'}&plus;\frac{\text{routing&space;engine&space;estimated&space;duration&space;from&space;last&space;observed&space;GPS&space;position'}}{\text{'estimated&space;time&space;factor'}}&space;\text{seconds}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{'time&space;elapsed&space;between&space;selection&space;and&space;last&space;observed&space;GPS&space;position'}&plus;\frac{\text{routing&space;engine&space;estimated&space;duration&space;from&space;last&space;observed&space;GPS&space;position'}}{\text{'estimated&space;time&space;factor'}}&space;\text{seconds}" /></a>
* `intervention count` (int): Number of units sent for this intervention
* `rescaled longitude before departure`
* `rescaled longitude intervention`
* `rescaled latitude before departure`
* `rescaled latitude intervention`

Backup your work in the parent folder in a ZIP file without data, models...
```
zip -r ../unit-response-oracle_$(date +%Y%m%d_%H%M%S).zip . -x "*.csv" "*.pkl" "*__pycache__*" "*.git*" "*.ipynb*" "*mlruns*" ".tar"
```

## Track, Manage and Share Models with MLFlow and Neptune
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
```bash
source ~/.bashrc
```

### Usage

```bash
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
```bash
cd project_repo
python example.py
```
By now the tracking API should have write data into files into your local ./mlruns directory.

#### Monitor your experiments with MLFlow locally
To launch the MLFlow web interface execute
```bash
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

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML" async></script>
