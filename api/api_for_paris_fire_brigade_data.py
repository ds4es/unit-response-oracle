"""Launch an example API (under http://127.0.0.1:5000/api) to consume the 
predictive model to predict units response times


Execution
---------
    # Load your Python environment
    # Add to the PYTHONPATH variable the project root directory
    export PYTHONPATH=$PYTHONPATH:$(pwd)
     
    # Launch the script
    python api/api_for_paris_fire_brigade_data.py

"""

# Authors: Benjamin Berhault
#
# Email:   ds4es.mailbox@gmail.com

from flask import  Flask, request,  json
import pandas as pd
import swifter
import logging
from datetime import datetime

from src.config import *
from src.models import predict
from src.utils import utils
from src.models.predict import make_prediction

INTERVENTION_COLUMNS = [
    'intervention',
    'alert reason category',
    'alert reason',
    'intervention on public roads',
    'floor',
    'location of the event',
    'longitude intervention',
    'latitude intervention',
    'rescaled longitude intervention',
    'rescaled latitude intervention',
    'time day',
    'time week',
    'time year'
]

UNIT_COLUMNS = [
    'emergency vehicle selection',
    'emergency vehicle',
    'emergency vehicle type',
    'rescue center',
    'selection time',
    'delta status preceding selection-selection',
    'departed from its rescue center',
    'longitude before departure',
    'latitude before departure',
    'delta position gps previous departure-departure',
    'routing engine estimated distance',
    'routing engine estimated duration',
    'routing engine estimated distance from last observed GPS position',
    'routing engine estimated duration from last observed GPS position',
    'time elapsed between selection and last observed GPS position',
    'updated routing engine estimated duration',
    'departure center',
    'GPS tracks count',
    'estimated speed',
    'estimated duration from speed',
    'estimated time factor',
    'estimated duration from time',
    'intervention count',
    'rescaled longitude before departure',
    'rescaled latitude before departure'
]

INT_COLUMNS = [
    'emergency vehicle',
    'intervention',
    'alert reason category',
    'alert reason',
    'intervention on public roads',
    'floor',
    'location of the event',
    'emergency vehicle',
    'emergency vehicle type',
    'rescue center',
    'delta status preceding selection-selection',
    'departed from its rescue center',
    'delta position gps previous departure-departure',
    'departure center',
    'GPS tracks count',
    'intervention count'
]

linear_model = predict.load_linear_model(LINEAR_MODEL_FILE)
lgbm_model = predict.load_lgbm_model(LGBM_MODEL_WITH_CV_FILE)
    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api', methods = ['POST'])
def api():

    if request.headers['Content-Type'] == 'application/json':

        data = request.json
        
        intervention = pd.DataFrame(data['intervention'],columns=INTERVENTION_COLUMNS, index=['intervention'])

        units = pd.DataFrame(data['units'],columns=UNIT_COLUMNS)

        for i in INTERVENTION_COLUMNS:
            units[i] = intervention[i][0]

        for f in INT_COLUMNS:
            units[f] = utils.to_int_keys_pd(units[f])
            logging.info(f'categorical feature {f}: {len(set(units[f]))} values')

        units.set_index(ID, inplace=True)
        prediction = make_prediction(units, linear_model, lgbm_model)
        prediction.reset_index(level=0, inplace=True)
        prediction[ID] = prediction[ID].astype(int)

        print("Prediction asked for an intervention on the",datetime.utcfromtimestamp(int(min(list(units["selection time"])))).strftime('%Y-%m-%d %H:%M:%S'), "for ", str(len(units)),"units")
        
        return prediction.to_json(orient='records', lines=True) + '\n'

    else:
        return '415 Unsupported Media Type ;)'


if __name__ == '__main__':
    app.run(debug=True)