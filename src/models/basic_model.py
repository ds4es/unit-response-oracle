import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

import sys
sys.path.append("..")

from data import raw

def train(x_train, y_train):
	x_train_transit = x_train[['OSRM estimated distance','intervention on public roads','floor']]
	y_train_transit = y_train[['delta departure-presentation']]

	# Create a predictive model for the 'delta departure-presentation'
	# based on 'OSRM estimated distance', 'intervention on public roads' and 'floor'
	polynomial_features= PolynomialFeatures(degree=3)
	x_train_transit_poly = polynomial_features.fit_transform(x_train_transit)
	model = LinearRegression()
	model.fit(x_train_transit_poly, y_train_transit)

	print("Modèle entrainé!!")

	return model, polynomial_features

def predict(model, polynomial_features, x_test, y_train):
	x_test_transit = x_test[['OSRM estimated distance','intervention on public roads','floor']]

	# Prediction of the 'delta selection-presentation'
	x_test_transit_poly = polynomial_features.fit_transform(x_test_transit)
	y_selection_presentation_predicted = y_train['delta selection-departure'].median() + model.predict(x_test_transit_poly)
	
	print('Aperçu de prédiction :')
	print(y_selection_presentation_predicted[:5])

	return y_selection_presentation_predicted