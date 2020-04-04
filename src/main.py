import data.raw
import models.basic_model

def main():
	x_train, y_train, x_test = data.raw.load_data()
	trained_model, polynomial_features = models.basic_model.train(x_train, y_train)
	prediction = models.basic_model.predict(trained_model, polynomial_features, x_test, y_train)
  
main()