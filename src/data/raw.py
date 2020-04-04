from pathlib import Path
import pandas as pd

FILE_DIR = Path(__file__).parent.absolute()
ROOT_DIR = FILE_DIR.parent.parent
RAW_DATA_DIR = ROOT_DIR / 'data/raw'

print(FILE_DIR)
print(ROOT_DIR)

def load_data():
	# Data reading
	x_train = pd.read_csv(RAW_DATA_DIR / 'x_train.csv', sep=',')
	# x_train_additional_file = pd.read_csv(DATA / 'x_train_additional_file.csv', sep=',')
	y_train = pd.read_csv(RAW_DATA_DIR / 'y_train.csv', sep=',')
	x_test = pd.read_csv(RAW_DATA_DIR / 'x_test.csv', sep=',')
	# x_test_additional_file = pd.read_csv(DATA / 'x_test_additional_file.csv', sep=',')
	print("Données chargées!!")
	return x_train, y_train, x_test


