# Makefile with commands like `make data` or `make train`
# For an example : https://github.com/mengdong/python-ml-structure/blob/master/Makefile

# Normally make runs every command in a recipe in a different subshell. However, setting 
# .ONESHELL: will run all the commands in a recipe in the same subshell, allowing you to 
# activate a virtualenv and then run commands inside it
.ONESHELL:

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
# BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')
PROFILE = default
PROJECT_NAME = emergency_service_unit_response_oracle
PYTHON_INTERPRETER = python
BIN=env/bin/

#################################################################################
# COMMANDS                                                                      #
#################################################################################

help: ## Shows this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


create_env: ## Set up a python interpreter environment with virtualenv
	virtualenv env;
	. $(BIN)activate;
	$(BIN)pip install -r requirements.txt;

main: ## Execute the Main script
	$(BIN)python ./src/main.py

notebook: ## Launch Jupyter Notebook from the dedicated folder
	cd $(PROJECT_DIR)/notebooks/eda
	$(PROJECT_DIR)/$(BIN)jupyter notebook