.PHONY:
	conda-environment
	virtual-env
	pip-install-requirements
	update-requirements

conda-environment:
	conda env create --name regression-methods-comparison --file=environments.yml

virtual-env:
	python -m venv .env

pip-install-requirements:
	python -m pip install -r requirements.txt

update-requirements:
	conda env export | grep -v "^prefix: " > environment.yml
	conda list -e > requirements.txt
