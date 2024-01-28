PYTHON_VERSION = 3.10.13
PYSCRIPT_DIR = src
CLI_SCRIPT = app.py

.PHONY: install_pyenv
install_pyenv: # install python version maneger tool
	sudo apt update
	sudo apt upgrade
	sudo apt install \
		libssl-dev libffi-dev libncurses5-dev zlib1g zlib1g-dev \
		libreadline-dev libbz2-dev libsqlite3-dev make gcc
	curl https://pyenv.run | bash
	echo '' >> ~/.bashrc
	echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
	echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
	echo 'eval "$(pyenv init --path)"' >> ~/.bashrc

.PHONY: install_poetry
install_poetry: # install python packages version and it's conflict maneger tool
	curl -sSL https://install.python-poetry.org | python3 -
	poetry config virtualenvs.in-project true

.PHONY: install_python
install_python:
	pyenv install ${PYTHON_VERSION}

.PHONY: build_python
build_python:
	pyenv local ${PYTHON_VERSION}
	python -m pip install --upgrade pip
	poetry install

.PHONY: export_requirements
export_requirements:
	poetry export -f requirements.txt --output requirements.txt

.PHONY: streamlit
streamlit:
	poetry run streamlit run ${CLI_SCRIPT}

.PHONY: format
format:
	poetry run black ${CLI_SCRIPT}
	poetry run black ${PYSCRIPT_DIR}
	poetry run isort ${CLI_SCRIPT}
	poetry run isort ${PYSCRIPT_DIR}