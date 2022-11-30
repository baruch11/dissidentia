function activate_venv() {
    poetry install &&
    poetry shell &&
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
}

activate_venv 

