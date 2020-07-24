function params {
    $PY_CONDA -c \
        "import yaml; y = yaml.safe_load(open('params.yaml'))$1; print(y)"
}