function params {
    $PY_SECONDARY -c \
        "import yaml; y = yaml.safe_load(open('params.yaml'))$1; print(y)"
}