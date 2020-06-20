# autoICD startup

<div style="text-align: center; padding: 10px; margin: 10px; border-outline: 1px solid darkgray; border-radius: 3px; background-color: #ff3333; color: white;">
  <strong>WARNING</strong> Never, ever open source this repository!
</div>

<hr/>

To get access to the MIMIC data, please authenticate first:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="$(PWD)/autoicd-gcp-credentials.json"
```

To run app locally, install [poetry](https://python-poetry.org/docs/#installation) and [DVC](https://dvc.org). Then:
```bash
dvc pull # <- this only needs to be done once
cd src
poetry install # <- this also only needs to be done once
poetry shell
make app
``` 

The rest of the pipeline can be run through make rules:
```
app                            run streamlit app locally
format_data                    data preprocessing step
train                          model training
```