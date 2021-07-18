# ML FastApi STreamlit

ML prediction

FastAPI [endpoint](http://localhost:8000/docs)

Streamlit [frontend](http://localhost:8501)

```bash
cd src && mv question_intent_recognition_model.h5 q11_serving_bert_model/fastapi

cd src/q11_serving_bert_model

sudo docker-compose build # sudo docker images
sudo docker-compose up    # sudo docker ps -a
sudo docker-compose logs  # sudo docker system prune
```

## Deployment options

Azure > CLI (or use Portal)

* create resource group: `az group create --name qaResourceGroup --location westeurope`
* create container registry: `az acr create --resource-group qaResourceGroup --name questionRecog --sku Basic`
* login to container registry: `az acr login --name questionRecog`
* change image property: `questionrecog.azurecr.io/fastapi:latest`, `questionrecog.azurecr.io/streamlit:latest`
* build and run locally: `docker-compose up --build -d` (check `docker images`, `docker ps`)
* stop app and remove containers: `docker-compose down`
* push the images to the cotainer registry: `docker-compose push`
* check images in remote registry: `az acr repository show --name questionRecog --repository fastapi`
* create azure context: `docker login azure && docker context create aci qrecogacicontext`
* confirm context creation: `docker context ls`, use context: `docker context use qrecogacicontext`
* start the app in Azure Conatainer Instances: `docker compose up` (close: `docker compose down`)
* Minimum Price: €32/mo  (vCPU 1 / Storage 1GB)
* More plans: (Price: €288+)

Heroku with [Dockhero](https://elements.heroku.com/addons/dockhero):

* rename `docker-compose.yml` to `dockhero-compose.yml`
* create an app named `questionrecog`
* enable the Dockhero plugin: `heroku plugins:install dockhero`
* deploy the app: `heroku dh:compose up -d --app questionrecog`
* browse app URL: `heroku dh:open --app questionrecog`
* browse the GUI at `http://dockhero-questionrecog.dockhero.io:8501`
* browse the API at `http://dockhero-questionrecog.dockhero.io:8000/docs`
* browse app logs: `heroku logs -p dockhero --app questionrecog`
* Minimum Price: Hobby $7/mo  (Memory 512MB / Storage 4GB)
* Recommended price: Standard $25/mo (Memory 1GB / Storage 8GB)
* More plans: (Price: $99+)