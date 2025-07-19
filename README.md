# nasrine_seraji
Montanier's project for Bain

Assuptions:
- api and training run in two seprated service. Do no expect an api call to get the training. The client gets 
some inference from the model. The datascience team manages the training of the model with update on the code
or new data coming in
- For now model weights are shared through filesystem. In real it would be on the cloud (blob storage with 
versioning etc...)

Approach:
- Docker to run everything in it
- main payload is in property_friends:
    the model
    definitions for prediction, post-proc and training
- API is in API
    calls to models prediction

Tech stack choices:
- Docker
- uv
- pydantic
- fastapi

Coder experience:
- docker compose -f docker/docker-compose.dev.yml run --rm property_friends to interact with the code. code binded. Not sure it's very
friendly for vscode, but it works nicely with my nvim setup. In real scenario i would pay a lot more attention
to make it friendly for everyone.


How to:
- pre-commit : pip install --user pipx ;  pipx ensurepath ; pipx install pre-commit ; pre-commit install ; pre-commit run --all-files
- run tests locally:  docker compose -f docker/docker-compose.dev.yml run --rm  --build property_friends; cd property_friends ;  uv run pytest tests
- Train locally with script: docker compose -f docker/docker-compose.dev.yml run --rm --build property_friends; cd property_friends ; uv run scripts/train_and_serialize_model.py
- start API service: docker compose -f docker/docker-compose.api.yml up --build

## Release Instructions

### Releasing a New Version

To release a new version of the property_friends package:

1. **Update the version number**:
   - Edit `property_friends/property_friends/__init__.py`
   - Change the `__version__` variable to your new version (e.g., `__version__ = "0.2.0"`)
   - Rebuild the package: `uv build`

2. **Update the API dependency**:
   - Edit `api/pyproject.toml` 
   - Update the wheel path in `[tool.uv.sources]` section:
     ```toml
     property-friends = { path = "../property_friends/dist/property_friends-NEW_VERSION-py3-none-any.whl" }
     ```
   - Replace `NEW_VERSION` with the version number from step 1
   - Regenerate the uv lock file : `uv lock`

3. **Build and test**:
   - Build the package: `cd property_friends && uv build`
   - Test the API: `docker compose -f docker/docker-compose.api.yml up --build`


Log:
0h45m : explo notebook, basic tech choices
2h15m : Base of project: docker, payload project, test, CI
3h05m : Preprocessor, format and lint
4h00m : property_friends package implemented, mypy, refacto
4h15m : fix CI follow up refacto
4h45m : local train with script (ugly but needed for time constraint)

future work:
- publish the docker image to a container registry so that tests are faster.
- publish the `property_friends` python package: cleaner version handling and code
- switch to a paid plan on github to have branch protection etc...
- system test for the api. Check that the API is answering correctly for basic queries.
- proper versioning of the model, e.g. mlwflow or wab
