# nasrine_seraji
Montanier's project for Bain

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

future work:
- publish the docker image to a container registry so that tests are faster.
- switch to a paid plan on github to have branch protection etc...

How to:
- pre-commit : pip install --user pipx ;  pipx ensurepath ; pipx install pre-commit ; pre-commit install ; pre-commit run --all-files
- run tests locally:  docker compose -f docker/docker-compose.dev.yml run --rm property_friends; cd property_friends ; uv sync --extra dev ;  uv run pytest tests


Log:
0h45m : explo notebook, basic tech choices
2h15m : Base of project: docker, payload project, test, CI
3h05m : Preprocessor, format and lint
