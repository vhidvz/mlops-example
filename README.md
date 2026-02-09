# MLOps Example

Python Version Management

```sh
# https://github.com/pyenv/pyenv
curl -fsSL https://pyenv.run | bash
pyenv install 3.12 && pyenv global 3.12

# ~/.zshrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init - zsh)"' >> ~/.zshrc

# ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
```

Environment Preparation

```sh
cp .env.example .env

python -m venv .venv
source .venv/bin/activate

pip install --no-cache-dir --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-3.1.6/constraints-3.12.txt" -r requirements.txt
```

## lakeFS

Docker QuickStart

```sh
docker run --rm -d -p 8000:8000 treeverse/lakefs:1.74.4 run --quickstart
```

## MLflow

```sh
docker compose --env-file docker/config/mlflow.env -f docker/docker-compose.mlf.yml up -d
```
