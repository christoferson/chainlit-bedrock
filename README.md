# chainlit-bedrock
Chainlit Bedrock


## Installation
pip install chainlit

### Copilot

https://aws.github.io/copilot-cli/docs/getting-started/install/

```
sudo curl -Lo /usr/local/bin/copilot https://github.com/aws/copilot-cli/releases/latest/download/copilot-linux && sudo chmod +x /usr/local/bin/copilot
```

### Run Locally

chainlit run app.py

### Deploy

```
export AWS_REGION=us-east-1
copilot app init bedrockchat-app
copilot deploy --name bedrockchat --env dev
```

## Teardown

copilot app delete


## Links

- [chainlit-github](https://github.com/Chainlit/chainlit)
- [langchain-github](https://github.com/langchain-ai/langchain)
- [chainlit-docs](https://docs.chainlit.io/get-started/overview)
- https://www.kaggle.com/
- https://archive.ics.uci.edu/
- https://en.wikipedia.org/wiki/List_of_datasets_for_machine-learning_research
- https://registry.opendata.aws/