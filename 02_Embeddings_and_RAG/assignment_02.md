# 

## Task 0: Set up the virtualenv and environment

- create a `.env` file that has a key `OPENAI_API_KEY` 
- `uv sync` to create a virtualenv populated with modules in `pyproject.toml`
- activate the virtualenv: `source .venv/bin/activate`
- add the virtualenv as a kerenel for jupyter to use: `python -m ipykernel install --user --name=.venv`
- start jupyter: `jupyter-lab Pythonic_RAG_Assignment.ipynb`

## Task 1: Imports and Utilities

- Add an import for PyPDF2
- Add PyPDF2 to `pyprojects.toml`
- `uv sync` to pull the module into the virtualenv

## Task 2: Documents
## Task 3: Embeddings and Vectors
## Task 4: Prompts
## Task 5: Retrieval Augmented Generation


# Question 1
> The default embedding dimension of text-embedding-3-small is 1536, as noted above.
>
> Is there any way to modify this dimension?

There is a `dimensions` parameter at construction-time.

> What technique does OpenAI use to achieve this?

# Question 2

> What are the benefits of using an async approach to collecting our embeddings?


# Question 3

> When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?

The LLM constructor has a `temperature` argument that varies from 0 to 1 and controls the degree of creativity in the answer. Setting to 0 gives the least variability. Setting to 1 gives the most freedom in creating an answer.

# Question 4

> What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?

> What is that strategy called?

