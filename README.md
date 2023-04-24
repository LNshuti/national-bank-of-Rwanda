# National Bank of Rwanda
Applied AI Embeddings with Python using National Economic Reports

## Tools Used 

* Python 

* Openai

* Pinecone

* Langchain

## Clone this repo to your machine

```bash 
git clone https://github.com/LNshuti/national-bank-of-Rwanda.git

cd national-bank-of-Rwanda/
```

## Install requirements 

### Conda
```bash 
conda env create --file=environment.yaml
conda env config vars set OPENAI_API_KEY=YOUR_API_KEY
```


## Import libraries

```python   
import os
import faiss
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI, ConversationChain
import time

# Load OpenAI API key from environment
openai_api_key = os.environ.get('OPENAI_API_KEY')
```


## Load API Keys 
    
```python
import os 
# Load openai api key 
openai_api_key = os.environ.get('OPENAI_API_KEY')

# Load Pinecone API key 
pinecone_api_key = os.environ.get('PINECONE_API_KEY')

# Load pinecode api env 
pinecone_api_env = os.environ.get('PINECONE_API_ENV')
```

# Load PDF document

```{python}
pdf_loader = UnstructuredPDFLoader("data/Annual_Report_2021_22_Web_English_Versio.pdf")
pdf_text = pdf_loader.load()
```

# Initialize embeddings and language model
```{python}
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
conversation = ConversationChain(llm=llm, verbose=True)
```

```{r}
# Get embeddings for PDF document
start_time = time.time()
pdf_emb = embeddings.embed(pdf_text)
end_time = time.time()
print(f"Embedding generation time: {end_time - start_time} seconds")

# Build Faiss index with HNSW algorithm
index = faiss.IndexHNSWFlat(pdf_emb.shape[1], 32)
# Add PDF embeddings to index
index.add(pdf_emb)
# Train the index
index.train(pdf_emb)
```
# Search the index for some questions

```{python}
questions = [
  "What are the key challenges that Rwanda faces this year?",
  "Summarize key metrics for Rwanda including the amount of reserves, federal direct investment, inflation, and average food prices"
]

for question in questions:
  start_time = time.time()
  # Generate response using OpenAI
  response = conversation.generate_response(question).text
  # Print response
  print(f"Question: {question}")
  print(f"Response: {response}")
  # Get embeddings for response
  response_emb = embeddings.embed(response)
  # Search index for similar embeddings
  D, I = index.search(response_emb, 5)
  end_time = time.time()
  # Print search results
  print(f"Search results: {I}")
  print(f"Search time: {end_time - start_time} seconds\n")
```
# Weaviate

```{python}
import weaviate
from weaviate.embedded import EmbeddedOptions

client = weaviate.Client(
  embedded_options=EmbeddedOptions()
)

data_obj = {
  "name": "Chardonnay",
  "description": "Goes with fish"
}

client.data_object.create(data_obj, "Wine")
```
