# National Bank of Rwanda
Applied AI Embeddings with Python using National Economic Reports


## Install requirements 

```bash
pip install -r requirements.txt
```

## Import libraries

```python   
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import CharacterTextSplitter 
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate
from langchain import OpenAI, ConversationChain

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

## Load Latest Annual Report
```python
bnr_report_reader = UnstructuredPDFLoader("data/Annual_Report_2021_22_Web_English_Versio.pdf")
bnr_report_reader_data = bnr_report_reader.load()

```

