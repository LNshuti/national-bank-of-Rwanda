import os
import PyPDF2
import pinecone
from sentence_transformers import SentenceTransformer

# Load openai api key 
openai_api_key = os.environ.get('OPENAI_API_KEY')
# Load Pinecone API key 
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
# Load pinecode api env 
pinecone_api_env = os.environ.get('PINECONE_API_ENV')


# Extract text from the PDF
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text

def get_all_pdf_files_in_directory(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

# Generate vector embeddings
model = SentenceTransformer('paraphrase-distilroberta-base-v1')

# Initialize Pinecone
pinecone.init(api_key=pinecone_api_key)

# Create Pinecone index
index_name = "pdf_embeddings_bnr"
pinecone.create_index(name=index_name, dimension=768)

data_directory = "/data"
pdf_files = get_all_pdf_files_in_directory(data_directory)

# Process each PDF file
for pdf_path in pdf_files:
    pdf_text = extract_text_from_pdf(pdf_path)
    embeddings = model.encode([pdf_text], convert_to_tensor=True)

    # Persist the embeddings on Pinecone
    pdf_id = os.path.basename(pdf_path).split('.')[0]
    pinecone.upsert(index_name=index_name, items={pdf_id: embeddings[0].tolist()})

# Clean up resources
pinecone.deinit()
pinecone.deinitialize_index(index_name=index_name)
