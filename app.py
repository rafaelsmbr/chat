import os
from flask import Flask, request, render_template, jsonify
from langchain_community.llms import VLLMOpenAI
import httpx
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings

app = Flask(__name__)

# Lê as variáveis de ambiente
llm_api_base = os.getenv('LLM_API_BASE', '')
llm_api_key = os.getenv('LLM_API_KEY', '')
llm_model_name = os.getenv('LLM_MODEL_NAME', '')

# Inicializa o LLM
llm = VLLMOpenAI(
    openai_api_base=f'{llm_api_base}/v1',
    openai_api_key=llm_api_key,
    model_name=llm_model_name,
    max_tokens=1024,
    async_client=httpx.AsyncClient(verify=False),
    http_client=httpx.Client(verify=False)
)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# Load Vector Store
vector_store = FAISS.load_local(embeddings=embeddings, folder_path="./faiss_db",allow_dangerous_deserialization=True)
 
# configure document retrieval 
retriever = vector_store.as_retriever(search_kwargs={'k': 2,"score_threshold": 0.8})

class Classification(BaseModel):
    servicos: str = Field(enum=["Saneamento básico", "Documentos pessoais", "Educação", "Impostos, dívidas e empresas", "Direitos e cidadania", "Moradia e servições sociais", "CNH, veículos, transporte, multas, licenciamento, transferencia de veiculos",
    "Atendimento a consumidores", "Saúde", "Serviços municipais"])
    tipo_manifestacao: str = Field(enum=["Sugestão", "Reclamação", "Solicitação de Providências", "Elogio", "Denúncia", "Pedido de Acesso à Informação"])


# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['prompt']
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    context = context if context.strip() else "N/A"

    if context == "N/A":
        print("Não tenho base de conhecimento para tratar desse assunto.")
        response = "Não tenho base de conhecimento para tratar desse assunto."
    else:
        output_parser = PydanticOutputParser(pydantic_object=Classification)

        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    "Classifique o seguinte texto: '{text}'.\nFormate sua resposta como um objeto JSON seguindo o seguinte esquema:\n{schema}\n"
                )
            ]
        ).partial(schema=output_parser.get_format_instructions())

        chain = prompt | llm | output_parser

        texto_para_classificar = question

        response = chain.invoke({"text": texto_para_classificar})

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

    

