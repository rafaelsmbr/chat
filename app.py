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
from langchain_core.exceptions import OutputParserException
import json
import re

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

model_name = "local_all_mpnet_base_v2"
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
retriever = vector_store.as_retriever(search_kwargs={'k': 1,"score_threshold": 0.8})

class Classification(BaseModel):
    tipo_manifestacao: str = Field(
        enum=[
            "Sugestão", "Reclamação", "Solicitação de Providências", 
            "Elogio", "Denúncia", "Pedido de Acesso à Informação"
        ],
        description=(
            "Tipo de manifestação enviada pelo usuário.\n"
            "Valores possíveis:\n"
            "- Sugestão: ideia para melhorar algo\n"
            "- Reclamação: queixa sobre algo\n"
            "- Solicitação de Providências: pedido de resolução de problema\n"
            "- Elogio: reconhecimento positivo\n"
            "- Denúncia: reporte de irregularidade\n"
            "- Pedido de Acesso à Informação: solicitação de dados/informações oficiais"
        )
    )

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
        resp = "Não tenho base de conhecimento para tratar desse assunto."
    else:
        arr = context.split("@@")
        org = arr[0]
        service = arr[1]
        output_parser = PydanticOutputParser(pydantic_object=Classification)
    
        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(
                    "Classifique o seguinte texto: '{text}'.\nFormate sua resposta como um objeto JSON estrito, sem nenhum texto adicional ou formatação markdown. O JSON deve seguir o seguinte esquema:\n{schema}\nCertifique-se de que o valor para 'tipo_manifestacao' sejam exatamente um dos seguintes:\n\nTipos de Manifestação: {tipo_manifestacao_enums}\n\nRetorne APENAS o objeto JSON válido."
                )
            ]
        ).partial(
            schema=output_parser.get_format_instructions(),
            tipo_manifestacao_enums=Classification.model_json_schema()['properties']['tipo_manifestacao']['enum']
        )
    
        chain = prompt | llm | output_parser
    
        texto_para_classificar = question
        
        try:
            response = chain.invoke({"text": texto_para_classificar})
        except OutputParserException as e:
            raw_output = e.llm_output
            #print("Saída bruta do modelo:", raw_output)
        
            # Try to extract the first valid JSON object from the string
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if match:
                try:
                    cleaned_json = json.loads(match.group(0))
                    response = Classification(**cleaned_json)
                except json.JSONDecodeError as decode_err:
                    response = {"error": f"Erro ao interpretar JSON: {decode_err}"}
            else:
                response = {"error": "Não foi possível extrair JSON da saída do modelo."}
        #print(response.model_dump())
        resp = "tipo_orgao='" +org+ "', tipo_servico='" +service+ "', " + str(response)
    

    return jsonify({'response': resp})

if __name__ == '__main__':
    app.run(debug=True)

    

