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

# Função externa simulando uma API de câmbio
def get_exchange_rate(currency: str) -> str:
    return "5.34" if currency.upper() == "USD" else "Moeda não suportada"

# Tool que registra essa função
tools = [
    Tool(
        name="GetExchangeRate",
        func=lambda x: get_exchange_rate(x),
        description="Use esta ferramenta para consultar a cotação atual de uma moeda. Forneça o código (ex: USD)."
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['prompt']
    query = "Qual é a cotação do dólar agora? E quanto é 100 dólares em reais?"

    response = agent.run(query)

    return jsonify({'response': str(response)})

if __name__ == '__main__':
    app.run(debug=True)
