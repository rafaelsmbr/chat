import os
from flask import Flask, request, render_template, jsonify
from langchain_community.llms import VLLMOpenAI

app = Flask(__name__)

# Lê as variáveis de ambiente
llm_api_base = os.getenv('LLM_API_BASE', '')
llm_api_key = os.getenv('LLM_API_KEY', '')
llm_model_name = 'falcon-40b'

# Inicializa o LLM
llm = VLLMOpenAI(
    openai_api_base=f'{llm_api_base}/v1',
    openai_api_key=llm_api_key,
    model_name=llm_model_name,
    max_tokens=1024
)

# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['prompt']
    response = llm.invoke(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
