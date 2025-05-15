import os
from flask import Flask, request, render_template, jsonify
from langchain_community.llms import VLLMOpenAI
import httpx
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.exceptions import OutputParserException
import json
import re

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.checkpoint.memory import MemorySaver

df = pd.read_excel("./faiss_db/planilha_atualizada.xlsx")

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

class State(TypedDict):
    input: str
    resposta1: str
    resposta2: list[str]
    resposta3: str
    user_feedback: str

class ClassificationOrgao(BaseModel):
    tipo_orgao: str = Field(
        enum=["Administração Penitenciária","Agricultura e Abastecimento","Casa Civil","Casa Militar e Defesa Civil","Ciência, Tecnologia e Inovação","Comunicação","Controladoria Geral do Estado","Cultura, Economia e Indústrias Criativas","Desenvolvimento Econômico","Desenvolvimento Social","Desenvolvimento Urbano e Habitação","Direitos da Pessoa com Deficiência","Educação","Esportes","Fazenda e Planejamento","Gestão e Governo Digital","Governo e Relações Institucionais","Justiça e Cidadania","Meio Ambiente, Infraestrutura e Logística","Negócios Internacionais","Parcerias em Investimentos","Políticas para a Mulher","Procuradoria Geral do Estado","Projetos Estratégicos","Saúde","Segurança Pública","Transportes Metropolitanos","Turismo e Viagens","CDHU","Cetesb","Companhia Docas de São Sebastião","CPP","CPSEC","CPTM","Desenvolve SP","EMAE","EMTU","IPT","Metrô","Prodesp","Sabesp","AGEM","AGEMCAMP","AGEMVALE","Arcesp","Artesp","CBPM","Centro Paula Souza","DAEE","DER","Detran.SP","Famema","Famerp","HC Botucatu","HC Famema","HC USP-RP","HC USP-SP","Iamspe","IMESC","Ipem-SP","Ipen","Ipesp","Jucesp","SPPREV","Unesp","Unicamp","USP","Fapesp","FDE","Funap","Fundação Casa","Fundação Florestal","Fundação Itesp","Fundação Padre Anchieta","Fundação Procon-SP","Furp","Memorial da América Latina","Oncocentro","Pró-Sangue","Seade","Prevcom","Univesp"],
        description=(
            """Valores possíveis para os orgãos:\n
- Administração Penitenciária: Responsável pela gestão do sistema prisional do estado.
- Agricultura e Abastecimento: Coordena políticas para o agronegócio e segurança alimentar.
- Casa Civil: Articula ações entre o governador e demais secretarias.
- Casa Militar e Defesa Civil: Atua na proteção civil e apoio a desastres.
- Ciência, Tecnologia e Inovação: Promove pesquisa, inovação e desenvolvimento tecnológico.
- Comunicação: Gerencia a comunicação institucional e relações com a mídia.
- Controladoria Geral do Estado: Fiscaliza a legalidade e eficiência dos gastos públicos.
- Cultura, Economia e Indústrias Criativas: Incentiva a produção cultural e setores criativos.
- Desenvolvimento Econômico: Formula políticas para o crescimento e geração de empregos.
- Desenvolvimento Social: Executa ações de assistência e inclusão social.
- Desenvolvimento Urbano e Habitação: Planeja a urbanização e programas habitacionais.
- Direitos da Pessoa com Deficiência: Garante políticas públicas de inclusão e acessibilidade.
- Educação: Administra a rede estadual de ensino e políticas educacionais.
- Esportes: Incentiva a prática esportiva e organiza eventos do setor.
- Fazenda e Planejamento: Coordena o orçamento, arrecadação e planejamento fiscal.
- Gestão e Governo Digital: Moderniza a administração pública por meio de tecnologia.
- Governo e Relações Institucionais: Conduz o relacionamento com entidades e poderes.
- Justiça e Cidadania: Promove direitos humanos, cidadania e justiça social.
- Meio Ambiente, Infraestrutura e Logística: Atua na preservação ambiental e obras públicas.
- Negócios Internacionais: Fortalece relações comerciais e diplomáticas do estado.
- Parcerias em Investimentos: Desenvolve projetos com o setor privado via PPPs.
- Políticas para a Mulher: Coordena ações voltadas à equidade de gênero.
- Procuradoria Geral do Estado: Representa o estado judicialmente e em pareceres legais.
- Projetos Estratégicos: Gerencia iniciativas prioritárias para o governo.
- Saúde: Gerencia o SUS estadual e políticas públicas de saúde.
- Segurança Pública: Coordena as polícias e ações de segurança do estado.
- Transportes Metropolitanos: Planeja e executa políticas de mobilidade urbana.
- Turismo e Viagens: Fomenta o turismo e promove os destinos do estado.
- CDHU: Constrói e financia moradias populares.
- Cetesb: Fiscaliza e monitora o meio ambiente.
- Companhia Docas de São Sebastião: Administra o porto de São Sebastião.
- CPP: Cuida da previdência dos policiais civis paulistas.
- CPSEC: Atua na gestão de riscos e garantias do estado.
- CPTM: Opera trens metropolitanos da Grande São Paulo.
- Desenvolve SP: Oferece crédito e apoio ao desenvolvimento empresarial.
- EMAE: Gera energia elétrica e administra recursos hídricos.
- EMTU: Organiza o transporte metropolitano intermunicipal.
- IPT: Instituto de pesquisa e desenvolvimento tecnológico.
- Metrô: Opera e expande a malha metroviária paulista.
- Prodesp: Empresa de tecnologia da informação do governo.
- Sabesp: Fornece água e saneamento para grande parte do estado.
- AGEM: Planeja e coordena políticas regionais da Baixada Santista.
- AGEMCAMP: Atende à Região Metropolitana de Campinas.
- AGEMVALE: Atua no planejamento do Vale do Paraíba e Litoral Norte.
- Arcesp: Regula serviços de saneamento básico.
- Artesp: Regula e fiscaliza o transporte rodoviário e concessões.
- CBPM: Atua na produção de materiais gráficos e segurança documental.
- Centro Paula Souza: Administra Etecs e Fatecs no estado.
- DAEE: Cuida dos recursos hídricos e obras hidráulicas.
- DER: Gerencia rodovias estaduais não concedidas.
- Detran.SP: Administra registros e fiscalização de veículos e condutores.
- Famema: Faculdade de Medicina de Marília.
- Famerp: Faculdade de Medicina de São José do Rio Preto.
- HC Botucatu: Hospital universitário da Unesp de Botucatu.
- HC Famema: Hospital vinculado à Famema.
- HC USP-RP: Hospital das Clínicas da USP de Ribeirão Preto.
- HC USP-SP: Hospital das Clínicas da Faculdade de Medicina da USP.
- Iamspe: Sistema de saúde para servidores públicos estaduais.
- IMESC: Produz laudos e perícias médico-legais.
- Ipem-SP: Fiscaliza pesos, medidas e produtos regulamentados.
- Ipen: Atua em pesquisas nucleares e radioisótopos.
- Ipesp: Órgão previdenciário e cartorial desativado, com gestão residual.
- Jucesp: Junta comercial que registra empresas e atos mercantis.
- SPPREV: Previdência dos servidores públicos estaduais.
- Unesp: Universidade Estadual Paulista.
- Unicamp: Universidade Estadual de Campinas.
- USP: Universidade de São Paulo.
- Fapesp: Financia pesquisas científicas no estado.
- FDE: Apoia obras e serviços para a rede estadual de ensino.
- Funap: Oferece trabalho e educação para presos.
- Fundação Casa: Responsável pela reabilitação de menores infratores.
- Fundação Florestal: Administra unidades de conservação ambiental.
- Fundação Itesp: Regulariza terras e apoia assentamentos.
- Fundação Padre Anchieta: Responsável pela TV Cultura e rádios educativas.
- Fundação Procon-SP: Defesa dos direitos do consumidor.
- Furp: Produz medicamentos para a rede pública.
- Memorial da América Latina: Espaço cultural e de integração latino-americana.
- Oncocentro: Pesquisa e diagnóstico em oncologia.
- Pró-Sangue: Banco de sangue que abastece hospitais do estado.
- Seade: Produz estatísticas e indicadores socioeconômicos.
- Prevcom: Previdência complementar dos servidores públicos.
- Univesp: Universidade virtual pública com ensino a distância."""
        )
    )

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


def step_1(state):
    output_parser = PydanticOutputParser(pydantic_object=ClassificationOrgao)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            HumanMessagePromptTemplate.from_template(
                "Classifique o seguinte texto: '{text}'.\n"
                "Responda com um JSON estrito no seguinte formato:\n\n"
                '{{"tipo_orgao": string }}\n\n'
                "O campo 'tipo_orgao' deve conter exatamente um dos seguintes valores:\n"
                "{tipo_orgao_enums}\n\n"
                "Retorne APENAS o objeto JSON válido, sem explicações ou formatação extra."
            )
        ]
    ).partial(
        schema=output_parser.get_format_instructions(),
        tipo_orgao_enums=ClassificationOrgao.model_json_schema()['properties']['tipo_orgao']['enum']
    )
    
    chain = prompt | llm | output_parser

    try:
        response = chain.invoke({"text": state['input']})
    except OutputParserException as e:
        raw_output = e.llm_output
        match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if match:
            try:
                cleaned_json = json.loads(match.group(0))
                response = ClassificationOrgao(**cleaned_json)
            except json.JSONDecodeError as decode_err:
                response = {"error": f"Erro ao interpretar JSON: {decode_err}"}
        else:
            response = {"error": "Não foi possível extrair JSON da saída do modelo."}

    return {"resposta1": response.model_dump_json()}


def step_2(state):
    vv = json.loads(state['resposta1'])
    df_filtrado = df[df.iloc[:,3] == vv['tipo_orgao']]#.head(5)  
    if not df_filtrado.empty:

        lista_enum = df_filtrado["servico"].unique().tolist()
        class ClassificationServico(BaseModel):
            tipo_servico: str = Field(
                enum=lista_enum,
                description="Lista de órgãos filtrados dinamicamente."
            )

        output_parser = PydanticOutputParser(pydantic_object=ClassificationServico)

        prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate.from_template(                   
                    "Pela lista de serviço prestados pelo Governo do Estado abaixo, qual o item que melhor atende ao pedido na manifestação?\n"
                    "Lista de serviço: '{tipo_servico_enums}'.\n"
                    "Manifestação: '{text}'.\n"
                    "Responda com um JSON estrito no seguinte formato:\n\n"
                    '{{"tipo_servico": string }}\n\n'
                    "Retorne APENAS o objeto JSON válido, sem explicações ou formatação extra."
                )
            ]
        ).partial(
            schema=output_parser.get_format_instructions(),
            tipo_servico_enums=ClassificationServico.model_json_schema()['properties']['tipo_servico']['enum']
        )

        chain = prompt | llm | output_parser

        try:
            response = chain.invoke({"text": state['input']})
        except OutputParserException as e:
            raw_output = e.llm_output
            #print("Saída bruta do modelo:", raw_output)
            # Try to extract the first valid JSON object from the string
            match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if match:
                try:
                    cleaned_json = json.loads(match.group(0))
                    response = ClassificationServico(**cleaned_json)
                except json.JSONDecodeError as decode_err:
                    response = {"error": f"Erro ao interpretar JSON: {decode_err}"}
            else:
                response = {"error": "Não foi possível extrair JSON da saída do modelo."}

        return {"resposta2": response.model_dump_json()}  

def step_3(state):
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

    texto_para_classificar = state['input']
    
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

    return {"resposta3": response.model_dump_json()} 


builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

# Set up memory
memory = MemorySaver()

# Add
graph = builder.compile(checkpointer=memory)


# Flask routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
initial_input = {"input": request.form['prompt']}

# Thread
thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(initial_input, thread, stream_mode="updates"):
    print(event)
    if "step_1" in event:
        tipo_orgao=str(event["step_1"])
    if "step_2" in event:
        tipo_servico=str(event["step_2"])
    if "step_3" in event:
        tipo_manifestacao=str(event["step_3"])
    print("\n")

    return jsonify({'response': tipo_orgao+", "+tipo_servico+", "+tipo_manifestacao})

if __name__ == '__main__':
    app.run(debug=True)


