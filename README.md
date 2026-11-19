# Global Solution - Segurança Pública Inteligente em São Paulo
**Classificação de ocorrências de subtração de celulares (Roubo x Furto) com MLP**

- **Disciplina:** Aprendizado Profundo com Redes Neurais Convolucionais
- **Aplicação Web Online:** [Acessar Sistema no Streamlit]((https://global-6pd2lwztfpzkvnfxh9nwkx.streamlit.app/))

## Integrantes
- Kaue Souza Rodrigues — RM 567716
- Gabriel Kenishi Furuzawa — RM 568245
- Pedro Henrique Camacho de Alencar — RM 568071

## Fonte dos Dados
- **Origem:** Secretaria da Segurança Pública de São Paulo (SSP-SP) — Portal de Estatísticas
- **Link:** https://www.ssp.sp.gov.br/estatistica/consultas (seção *Bases de Dados* -> *Celulares Subtraídos - 2025*)
- **Recorte utilizado:** Boletins únicos registrados na capital (`S.PAULO`) entre janeiro e dezembro de 2025 (161.669 ocorrências únicas, com amostra estratificada de 20.000 registros para treino/validação/teste).

## Estrutura do Repositório
- `notebook_gs.ipynb`: Notebook completo com análise exploratória, prevenção de *data leakage*, pré-processamento, comparação entre 5 arquiteturas de MLP e avaliação no conjunto de teste.
- `app.py`: Aplicação web desenvolvida em Streamlit para inferência interativa.
- `artefatos/`: Modelo treinado (`modelo_mlp.keras`), pré-processador (`preprocessador.joblib`), opções do formulário (`opcoes_formulario.json`) e métricas (`resumo_modelo.json`).
- `requirements.txt`: Dependências necessárias para execução.

## Como Executar Localmente
```bash
git clone [https://github.com/kaue-sr/Global.git](https://github.com/kaue-sr/Global.git)
cd Global
pip install -r requirements.txt
streamlit run app.py
