
# Chatbot FAQ com Aprendizado Dinâmico

Este é um chatbot interativo para responder a perguntas frequentes (FAQ), com a capacidade de aprender novas intenções diretamente a partir do chat. O bot pode ser treinado em tempo real sem a necessidade de re-treinamento manual, utilizando um comando especial para adicionar novas intenções.

## Funcionalidades

- Responde a perguntas frequentes.
- Aprende novas intenções com o comando `!ensinar`.
- Utiliza um modelo de rede neural treinado com TensorFlow/Keras.
- Suporta interação em tempo real via Flask e frontend em HTML/JavaScript.
  
## Tecnologias Utilizadas

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Modelo de Aprendizado**: TensorFlow, Keras
- **Natural Language Processing**: NLTK
- **Banco de Dados**: Arquivos JSON e Pickle

## Estrutura do Projeto

```
ChatBot-2/
│
├──model/ 
│   ├── words.pkl   
│   ├── classes.pkl            
│   └── chatbot_model.h5        
│
├── intents.json        
├── app.py                 
├──templates/
    ├── index.html          
├──             
├── .gitignore                  
└── README.md                   
```

## Como Rodar o Projeto

### 1. Instalar Dependências

Primeiro, instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

### 2. Treinando o Modelo

Quando o bot for iniciado pela primeira vez, ele irá tentar carregar um modelo treinado. Caso não exista, ele criará um novo modelo de rede neural e o treinará utilizando os dados de intenções do arquivo `intents.json`.

### 3. Rodando o Servidor Flask

Execute o seguinte comando para iniciar o servidor Flask:

```bash
python app.py
```

O servidor estará disponível em `http://localhost:5000`.

### 4. Interagir com o Chatbot

Acesse `http://localhost:5000` no seu navegador para começar a interagir com o chatbot.

### 5. Ensinando o Bot

Para ensinar o bot novas intenções, envie um comando especial no chat com a seguinte estrutura:

```
!ensinar [tag] ; [mensagem] ; [resposta]
```

Exemplo:

```
!ensinar saudacao ; Olá! ; Olá, como posso te ajudar?
```

Isso irá adicionar uma nova intenção ao arquivo `intents.json`, treinar o modelo novamente e permitir que o bot responda a novas perguntas relacionadas à tag "saudacao".

### 6. Reiniciando o Servidor

Caso o bot aprenda novas intenções, o modelo será treinado novamente. Se isso ocorrer, reinicie o servidor para que o modelo mais recente seja carregado.

## Contribuindo

Sinta-se à vontade para contribuir com este projeto. Abra um *pull request* ou envie um *issue* para sugestões de melhorias.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.
