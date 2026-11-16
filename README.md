# ChatBot-2 — assistente de intenções

Aplicação local de estudo com Flask, base JSON e interface responsiva. Funciona ao iniciar sem modelo treinado: combina padrões normalizados e responde com uma das respostas cadastradas. TensorFlow/Keras é opcional, treinado por um comando explícito.

## Executar

Recomendado: Python 3.12 e ambiente virtual próprio. A base web não instala TensorFlow.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --only-binary=:all: --require-hashes -r requirements.lock
.venv\Scripts\python app.py
```

Abra `http://127.0.0.1:5000`. O servidor fica em loopback e sem debug. O arquivo `.env.example` documenta variáveis; nenhum carregamento automático de `.env` é feito.

A conversa aceita até 500 caracteres, oferece sugestões, preserva a pergunta quando a rede falha e permite limpar o histórico. Mensagens são texto, com identificação do autor. As fontes são locais.

## Cadastrar intenções

A edição vem desligada. Um operador pode habilitá-la configurando `CHATBOT_ENABLE_LEARNING=1` e uma `CHATBOT_LEARNING_TOKEN` aleatória de 32 a 128 caracteres ASCII (`A–Z a–z 0–9 . _ ~ -`) no ambiente do processo. Gere a chave localmente, por exemplo com `python -c "import secrets; print(secrets.token_urlsafe(32))"`, e não a versione.

Com edição habilitada, o botão **Ensinar** abre um formulário. A chave é usada somente no cabeçalho da requisição atual e apagada do campo; não vai para HTML, URL, JSON, cookies ou armazenamento do navegador. A API `POST /learn` exige `Authorization: Bearer <chave>`.

Uma intenção tem tag iniciada por `a–z`, seguida de `a–z`, `0–9` ou `_` (até 64 caracteres), pergunta de até 500 e resposta de até 2.000. A escrita valida a base, rejeita conflito de padrão entre tags, deduplica repetições e usa lock exclusivo mais troca atômica. Os testes escrevem apenas em arquivos temporários.

O cadastro fica disponível no modo de padrões imediatamente. Não existe comando `!ensinar` nem treinamento durante uma requisição.

## Treinamento neural opcional

Primeiro, confira a base sem TensorFlow, rede ou escrita:

```text
python train.py --prepare-only
```

Instale as dependências opcionais no ambiente virtual e treine para uma **nova** pasta:

```powershell
.venv\Scripts\python -m pip install --only-binary=:all: --require-hashes -r requirements-ml.lock
.venv\Scripts\python train.py --epochs 40 --output model-trained
$env:CHATBOT_MODEL_DIRECTORY = (Resolve-Path model-trained).Path
.venv\Scripts\python app.py
```

O CLI limita épocas, vocabulário, amostras, matriz e threads, utiliza CPU e recusa uma saída existente. Salva `model.keras` e metadados JSON com versão do tokenizer, classes, vocabulário e SHA-256. Não baixa corpora, não importa NLTK e não lê os Pickles históricos preservados em `model/`.

Padrões exatos e correspondências claras têm prioridade; o modelo é consultado quando o matcher não reconhece a mensagem. Respostas neurais abaixo de 0,65 usam fallback. Alterar a base torna o modelo obsoleto; o serviço volta aos padrões até novo treinamento. `GET /status` informa esse estado.

Só carregue modelos produzidos e revisados pelo operador. Hash confirma integridade, não torna confiável um arquivo de terceiros. Métricas do treinamento não medem a qualidade em perguntas novas.

## Validação

```text
python -m unittest discover -s tests -v
node --test tests/chatbot.test.cjs tests/teacher.test.cjs
python train.py --prepare-only
```

Para repetir a passagem web, instale a ferramenta de teste com `pnpm install --frozen-lockfile --ignore-scripts`, execute `pnpm exec playwright install chromium` e `pnpm test:e2e`. O script inicia apenas servidores locais, usa fixture descartável para edição e registra oito larguras em cada modo. Configure `PYTHON_EXECUTABLE` se o Python do ambiente não estiver em `.venv`.

Validação local em 2026-11-15: **20 testes Python + 8 Node**, smoke neural real em fixture descartável e Playwright nos dois modos em **16 layouts / oito larguras**, com quatro capturas reais revisadas e zero erros inesperados. O smoke neural verifica funcionamento e integridade, sem afirmar qualidade de generalização.

A suíte cobre esquema, normalização, ambiguidade, conflitos, edição atômica, autorização, limites HTTP, invalidação de modelo, preservação de dados, CLI, mensagens seguras, requisições concorrentes, repetição após falha e limpeza da chave. Consulte `docs/ai-context/LAST_ANALYSIS.md` para resultados efetivamente executados.

Design editável: [telas do assistente no Figma](https://www.figma.com/design/Huk9z4K8ORoDykKz85r7Ad?node-id=77-653). A implementação usa HTML/CSS/JavaScript nativos, com DM Sans e Space Grotesk locais e suas licenças OFL.

## Operação e licença

Este é um projeto de estudo com respostas demonstrativas. Para hospedagem pública, é necessário definir servidor WSGI, HTTPS, controle de operadores e limites próprios; nenhum serviço público foi configurado nesta revisão.

A [licença MIT existente](LICENSE) foi preservada. As licenças de dependências e fontes continuam próprias. Memória técnica, decisões e limites estão em `docs/ai-context/`.
