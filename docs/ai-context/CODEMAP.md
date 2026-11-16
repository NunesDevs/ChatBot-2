# Mapa do código

- app.py: factory Flask, limites HTTP, autenticação de edição e headers.
- intents_core.py: esquema, normalização, matching, cache e gravação atômica com lock exclusivo.
- chat_service.py: respostas e invalidação de modelo desatualizado.
- train.py: preparação e treinamento explícito, CPU e threads limitadas, publicação em diretório novo.
- model_runtime.py: metadados/versionamento/SHA-256 e carregamento neural opcional.
- templates/index.html e static/: interface derivada do Figma, conversa e diálogo de edição.
- tests/: regressões Python/API/CLI e JavaScript.
- intents.json: base demonstrativa versionada. model/*.pkl: artefatos históricos preservados e não carregados.
