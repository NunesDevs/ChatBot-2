# Segurança e operação

Edição desabilitada por padrão. Para ativar, o operador configura flag booleana e chave de 32 a 128 caracteres ASCII seguros. POST /learn exige Bearer em comparação constante; a chave nunca é renderizada nem persistida pelo cliente. Não habilite edição em hospedagem compartilhada sem controlar operadores e HTTPS.

Requisições limitadas a 16 KiB; mensagens de 1–500 caracteres e respostas de 1–2.000. Intenções validadas antes da escrita, conflitos de padrão rejeitados. Lock O_EXCL, arquivo temporário na mesma pasta, fsync e replace atômico; falhas preservam os dados anteriores. Um lock abandonado exige inspeção manual, sem apagamento automático de lock alheio.

Mensagens são textContent/text nodes, sem innerHTML. CSP somente mesma origem, sem código inline; no-store/nosniff/no-referrer; servidor de desenvolvimento somente loopback e sem debug. Não há downloads, treinamento nem unpickle em requisições.

Modelos devem ser produzidos e revisados pelo operador. SHA-256 confere integridade, não identifica a autoria de um arquivo não confiável. Metadados e arquivos são limitados; Keras safe_mode está habilitado. Não carregue modelos recebidos de terceiros. Não existe endpoint de upload de modelo.
