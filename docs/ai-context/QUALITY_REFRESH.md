# Validação em 2026-11-15

20 testes Python/API/CLI e 8 testes Node passaram na venv isolada. Instalação de 39 dependências opcionais feita com wheels oficiais e SHA-256 conferidos antes de instalar; pip check sem incompatibilidades.

Smoke real: TensorFlow 2.21.0/NumPy 2.5.4 treinou 60 épocas em fixture descartável de 8 padrões/2 classes, salvou/carregou Keras e classificou as duas amostras verificadas. Modelo adulterado foi rejeitado por hash e edição da base invalidou o modelo, voltando aos padrões. Isso comprova funcionamento, não generalização. Nenhum arquivo treinado ou dado da fixture foi mantido no repositório.

Playwright com Edge real: 2 fluxos (somente leitura e edição autenticada), 16 layouts (320,360,390,430,768,1024,1440,1920 em cada modo), 4 capturas reais e nenhum erro JavaScript/HTTP inesperado. Falhas 401/503 foram provocadas apenas para testar recuperação; dataset original permaneceu inalterado. Foco, Escape/Tab, teclado/touch, texto seguro, chave removida e ausência de armazenamento de chave foram verificados. Capturas desktop/mobile de conversa e diálogo foram inspecionadas visualmente.

Figma: cinco telas editáveis; recorte móvel corrigido para altura automática de 88 px. Interface aplicada aos templates/static do projeto. CI fixa ações oficiais por SHA e repete regressões e passagem web com Chromium, sem treinamento nem serviços externos.

Uma tentativa anterior de iniciar o Python saiu com STATUS_DLL_INIT_FAILED(-1073741502) durante uso compartilhado do PC. Ela não foi contada como aprovada; toda a suíte foi repetida com êxito após liberação exclusiva.

Execução: python app.py; factory inerte ao importar. Testes de edição e smoke neural usaram somente arquivos temporários. Pickles históricos preservados sem carregamento. MIT existente preservada; fontes locais incluem OFL. Sem hospedagem pública configurada; distribuição é pacote local.
