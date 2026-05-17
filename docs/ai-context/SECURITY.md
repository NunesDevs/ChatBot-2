# Segurança

- app.py contém termo sensível: token
- CLAUDE.md contém termo sensível: secret
- train.py contém termo sensível: token
- docs\ai-context\LAST_ANALYSIS.md contém termo sensível: token
- docs\ai-context\PROJECT.md contém termo sensível: secret
- docs\ai-context\PROMPTS.md contém termo sensível: secret
- docs\ai-context\SECURITY.md contém termo sensível: secret

## Checklist
- Verificar secrets.
- Verificar autenticação.
- Verificar autorização.
- Verificar IDOR.
- Verificar XSS.
- Verificar SQL Injection.
- Verificar CSRF.
- Verificar rate limit.
- Verificar uploads.
- Verificar logs.

<!-- AUTO-GENERATED:START -->
## Achados automáticos
- `app.py` contém termo sensível: **token**
- `CLAUDE.md` contém termo sensível: **secret**
- `train.py` contém termo sensível: **token**

## Checklist
- Verificar secrets.
- Verificar autenticação e autorização.
- Verificar IDOR, XSS, SQL Injection e CSRF quando aplicável.
- Verificar uploads, logs e rate limit.
<!-- AUTO-GENERATED:END -->

<!-- JARVIS-AUTO:START -->
## Alertas automáticos
- `app.py` contém termo sensível: token
- `CLAUDE.md` contém termo sensível: secret
- `train.py` contém termo sensível: token

## Regras obrigatórias
- Nunca commitar `.env`.
- Verificar secrets antes de commit.
- Verificar autenticação, autorização, uploads e logs quando aplicável.
<!-- JARVIS-AUTO:END -->
