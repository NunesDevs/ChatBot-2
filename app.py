"""Validated local intent service; learning is an explicitly enabled operator action."""
import hmac
import os
from pathlib import Path
import re
from flask import Flask, jsonify, render_template, request
from chat_service import ChatService
from intents_core import IntentStore, MAX_MESSAGE, text


def create_app(config=None, predictor=None):
    app = Flask(__name__)
    app.config.update(MAX_CONTENT_LENGTH=16384, INTENTS_PATH=Path(__file__).resolve().parent / 'intents.json', LEARNING_ENABLED=os.environ.get('CHATBOT_ENABLE_LEARNING', '').casefold() in ('1', 'true'), LEARNING_TOKEN=os.environ.get('CHATBOT_LEARNING_TOKEN', ''), MODEL_DIRECTORY=os.environ.get('CHATBOT_MODEL_DIRECTORY', ''))
    if config:
        app.config.update(config)
    if not isinstance(app.config['LEARNING_ENABLED'], bool):
        raise ValueError('A disponibilidade de edição deve ser um booleano')
    token = app.config['LEARNING_TOKEN']
    if app.config['LEARNING_ENABLED'] and (not isinstance(token, str) or not re.fullmatch(r'[A-Za-z0-9._~-]{32,128}', token)):
        raise ValueError('Edição habilitada exige uma chave de operador de 32 a 128 caracteres seguros')
    store = IntentStore(app.config['INTENTS_PATH'])
    if app.config['MODEL_DIRECTORY']:
        if predictor is not None:
            raise ValueError('Configure somente um provedor neural')
        from model_runtime import KerasPredictor
        predictor = KerasPredictor(app.config['MODEL_DIRECTORY'])
    service = ChatService(store, predictor)
    app.extensions['intent_store'] = store
    app.extensions['chat_service'] = service

    @app.get('/')
    def index():
        return render_template('index.html', editing_enabled=bool(app.config['LEARNING_ENABLED']))

    def payload():
        if not request.is_json:
            return None, (jsonify(error='Envie um objeto JSON.'), 415)
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return None, (jsonify(error='Envie um objeto JSON.'), 400)
        return data, None

    @app.post('/chat')
    def chat():
        data, error = payload()
        if error:
            return error
        try:
            message = text(data.get('message'), MAX_MESSAGE, 'Mensagem')
        except ValueError as error:
            return jsonify(error=str(error)), 400
        try:
            return jsonify(service.respond(message))
        except (OSError, RuntimeError, ValueError):
            return jsonify(error='O serviço não conseguiu responder agora.'), 503

    @app.get('/status')
    def status():
        try:
            return jsonify(**service.state(), editing_enabled=bool(app.config['LEARNING_ENABLED']))
        except (OSError, RuntimeError, ValueError):
            return jsonify(error='O serviço está indisponível agora.'), 503

    @app.post('/learn')
    def learn():
        if not app.config['LEARNING_ENABLED']:
            return jsonify(error='A edição de intenções está desativada.'), 403
        authorization = request.headers.get('Authorization', '')
        scheme, _, value = authorization.partition(' ')
        presented = value.strip() if scheme.casefold() == 'bearer' else ''
        if not re.fullmatch(r'[A-Za-z0-9._~-]{32,128}', presented) or not hmac.compare_digest(presented, token):
            return jsonify(error='Chave de edição inválida.'), 401
        data, error = payload()
        if error:
            return error
        try:
            changed = store.teach(data.get('message'), data.get('tag'), data.get('response'))
            return jsonify(response='Intenção cadastrada.' if changed else 'Esta intenção já estava cadastrada.', changed=changed)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        except (OSError, RuntimeError):
            return jsonify(error='Não foi possível salvar a intenção agora.'), 503

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error='A requisição excede o limite permitido.'), 413

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'no-referrer'
        response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        return response
    return app


if __name__ == '__main__':
    create_app().run(host='127.0.0.1', debug=False)
