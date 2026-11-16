"""Controlled QA server: explicit repo and temporary fixture, no debug/reloader."""
import json
from pathlib import Path
import sys
from werkzeug.serving import make_server
project=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(project))
from app import create_app
config={'LEARNING_ENABLED':False,'LEARNING_TOKEN':'','MODEL_DIRECTORY':''}
if len(sys.argv)>1:
    fixture=Path(sys.argv[1]).resolve()
    if not fixture.is_file() or not fixture.name.startswith('chatbot2-fixture-'):
        raise RuntimeError('QA fixture is outside the dedicated control directory')
    config.update(INTENTS_PATH=fixture,LEARNING_ENABLED=True,LEARNING_TOKEN='unit-test-learning-key-for-local-fixtures-only')
server=make_server('127.0.0.1',0,create_app(config),threaded=True)
print(json.dumps({'port':server.server_port}),flush=True)
server.serve_forever()
