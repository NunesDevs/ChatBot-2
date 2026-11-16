'use strict';

let teaching = false;
let teachController = null;
let teachGeneration = 0;

function teachStatus(message, error = false) {
    const element = document.getElementById('teach_status');
    element.textContent = message;
    element.dataset.state = error ? 'error' : 'ready';
}

function teachBusy(value) {
    teaching = value;
    document.getElementById('save_teach').disabled = value;
    document.getElementById('save_teach').textContent = value ? 'Salvando…' : 'Salvar intenção';
    for (const id of ['operator_key', 'intent_tag', 'intent_pattern', 'intent_response']) document.getElementById(id).disabled = value;
    document.getElementById('teach_form').setAttribute('aria-busy', String(value));
}

function resetTeacher() {
    teachGeneration++;
    teachController?.abort();
    teachController = null;
    teachBusy(false);
    document.getElementById('teach_form').reset();
    document.getElementById('teach').focus();
}

async function saveIntent() {
    if (teaching) return;
    const form = document.getElementById('teach_form');
    if (!form.reportValidity()) return;
    const key = document.getElementById('operator_key');
    const token = key.value;
    if (!/^[A-Za-z0-9._~-]{32,128}$/.test(token)) {
        key.value = '';
        teachStatus('Use a chave de edição fornecida pelo operador.', true);
        key.focus();
        return;
    }
    const data = {
        tag: document.getElementById('intent_tag').value.trim(),
        message: document.getElementById('intent_pattern').value.trim(),
        response: document.getElementById('intent_response').value.trim(),
    };
    if (!/^[a-z][a-z0-9_]{0,63}$/.test(data.tag) || !data.message || !data.response || data.message.length > 500 || data.response.length > 2000) {
        teachStatus('Confira a tag, a pergunta e a resposta antes de salvar.', true);
        return;
    }
    // Keep the key only in this request closure; never store it in page state or storage.
    key.value = '';
    const current = ++teachGeneration;
    teachController = new AbortController();
    const controller = teachController;
    const timeout = setTimeout(() => controller.abort(), 10000);
    teachBusy(true);
    teachStatus('Salvando intenção…');
    try {
        const response = await fetch(document.body.dataset.learnUrl || '/learn', {
            method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: 'Bearer ' + token },
            body: JSON.stringify(data), signal: controller.signal,
        });
        const result = await response.json();
        if (current !== teachGeneration) return;
        if (!response.ok) {
            if (response.status === 401 || response.status === 403) throw new Error('A chave está incorreta ou a edição foi desativada.');
            if (response.status === 400) throw new Error('Confira os dados; o padrão pode pertencer a outra intenção.');
            throw new Error('Não foi possível confirmar a gravação. Repetir os mesmos dados é seguro.');
        }
        if (typeof result.changed !== 'boolean') throw new Error('Não foi possível confirmar a gravação.');
        form.reset();
        teachStatus(result.changed ? 'Intenção cadastrada. O treinamento neural é separado.' : 'Esta intenção já estava cadastrada.');
    } catch (error) {
        if (current !== teachGeneration) return;
        teachStatus(error.name === 'AbortError' ? 'O tempo de espera acabou. Repetir os mesmos dados é seguro.' : error.message, true);
    } finally {
        clearTimeout(timeout);
        if (current === teachGeneration) {
            teachController = null;
            teachBusy(false);
            key.value = '';
        }
    }
}

function trapTeacherFocus(event) {
    if (event.key !== 'Tab') return;
    const dialog = document.getElementById('teach_dialog');
    const controls = [...dialog.querySelectorAll('button:not([disabled]), input:not([disabled]), textarea:not([disabled])')].filter(node => node.getClientRects().length);
    const first = controls[0], last = controls.at(-1);
    if (!first) return;
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
}

if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('DOMContentLoaded', () => {
        const dialog = document.getElementById('teach_dialog');
        document.getElementById('teach').addEventListener('click', () => {
            document.getElementById('teach_form').reset();
            teachStatus('A base local é atualizada. O treinamento neural é executado separadamente pelo operador.');
            dialog.showModal();
            document.getElementById('operator_key').focus();
        });
        for (const id of ['close_teach', 'cancel_teach']) document.getElementById(id).addEventListener('click', () => dialog.close());
        dialog.addEventListener('close', resetTeacher);
        dialog.addEventListener('keydown', trapTeacherFocus);
        document.getElementById('teach_form').addEventListener('submit', event => { event.preventDefault(); saveIntent(); });
        window.addEventListener('pagehide', () => { teachController?.abort(); document.getElementById('operator_key').value = ''; });
    });
}
