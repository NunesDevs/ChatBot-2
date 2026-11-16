'use strict';

const greeting = 'Olá! Posso ajudar com as intenções cadastradas.';
let sending = false;
let generation = 0;
let controller = null;
let failedMessage = null;
let failedRow = null;

function appendMessage(chatbox, speaker, message) {
    const row = document.createElement('div');
    row.className = 'message ' + (speaker === 'Você' ? 'user' : 'assistant');
    const label = document.createElement('strong');
    label.className = 'speaker';
    label.textContent = speaker;
    row.appendChild(label);
    row.appendChild(document.createTextNode(String(message)));
    chatbox.appendChild(row);
    while (chatbox.children.length > 100) chatbox.firstElementChild.remove();
    chatbox.scrollTop = chatbox.scrollHeight;
    return row;
}

function status(message, state = 'ready') {
    const element = document.getElementById('chat_status');
    element.textContent = message;
    element.dataset.state = state;
}

function setSending(value) {
    sending = value;
    document.getElementById('send').disabled = value;
    document.getElementById('send').textContent = value ? 'Enviando…' : 'Enviar';
    document.getElementById('user_input').disabled = value;
    document.getElementById('chatbox').setAttribute('aria-busy', String(value));
    for (const button of document.querySelectorAll('[data-question]')) button.disabled = value;
}

async function sendMessage() {
    const input = document.getElementById('user_input');
    const message = input.value.trim();
    if (sending) return;
    if (!message || message.length > 500) { status('Use uma mensagem de 1 a 500 caracteres.', 'error'); return; }
    const chatbox = document.getElementById('chatbox');
    const row = failedMessage === message && failedRow?.isConnected ? failedRow : appendMessage(chatbox, 'Você', message);
    const current = ++generation;
    controller = new AbortController();
    const requestController = controller;
    const timeout = setTimeout(() => requestController.abort(), 10000);
    input.value = '';
    setSending(true);
    status('Aguardando resposta…', 'sending');
    try {
        const response = await fetch(document.body?.dataset.chatUrl || '/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
            signal: requestController.signal,
        });
        if (!response.ok) throw new Error('Request failed');
        const data = await response.json();
        if (typeof data.response !== 'string' || !data.response || data.response.length > 2000) throw new Error('Invalid response');
        if (current !== generation) return;
        appendMessage(chatbox, 'Assistente', data.response);
        failedMessage = failedRow = null;
        status('Resposta recebida. Enter para enviar. Até 500 caracteres.');
    } catch {
        if (current !== generation) return;
        input.value = message;
        failedMessage = message;
        failedRow = row;
        status('Não foi possível enviar. Sua pergunta foi preservada.', 'error');
    } finally {
        clearTimeout(timeout);
        if (current === generation) {
            controller = null;
            setSending(false);
            input.focus();
        }
    }
}

function clearChat() {
    generation++;
    controller?.abort();
    controller = null;
    failedMessage = failedRow = null;
    const chatbox = document.getElementById('chatbox');
    chatbox.replaceChildren();
    appendMessage(chatbox, 'Assistente', greeting);
    document.getElementById('user_input').value = '';
    setSending(false);
    status('Enter para enviar. Até 500 caracteres.');
    document.getElementById('user_input').focus();
}

if (typeof document !== 'undefined' && document.addEventListener) {
    document.addEventListener('DOMContentLoaded', () => {
        appendMessage(document.getElementById('chatbox'), 'Assistente', greeting);
        document.getElementById('chat_form').addEventListener('submit', event => { event.preventDefault(); sendMessage(); });
        document.getElementById('clear').addEventListener('click', clearChat);
        for (const button of document.querySelectorAll('[data-question]')) button.addEventListener('click', () => {
            document.getElementById('user_input').value = button.dataset.question;
            sendMessage();
        });
        window.addEventListener('pagehide', () => controller?.abort());
    });
}
