const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function fixture(fetch) {
  const element = () => ({
    children: [], dataset: {}, value: '', attrs: {}, isConnected: false,
    appendChild(child) { child.parent = this; child.isConnected = true; this.children.push(child); },
    get firstElementChild() { return this.children[0]; },
    remove() { this.parent.children = this.parent.children.filter(c => c !== this); this.isConnected = false; },
    replaceChildren() { for (const child of this.children) child.isConnected = false; this.children = []; },
    setAttribute(key, value) { this.attrs[key] = value; }, focus() {},
  });
  const ids = Object.fromEntries(['user_input', 'chatbox', 'chat_status', 'send'].map(id => [id, element()]));
  Object.defineProperty(ids.chatbox, 'innerHTML', { set() { throw new Error('HTML insertion is unsafe'); } });
  const context = { document: { getElementById: id => ids[id], createElement: element, createTextNode: text => ({ textContent: text }), querySelectorAll: () => [], body: { dataset: {} } }, fetch, AbortController, setTimeout, clearTimeout };
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname, '../static/js/chatbot.js'), 'utf8'), context);
  return { context, ...ids };
}
const response = text => ({ ok: true, json: async () => ({ response: text }) });

test('renders user input and bot replies as text instead of executable HTML', async () => {
  const payload = '<img src=x onerror=alert(1)>', reply = '<script>alert(2)</script>';
  const f = fixture(async () => response(reply)); f.user_input.value = payload;
  await f.context.sendMessage();
  assert.equal(f.chatbox.children.length, 2);
  assert.equal(f.chatbox.children[0].children[1].textContent, payload);
  assert.equal(f.chatbox.children[1].children[1].textContent, reply);
  assert.equal(f.user_input.value, '');
  assert.equal(f.chat_status.dataset.state, 'ready');
});

test('an in-flight request disables duplicate sends and restores controls', async () => {
  let resolve, calls = 0;
  const f = fixture(() => { calls++; return new Promise(done => { resolve = done; }); });
  f.user_input.value = 'preço'; const pending = f.context.sendMessage();
  assert.equal(f.send.disabled, true); assert.equal(f.user_input.disabled, true);
  await f.context.sendMessage(); assert.equal(calls, 1);
  resolve(response('499')); await pending;
  assert.equal(f.send.disabled, false); assert.equal(f.user_input.disabled, false);
});

test('failed requests preserve input and retry without duplicate user rows', async () => {
  let calls = 0;
  const f = fixture(async () => ++calls === 1 ? { ok: false } : response('Resposta'));
  f.user_input.value = 'garantia'; await f.context.sendMessage();
  assert.equal(f.user_input.value, 'garantia'); assert.equal(f.chat_status.dataset.state, 'error');
  await f.context.sendMessage(); assert.equal(f.chatbox.children.length, 2);
  assert.equal(f.chatbox.children[0].children[1].textContent, 'garantia');
});

test('clearing cancels the request and ignores a late response', async () => {
  let resolve, signal;
  const f = fixture((_url, options) => { signal = options.signal; return new Promise(done => { resolve = done; }); });
  f.user_input.value = 'preço'; const pending = f.context.sendMessage();
  f.context.clearChat(); assert.equal(signal.aborted, true);
  resolve(response('Late response')); await pending;
  assert.equal(f.chatbox.children.length, 1);
  assert.match(f.chatbox.children[0].children[1].textContent, /Olá/);
  assert.equal(f.user_input.value, '');
});
