const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const token = 'unit-test-learning-key-for-local-fixtures-only';

function fixture(fetch) {
  const ids = Object.fromEntries(['teach_form','teach_status','save_teach','operator_key','intent_tag','intent_pattern','intent_response','teach'].map(id => [id, {
    value:'',dataset:{},attrs:{},focus(){},setAttribute(key,value){this.attrs[key]=value;},
  }]));
  ids.teach_form.reportValidity = () => true;
  ids.teach_form.reset = () => { for(const id of ['operator_key','intent_tag','intent_pattern','intent_response']) ids[id].value=''; };
  ids.operator_key.value=token;
  ids.intent_tag.value='hours';ids.intent_pattern.value='Horário?';ids.intent_response.value='Das 9 às 18.';
  const context = {document:{getElementById:id=>ids[id],body:{dataset:{}}},fetch,AbortController,setTimeout,clearTimeout};
  Object.defineProperty(context,'localStorage',{get(){throw new Error('Keys must not reach storage');}});
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/js/teacher.js'),'utf8'),context);
  return {context,...ids};
}

test('operator key is sent only in Authorization and removed from the form',async()=>{
  let request;
  const f=fixture(async(_url,options)=>{request=options;return{ok:true,json:async()=>({changed:true})};});
  await f.context.saveIntent();
  assert.equal(request.headers.Authorization,'Bearer '+token);
  assert.ok(!request.body.includes(token));
  assert.equal(f.operator_key.value,'');
  assert.equal(f.intent_pattern.value,'');
  assert.match(f.teach_status.textContent,/cadastrada/);
});

test('failed authorization clears the key and preserves the proposed intent',async()=>{
  const f=fixture(async()=>({ok:false,status:401,json:async()=>({error:'irrelevant'})}));
  await f.context.saveIntent();
  assert.equal(f.operator_key.value,'');
  assert.equal(f.intent_pattern.value,'Horário?');
  assert.equal(f.teach_status.dataset.state,'error');
  assert.match(f.teach_status.textContent,/chave/);
});

test('parallel submissions are blocked and late responses cannot modify a closed editor',async()=>{
  let resolve,options,calls=0;
  const f=fixture((_url,request)=>{options=request;calls++;return new Promise(done=>resolve=done);});
  const pending=f.context.saveIntent();
  await f.context.saveIntent();assert.equal(calls,1);
  assert.equal(f.save_teach.disabled,true);
  f.context.resetTeacher();
  assert.equal(options.signal.aborted,true);
  const previous=f.teach_status.textContent;
  resolve({ok:true,json:async()=>({changed:true})});await pending;
  assert.equal(f.teach_status.textContent,previous);
  assert.equal(f.save_teach.disabled,false);
  assert.equal(f.operator_key.value,'');
});

test('invalid tags do not send network requests',async()=>{
  let calls=0;const f=fixture(async()=>{calls++;});f.intent_tag.value='Inválida';
  await f.context.saveIntent();assert.equal(calls,0);
  assert.equal(f.teach_status.dataset.state,'error');
});
