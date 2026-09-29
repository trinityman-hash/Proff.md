import test from 'node:test';
import assert from 'node:assert/strict';
import worker, { encode, decode, predict, loadModel } from '../worker/index.js';
import fs from 'node:fs/promises';

const raw = JSON.parse(await fs.readFile(new URL('../public/model.json', import.meta.url), 'utf8'));
const tables = Array.from({length: raw.max_order + 1}, () => new Map());
for (const [k, ctx, token, count] of raw.transitions) {
  const key = JSON.stringify(ctx); let list = tables[k].get(key);
  if (!list) tables[k].set(key, list = []);
  list.push([token, count]);
}
const model = {...raw, tables};
const env = { ASSETS: { fetch: async () => new Response(JSON.stringify(raw), {status: 200}) } };

 test('tokenizer matches baseline behavior', () => {
  assert.deepEqual(encode("Hello, world's!"), ['hello', ',', "world's", '!']);
  assert.equal(decode(['hello', ',', 'world', '!']), 'hello, world!');
});
test('model produces a registered continuation', () => {
  assert.equal(predict(model, 'a deterministic engine').token, 'returns');
});
test('prediction ordering is deterministic', () => {
  assert.deepEqual(predict(model, 'the river').candidates, predict(model, 'the river').candidates);
});
test('unknown context returns controlled halt', () => {
  assert.equal(predict(model, 'zzunknownzz').status, 'NO_SUPPORTED_CONTINUATION');
});
test('model loads through the static asset binding', async () => {
  const loaded = await loadModel(env);
  assert.equal(loaded.tokenizer, 'drle-regex-en-v1');
  assert.ok(loaded.transitions.length > 0);
});
test('health and predict routes work', async () => {
  const health = await worker.fetch(new Request('https://example.test/health'), env);
  assert.equal(health.status, 200);
  assert.equal((await health.json()).status, 'ok');
  const response = await worker.fetch(new Request('https://example.test/predict', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({context:'a deterministic engine'})}), env);
  assert.equal(response.status, 200);
  assert.equal((await response.json()).token, 'returns');
});
test('invalid request is rejected', async () => {
  const response = await worker.fetch(new Request('https://example.test/predict', {method:'POST', headers:{'content-type':'application/json'}, body:'{}'}), env);
  assert.equal(response.status, 400);
});
test('generation respects token cap', async () => {
  const response = await worker.fetch(new Request('https://example.test/generate', {method:'POST', headers:{'content-type':'application/json'}, body:JSON.stringify({prompt:'a deterministic engine',max_tokens:4})}), env);
  assert.equal(response.status, 200);
  assert.ok((await response.json()).generated_tokens <= 4);
});
