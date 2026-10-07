// ComfyUI 화면 캡처(강사용) — 헤드리스 Chrome(CDP)으로 실제 실행 화면을 찍음. 결과 : 5-2/screens/*.png
//   node tools/screens/capture.js [sd15 flux qwen zimg swap]   (인자 없으면 전부, ComfyUI 가 127.0.0.1:8188 에 떠 있어야 함)
//   그 뒤 번호 표시 그림 : docker run --rm -v "<5-2/screens>:/s" -v "<5-2/tools/screens>:/w" ai4-lab/jupyter-clip python /w/annotate.py
//   Chrome 경로는 cdp.js 의 CHROME
const fs = require('fs');
const { launch, connect, sleep } = require('./cdp');
const REPO = require('path').resolve(__dirname, '../..');
const OUT = REPO + '/screens';
fs.mkdirSync(OUT, { recursive: true });
const WF = n => fs.readFileSync(`${REPO}/workflows/${n}`, 'utf8');
const W = 1920, H = 1080, DPR = 1.5;
const only = process.argv.slice(2);
const want = k => !only.length || only.includes(k);

(async () => {
  const { proc, ws } = await launch(9334, require('path').join(require('os').tmpdir(), 'comfy-capture-profile'));
  const c = await connect(ws);
  await c.send('Page.enable');
  await c.send('Emulation.setDeviceMetricsOverride', { width: W, height: H, deviceScaleFactor: DPR, mobile: false });
  await c.send('Page.navigate', { url: 'http://127.0.0.1:8188/' });
  for (let i = 0; i < 60; i++) { await sleep(500); try { if (await c.evalJs('!!(window.app && window.app.graph)')) break; } catch {} }
  await sleep(1500);
  await c.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await c.send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  await c.evalJs(`(async () => { const s = app.extensionManager.setting;
    await s.set('Comfy.VueNodes.Enabled', false); await s.set('Comfy.Execution.PreviewMethod', 'latent2rgb'); try { await s.set('Comfy.Minimap.Visible', false); } catch (e) {} return 1; })()`);
  await sleep(500);

  const load = async (file, name) => {
    await c.evalJs(`(async () => { await app.loadGraphData(${WF(file)}, true, true, ${JSON.stringify(name)}); return 1; })()`);
    await sleep(1200);
  };
  // 노드 id 목록을 화면에 맞춤. 반환 : 캔버스 → 화면 변환에 쓸 값
  const fit = async (ids, pad = 60, maxScale = 1.6) => c.evalJs(`(() => {
    const ns = app.graph._nodes.filter(n => ${JSON.stringify(ids)}.length === 0 || ${JSON.stringify(ids)}.includes(n.id));
    let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
    for (const n of ns) { x0 = Math.min(x0, n.pos[0]); y0 = Math.min(y0, n.pos[1] - 32); x1 = Math.max(x1, n.pos[0] + n.size[0]); y1 = Math.max(y1, n.pos[1] + n.size[1]); }
    const el = app.canvas.canvas.getBoundingClientRect();
    const cw = el.width, ch = el.height - 60;
    const s = Math.min((cw - 2 * ${pad}) / (x1 - x0), (ch - 2 * ${pad}) / (y1 - y0), ${maxScale});
    app.canvas.ds.scale = s;
    app.canvas.ds.offset[0] = (cw / s - (x1 - x0)) / 2 - x0;
    app.canvas.ds.offset[1] = (ch / s - (y1 - y0)) / 2 - y0 + 30 / s;
    app.canvas.setDirty(true, true); app.canvas.draw(true, true);
    return JSON.stringify({ s, left: el.left, top: el.top });
  })()`);
  const shot = async (name, clip) => { await sleep(700); await c.shot(`${OUT}/${name}.png`, clip || { x: 56, y: 40, width: W - 56, height: H - 40 }, DPR); console.log('saved', name); };
  const run = async (timeoutS = 600, onTick) => {
    const before = await c.evalJs(`fetch('/history?max_items=1').then(r => r.json()).then(h => Object.keys(h)[0] || '')`);
    await c.evalJs(`(async () => { await app.queuePrompt(0, 1); return 1; })()`);
    const t0 = Date.now();
    while ((Date.now() - t0) / 1000 < timeoutS) {
      await sleep(1000);
      if (onTick) await onTick((Date.now() - t0) / 1000);
      const q = await c.evalJs(`fetch('/queue').then(r => r.json()).then(q => q.queue_running.length + q.queue_pending.length)`);
      if (q === 0 && (Date.now() - t0) > 3000) break;
    }
    await sleep(1500);
    return (Date.now() - t0) / 1000;
  };
  const screenOf = async (nodeId) => JSON.parse(await c.evalJs(`(() => { const n = app.graph.getNodeById(${nodeId});
    const el = app.canvas.canvas.getBoundingClientRect(), ds = app.canvas.ds;
    const tx = x => (x + ds.offset[0]) * ds.scale + el.left, ty = y => (y + ds.offset[1]) * ds.scale + el.top;
    return JSON.stringify({ x: tx(n.pos[0]), y: ty(n.pos[1] - 32), w: n.size[0] * ds.scale, h: (n.size[1] + 32) * ds.scale,
      widgets: n.widgets.map(w => ({ name: w.name, y: ty(n.pos[1] + w.last_y + 10), x: tx(n.pos[0] + n.size[0] / 2) })) }); })()`));
  const click = async (x, y) => {
    for (const type of ['mousePressed', 'mouseReleased']) await c.send('Input.dispatchMouseEvent', { type, x, y, button: 'left', clickCount: 1 });
  };

  // ① SD1.5 전체 그래프 실행 후
  if (want('sd15')) {
    await load('SD1.5_t2i.json', 'SD1.5');
    await c.evalJs(`(() => { const n = app.graph._nodes.find(n => n.type === 'CLIPTextEncode'); n.widgets[0].value = 'a red car on the beach'; return 1; })()`);
    await fit([]);
    const sec = await run(300);
    await fit([]);
    console.log('sd15 run', sec.toFixed(1));
    await shot('c1_sd15_graph');
    // ② KSampler 확대 + 샘플러 목록
    const ks = await c.evalJs(`app.graph._nodes.find(n => n.type === 'KSampler').id`);
    await fit([ks], 120, 2.4);
    await shot('c2_ksampler');
    const p = await screenOf(ks);
    const w = p.widgets.find(x => x.name === 'sampler_name');
    await click(w.x, w.y); await sleep(800);
    await shot('c3_sampler_list', { x: 56, y: 40, width: W - 56, height: H - 40 });
    await c.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
    await click(30 + 56, H - 30); await sleep(300);
    const w2 = p.widgets.find(x => x.name === 'scheduler');
    await click(w2.x, w2.y); await sleep(800);
    await shot('c4_scheduler_list');
    await c.send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'Escape', code: 'Escape', windowsVirtualKeyCode: 27 });
  }
  // ③ FLUX · Qwen-Image 그래프 (실행 없이)
  if (want('flux')) { await load('FLUX.1-dev_t2i.json', 'FLUX'); await fit([]); await shot('c5_flux_graph'); }
  if (want('qwen')) { await load('Qwen-Image_t2i.json', 'Qwen-Image'); await fit([]); await shot('c6_qwen_graph'); }
  // ④ Z-Image 실행 중 미리보기 = 단계별 디노이즈
  if (want('zimg')) {
    await load('Z-Image_t2i.json', 'Z-Image');
    await c.evalJs(`(() => { const n = app.graph._nodes.find(n => n.type === 'KSampler'); n.widgets.find(w => w.name === 'seed').value = 11; n.size[1] += 520; return 1; })()`);
    const ks = await c.evalJs(`app.graph._nodes.find(n => n.type === 'KSampler').id`);
    await fit([ks], 30, 2.2);
    for (const f of fs.readdirSync(OUT)) if (f.startsWith('zseq_')) fs.unlinkSync(OUT + '/' + f);
    let k = 0, last = -10;
    const sec = await run(900, async t => {
      if (t - last >= 1.5) { last = t; const p = await screenOf(ks);
        await c.shot(`${OUT}/zseq_${String(++k).padStart(2, '0')}.png`, { x: p.x - 4, y: p.y - 4, width: p.w + 8, height: p.h + 8 }, 1); }
    });
    console.log('zimg run', sec.toFixed(1), 'frames', k);
    await fit([]); await shot('c7_zimage_done');
  }
  // ⑤ 조건 인코더 바꿔치기 오류(Qwen3.5-2B, 폭 2048)
  if (want('swap')) {
    await load('Z-Image_t2i.json', 'Z-Image swap');
    await c.evalJs(`(() => { const n = app.graph._nodes.find(n => n.type === 'CLIPLoader'); n.widgets.find(w => w.name === 'clip_name').value = 'qwen3.5_2b_bf16.safetensors'; return 1; })()`);
    await fit([]);
    await run(300);
    await sleep(1500);
    await shot('c8_encoder_swap_error');
    await c.evalJs(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.textContent.trim() === '자세히 보기'); if (b) b.click(); return !!b; })()`);
    await sleep(1500);
    await shot('c8b_encoder_swap_detail', { x: 0, y: 0, width: W, height: H });
  }
  c.close(); proc.kill();
})().catch(e => { console.error(e); process.exit(1); });
