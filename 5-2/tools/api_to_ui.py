"""API 형식 워크플로(노드 번호 → class_type · inputs) → ComfyUI 화면용 형식(nodes · links · 위치)

    from api_to_ui import to_ui
    ui = to_ui(api_graph, object_info)      # object_info = GET /object_info

노드 위치는 연결 깊이로 열을 나눠 왼쪽 → 오른쪽으로 놓는다.
"""
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
SEED_INPUTS = {"seed", "noise_seed"}          # 화면은 시드 뒤에 '생성 후 제어' 값을 하나 더 둔다


def _spec(oi, cls):
    d = oi[cls]["input"]
    order = list(d.get("required", {}).items()) + list(d.get("optional", {}).items())
    return order, oi[cls]["output"], oi[cls].get("output_name", oi[cls]["output"])


def _is_widget(t):
    return isinstance(t, list) or t in WIDGET_TYPES


def to_ui(g, oi):
    ids = {k: i + 1 for i, k in enumerate(g)}
    depth = {}
    def dep(k):
        if k not in depth:
            ups = [v[0] for v in g[k]["inputs"].values() if isinstance(v, list) and len(v) == 2 and isinstance(v[0], str)]
            depth[k] = 1 + max((dep(u) for u in ups), default=-1)
        return depth[k]
    for k in g: dep(k)
    col_y, nodes, links, out_links = {}, [], [], {}
    for k, v in g.items():
        order, outs, out_names = _spec(oi, v["class_type"])
        inputs, widgets = [], []
        for name, (t, *_) in order:
            val = v["inputs"].get(name)
            if _is_widget(t):
                if name in v["inputs"]:
                    widgets.append(val)
                    if name in SEED_INPUTS: widgets.append("fixed")
                continue
            link = None
            if isinstance(val, list):
                link = len(links) + 1
                src, slot = ids[val[0]], val[1]
                links.append([link, src, slot, ids[k], len(inputs), t])
                out_links.setdefault((src, slot), []).append(link)
            inputs.append({"name": name, "type": t, "link": link})
        d = depth[k]
        y = col_y.get(d, 40)
        h = 60 + 26 * (len(inputs) + len(widgets) + len(outs))
        col_y[d] = y + h + 40
        nodes.append({"id": ids[k], "type": v["class_type"], "pos": [40 + 340 * d, y], "size": [300, h], "flags": {},
                      "order": len(nodes), "mode": 0, "inputs": inputs,
                      "outputs": [{"name": n, "type": t, "links": [], "slot_index": i} for i, (t, n) in enumerate(zip(outs, out_names))],
                      "properties": {"Node name for S&R": v["class_type"]}, "widgets_values": widgets})
    for n in nodes:
        for o in n["outputs"]:
            o["links"] = out_links.get((n["id"], o["slot_index"]), [])
    return {"last_node_id": len(nodes), "last_link_id": len(links), "nodes": nodes, "links": links,
            "groups": [], "config": {}, "extra": {}, "version": 0.4}
