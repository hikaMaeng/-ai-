"""실행한 노트북(_out)의 셀별 글 출력을 보여 주고 그림을 _out/img/ 에 꺼낸다(강사용 검증)

    python tools/show_out.py 01
"""
import base64, json, sys
from pathlib import Path

nb_dir = Path(__file__).resolve().parents[1] / "notebooks" / "_out"
n = sys.argv[1]
p = next(nb_dir.glob(f"{n}_*.ipynb"))
img = nb_dir / "img"; img.mkdir(exist_ok=True)
nb = json.loads(p.read_text(encoding="utf-8"))
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    head = "".join(c["source"]).splitlines()[0]
    print(f"\n##### {head}")
    k = 0
    for o in c.get("outputs", []):
        if o.get("output_type") == "stream":
            print("".join(o["text"])[-4000:], end="")
        elif o.get("output_type") == "error":
            print("ERROR", o["ename"], o["evalue"][:600])
        elif "data" in o:
            if "image/png" in o["data"]:
                k += 1
                f = img / f"{n}_{head.split('·')[0].strip('# ').replace(' ', '')}_{k}.png"
                f.write_bytes(base64.b64decode(o["data"]["image/png"]))
                print(f"[그림] {f.name}")
            elif "text/plain" in o["data"] and "Audio" not in "".join(o["data"]["text/plain"]):
                print("".join(o["data"]["text/plain"])[-2500:])
