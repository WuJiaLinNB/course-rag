"""离线 embedder：语料 → chunk → embedding API → data/vectors.bin + data/chunks.json

用法：
    python tools/embedder.py --corpus corpus/ --out data/ \
        --course <课程名> --semester <学期> [--type 讲义] \
        [--base-url <url> --model <模型名> --api-key <key>]

--base-url / --model / --api-key 缺省时从环境变量读取：
    EMBED_API_BASE   OpenAI 兼容 embedding 服务地址（如 https://api.example.com/v1）
    EMBED_MODEL      embedding 模型名
    EMBED_API_KEY    API key

手工验证步骤（需要真实 API key，由用户执行）：
    1. 配好上述环境变量后运行脚本，确认输出 "embedded N chunks, dim=D"；
    2. 校验 data/vectors.bin 文件大小 == 16 + N*D*4 字节
       （16 字节头：magic "CRV1" + version(=1) + dim + count，均为小端）；
    3. 校验 data/chunks.json 数组长度 == N，且第 i 条 content 与 vectors.bin
       的第 i 个向量一一对应；
    4. 用 C++ 引擎加载该 data/ 目录跑一次检索，确认 VectorStore::load 不报错。
"""
import argparse, json, struct, os
import requests


def split_by_heading(text, max_chars=800):
    """按 Markdown 标题切分，超长段落再按 max_chars*2 边界二次切。"""
    chunks, cur = [], ""
    for line in text.splitlines():
        if line.startswith("#") and cur.strip():
            chunks.append(cur.strip()); cur = line + "\n"
        else:
            cur += line + "\n"
            if len(cur) >= max_chars * 2:
                chunks.append(cur.strip()); cur = ""
    if cur.strip(): chunks.append(cur.strip())
    return chunks


def embed_texts(texts, api_key, base_url, model):
    out = []
    for i in range(0, len(texts), 64):        # 批量 64
        batch = texts[i:i+64]
        r = requests.post(f"{base_url}/embeddings",
                          headers={"Authorization": f"Bearer {api_key}"},
                          json={"model": model, "input": batch}, timeout=60)
        r.raise_for_status()
        out.extend(d["embedding"] for d in r.json()["data"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", default="data")
    ap.add_argument("--base-url", default=os.environ.get("EMBED_API_BASE"))
    ap.add_argument("--model", default=os.environ.get("EMBED_MODEL"))
    ap.add_argument("--api-key", default=os.environ.get("EMBED_API_KEY"))
    ap.add_argument("--course", required=True)
    ap.add_argument("--semester", required=True)
    ap.add_argument("--type", default="讲义", dest="type_")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    chunks, metas = [], []
    for fn in sorted(os.listdir(a.corpus)):
        if not fn.endswith((".md", ".txt")): continue
        text = open(os.path.join(a.corpus, fn), encoding="utf-8").read()
        for c in split_by_heading(text):
            chunks.append(c)
            metas.append({"course": a.course, "semester": a.semester,
                          "type": a.type_, "title": fn, "content": c})
    vecs = embed_texts(chunks, a.api_key, a.base_url, a.model)
    dim = len(vecs[0])
    with open(os.path.join(a.out, "vectors.bin"), "wb") as f:
        # C++ 契约（core/src/vector_store.cpp 的 load()）：16 字节文件头 =
        #   magic "CRV1"(4B) + version(uint32 LE，必须为 1) + dim(uint32 LE) + count(uint32 LE)，
        #   之后是 count*dim 个 float32 LE；load() 还会校验文件总长 == 16 + count*dim*4。
        f.write(b"CRV1")                                  # magic
        f.write(struct.pack("<III", 1, dim, len(vecs)))   # version=1, dim, count（小端）
        for v in vecs:
            f.write(struct.pack(f"<{dim}f", *v))
    json.dump(metas, open(os.path.join(a.out, "chunks.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"embedded {len(chunks)} chunks, dim={dim}")


# 过关自测：
# 1) 为什么 vectors.bin 和 chunks.json 必须行号对应？
#    C++ 检索只返回向量下标 i，引擎用同一个 i 去 chunks.json 里取
#    course/semester/title/content。两者顺序一旦错位，"向量与文本"就对不上，
#    每条搜索结果都会答非所问。本脚本在同一次循环里同步 append chunks 与
#    metas，并按 chunks 顺序分批调 API（OpenAI 兼容接口的 data 顺序与 input
#    一致），因此下标天然对齐；任何一侧单独重排都会破坏这个约定。
# 2) magic "CRV1" 是防什么的？
#    防止把错误的文件喂给加载器：格式不对、别的二进制、截断/损坏的文件，
#    会在 load() 把一堆垃圾字节当 float 读进来之前，被 magic + version +
#    文件长度自检当场拒绝并给出明确报错，而不是悄悄产出错误的检索结果。

if __name__ == "__main__":
    main()
