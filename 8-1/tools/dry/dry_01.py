from dry_common import make_dry_llm


def install(mod):
    mod.LLM = make_dry_llm(mod)
