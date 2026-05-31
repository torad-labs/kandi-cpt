"""News articles (edm.com, Dancing Astronaut, Billboard, Vegas press) — trafilatura + LLM rescue.

Skip search/listing artifacts (they leak as high-anchor pages anyway, but skip early to save a
needless rescue call)."""
from . import html_article


def clean(artifact_path, meta, *, llm_rescue):
    if artifact_path.stem.startswith("search-"):
        return None
    return html_article.clean_html(artifact_path, meta, llm_rescue_on=llm_rescue,
                                    fallback_title=artifact_path.stem.replace("-", " "))
