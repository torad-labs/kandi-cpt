"""Music Festival Wizard — lineup/dates/venue reference pages (server-rendered tables)."""
from . import html_article


def clean(artifact_path, meta, *, llm_rescue):
    return html_article.clean_html(artifact_path, meta, llm_rescue_on=llm_rescue,
                                   fallback_title=artifact_path.stem.replace("-", " "))
