"""The shell the four screens are served from.

These cover the serving layer and nothing about the analysis: that each screen URL
answers with the one page, that the root points at the first screen, and that the
browser's own files are reachable at the paths the page asks for. A screen that
answers 404 here would fail in a browser as a blank tab, which is the failure worth
catching before anything is drawn into it.
"""

from __future__ import annotations

from app.main import SCREENS


def test_the_root_sends_the_browser_to_the_first_screen(client):
    """A bare localhost is the address someone types, so it has to arrive somewhere."""
    response = client.get("/", follow_redirects=False)

    assert response.status_code in (307, 308)
    assert response.headers["location"] == "/data"


def test_every_screen_url_answers_with_the_shell(client):
    """One page, four URLs. The client routes between them on ``location.pathname``."""
    pages = {screen: client.get(f"/{screen}") for screen in SCREENS}

    for screen, response in pages.items():
        assert response.status_code == 200, f"/{screen} did not answer"
        assert response.headers["content-type"].startswith("text/html")

    # Identical bodies: a diverging screen URL means a second page was written that
    # nothing routes to, which is the thing the one-shell design is meant to prevent.
    assert len({response.text for response in pages.values()}) == 1


def test_every_screen_is_named_in_the_rail(client):
    """The rail is the whole navigation, so a screen missing from it is unreachable."""
    shell = client.get("/data").text

    for screen in SCREENS:
        assert f'href="/{screen}"' in shell


def test_the_front_end_files_are_served(client):
    """The page names its stylesheet and its script, so both have to resolve."""
    for path in ("/static/style.css", "/static/app.js"):
        response = client.get(path)

        assert response.status_code == 200, f"{path} did not answer"
        assert response.text.strip(), f"{path} is empty"


def test_an_unknown_screen_is_not_a_page(client):
    """Only the four screens are served, so a typo in the address bar is a 404."""
    assert client.get("/heatmaps").status_code == 404
