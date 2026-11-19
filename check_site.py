"""Exercise mobile language switching and explicit, single-model download."""

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import expect, sync_playwright


class StaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def check(url):
    with sync_playwright() as browser_api:
        browser = browser_api.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 390, "height": 844}, has_touch=True, is_mobile=True)
        page.add_init_script("""const send = Worker.prototype.postMessage;
            Worker.prototype.postMessage = function (message, ...args) {
                if (message.type === 'play') window.__speechSpeed = message.speed;
                return send.call(this, message, ...args);
            };""")
        requests, errors = [], []
        page.on("request", lambda req: requests.append((req.method, req.url)))
        page.on("pageerror", lambda error: errors.append(str(error)))
        assert page.goto(url).ok
        language = page.locator("#language")
        expect(language).to_be_enabled()
        released = page.request.get(url + "languages.json").json()
        active = {"bn", *Path(__file__).with_name("release-languages.txt").read_text().split()}
        assert {locale["code"] for locale in released} == active
        options = language.locator("option").evaluate_all("options => options.map(option => option.value)")
        assert options == [locale["code"] for locale in released]
        deep_link = released[-1]["code"]
        assert page.goto(f"{url}?lang={deep_link}").ok
        expect(language).to_have_value(deep_link)
        for locale in released:
            code = locale["code"]
            before = len(requests)
            language.select_option(code)
            expect(page.locator("#language-name")).to_have_text(locale["name"])
            expect(page.locator("#voice-name")).to_have_text(locale["voice"])
            assert parse_qs(urlparse(page.url).query)["lang"] == [code]
            for field, selector in (("datasetCard", "#dataset-card"), ("modelCard", "#model-card")):
                link = page.locator(selector)
                if locale.get(field):
                    expect(link).to_be_visible()
                    expect(link).to_have_attribute("href", locale[field])
                else:
                    expect(link).to_be_hidden()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not any("weights.gz" in req for _, req in requests[before:])
            load = page.locator("#load")
            if not locale["models"]:
                expect(load).to_be_disabled()
                continue
            choice = locale["models"][0]
            evaluation = choice.get("evaluation") or choice.get("preview")
            if evaluation:
                expect(page.locator("#size")).to_contain_text(f'{evaluation["mos"]:.3f} predicted MOS')
            load.click()
            expect(page.locator("#model-status")).to_contain_text("is ready.", timeout=45000)
            matches = [request for _, request in requests[before:] if request.endswith("weights.gz")]
            assert len(matches) == 1, matches

            examples = [locale["examples"][0], locale["examples"][2]]
            if code == "bn":
                examples.append("ami bari jete chai, kaal jabo. Na December a jabo bhabchhi.")
            for sentence in examples:
                page.locator("#text").fill(sentence)
                previous_audio = page.locator("#audio").get_attribute("src")
                page.locator("#play").click()
                page.wait_for_function("old => { const a = document.querySelector('audio'); return a.src && a.src !== old && a.duration > 0 && a.currentTime > .09; }", arg=previous_audio, timeout=45000)
                assert page.evaluate("window.__speechSpeed") == choice.get("paceScale", 1)
                print(code, choice["id"], round(page.locator("#audio").evaluate("audio => audio.duration"), 2), sentence, flush=True)
        credits = page.get_by_role("link", name="Model credits")
        credits.click()
        expect(page.get_by_role("heading", name="Models and training data")).to_be_visible()
        expect(page.get_by_role("link", name="Dataset and model cards")).to_be_visible()
        assert all(method == "GET" for method, _ in requests), requests
        assert not errors, errors
        browser.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination")
    args = parser.parse_args()
    if args.destination.startswith(("http://", "https://")):
        check(args.destination.rstrip("/") + "/")
    else:
        handler = partial(StaticHandler, directory=Path(args.destination))
        with ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
            Thread(target=server.serve_forever, daemon=True).start()
            try:
                check(f"http://127.0.0.1:{server.server_port}/")
            finally:
                server.shutdown()
