import os
import asyncio
import arrow
from pyppeteer import launch
from dataclasses import dataclass


@dataclass
class OptionEntry:
    symbol: str
    time: str
    expiration: str
    strike: float
    side: str
    spot: float
    order_type: str
    premium: int


class Scraper(object):
    def __init__(self):
        self.email = os.environ["FLOW_EMAIL"]
        self.password = os.environ["FLOW_PASS"]
        self.page = None

    async def login(self):
        browser = await launch(headless=True)
        self.page = await browser.newPage()
        await self.page.setUserAgent(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36"
        )
        await self.page.setViewport({"width": 1920, "height": 1080})
        await self.page.goto("https://app.flowalgo.com/users/login")

        # enter details and submit
        await self.page.type("input[name=amember_login]", self.email)
        await self.page.type("input[name=amember_pass]", self.password)
        await self.page.click("input[type=submit]")

        await self.page.waitForNavigation()

    async def get_options(self):
        data = await self.page.evaluate(
            """() => {
            return [
                'ticker',
                'strike',
                'time',
                'expiry',
                'contract-type',
                'details',
                'type',
                'premium',
                'ref',
            ].map(item =>
                Array.from(
                    document.getElementsByClassName(item),
                    e => e.innerText
                ).slice(1)
            )}"""
        )

        def parse_premium(premium: str) -> float:
            value = float(premium[1:-1])
            return value * 1e6 if premium[-1] == "M" else value * 1e3

        def parse_expiry(exp: str) -> str:
            if exp[2] == "/":
                s = [int(x) for x in exp.split("/")]
                ds = f"20{s[2]}-{s[0]:02}-{s[1]:02}"
                return arrow.get(ds).format("YYYY-MM-DD")

            return arrow.get(exp).format("YYYY-MM-DD")

        def is_today(time):
            """
            Options with time > now were from yesterday
            """
            now = arrow.now()
            option_time = arrow.get(
                f"{now.format('YYYY-MM-DD')} {time}", "YYYY-MM-DD HH:mm A"
            )
            return now > option_time

        options = []
        for entry in zip(*data):
            try:
                if is_today(entry[2]):
                    options.append(
                        OptionEntry(
                            symbol=entry[0],
                            time=entry[2],
                            expiration=parse_expiry(entry[3]),
                            strike=float(entry[1]),
                            side=entry[4],
                            spot=float(entry[8]),
                            order_type=entry[6],
                            premium=parse_premium(entry[7]),
                        )
                    )
            except Exception as e:
                print(f"failed to parse option: {e}")

        return options


if __name__ == "__main__":
    from dotenv import load_dotenv

    complete = lambda f: asyncio.get_event_loop().run_until_complete(f)

    load_dotenv()
    scraper = Scraper()
    complete(scraper.login())
    complete(scraper.get_options())
