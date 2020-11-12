import os
import asyncio
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
    premium: float


class Scraper(object):
    def __init__(self):
        self.email = os.environ["FLOW_EMAIL"]
        self.password = os.environ["FLOW_PASS"]
        self.page = None

    async def login(self):
        browser = await launch(headless=False)
        self.page = await browser.newPage()
        await self.page.goto("https://app.flowalgo.com/users/login")
        await self.page.setViewport({'width': 1920, 'height': 1080})

        # enter details and submit
        await self.page.type("input[name=amember_login]", self.email)
        await self.page.type("input[name=amember_pass]", self.password)
        await self.page.click("input[type=submit]")

        await self.page.waitForNavigation()

    async def get_options(self):
        data = await self.page.evaluate("""() => {
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
            return value * 1e6 if premium[-1] == 'M' else value * 1e3

        options = []
        for entry in zip(*data):
            options.append(OptionEntry(
                symbol=entry[0],
                time=entry[2],
                expiration=entry[3],
                strike=float(entry[1]),
                side=entry[4],
                spot=float(entry[8]),
                order_type=entry[6],
                premium=parse_premium(entry[7])
            ))
        
        print(options)
        return options


if __name__ == "__main__":
    from dotenv import load_dotenv
    complete = lambda f: asyncio.get_event_loop().run_until_complete(f)

    load_dotenv()  
    scraper = Scraper()
    complete(scraper.login())
    complete(scraper.get_options())
