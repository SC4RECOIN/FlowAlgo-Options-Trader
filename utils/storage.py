import sqlite3
from options_scraper import OptionEntry
from dataclasses import asdict


class SQLiteStorage(object):
    def __init__(self):
        self.con = sqlite3.connect("options-trader.db")
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS option_trades
            (
                id          TEXT    PRIMARY KEY     NOT NULL,
                symbol      TEXT                    NOT NULL,
                time        TEXT                            ,
                expiration  TEXT                    NOT NULL,
                strike      REAL                            ,
                side        TEXT                            ,
                spot        REAL                    NOT NULL,
                order_type  TEXT                            ,
                premium     REAL
            );
            """
        )

    def __enter__(self):
        return self

    def __exit__(self):
        self.con.commit()
        self.con.close()

    def insert_option(self, option: OptionEntry):
        option = asdict(option)
        h = hash(frozenset(option.items()))

        self.con.execute(
            f"""
            INSERT INTO option_trades (id,{','.join(option.keys())})
            VALUES ("{h}","{'","'.join([str(x) for x in option.values()])}")
            """
        )

    def query_options(self, query):
        cursor = conn.execute(query)
        return [row for row in cursor]
