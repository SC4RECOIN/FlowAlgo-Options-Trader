import sqlite3
from utils.options_scraper import OptionEntry
import arrow
from dataclasses import asdict


class SQLiteStorage(object):
    def __init__(self):
        self.con = sqlite3.connect("options-trader.db")
        self.con.execute(
            """
            CREATE TABLE IF NOT EXISTS option_trades
            (
                id          TEXT    PRIMARY KEY     NOT NULL,
                date        DATE                            ,
                qty         INT                     NOT NULL,
                exited      BOOL                    NOT NULL,
                symbol      TEXT                    NOT NULL,
                time        TEXT                            ,
                expiration  DATE                    NOT NULL,
                strike      REAL                            ,
                side        TEXT                            ,
                spot        REAL                    NOT NULL,
                order_type  TEXT                            ,
                premium     REAL
            );
            """
        )

    def __enter__(self):
        self.con = sqlite3.connect("options-trader.db")
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.con.commit()
        self.con.close()

    def insert_option(self, option: OptionEntry, qty: int):
        try:
            option = asdict(option)
            h = hash(frozenset(option.items()))
            date = arrow.now().isoformat()

            q = f"""
                INSERT INTO option_trades (id,date,qty,exited,{','.join(option.keys())})
                VALUES ("{h}","{date}",{qty},false,"{'","'.join([str(x) for x in option.values()])}")
                """
            self.con.execute(q)
        except Exception as e:
            print(f"Error inserting to db: {e}\n{q}")

    def get_expired_positions(self):
        query = f"""
            SELECT * FROM option_trades
            WHERE exited = false AND expiration <= DATE('now');
            """
        cursor = self.con.execute(query)
        return [row for row in cursor]

    def mark_exited(self, option_id: str):
        self.con.execute(
            f"""
            UPDATE option_trades
            SET exited = true
            WHERE id = '{option_id}';
            """
        )
