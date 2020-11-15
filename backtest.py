import pandas as pd
import os
import arrow
from tqdm import tqdm


def clean_df(df, use_cache=True):
    """
    Convert time and expiry from various formats and convert premium to number

    df prior to cleaning:
    =====================
    Date                            Time Ticker    Expiry Strike   C/P    Spot   Qty     Price   Type Volume       OI   Premium   Sector Unusual
    0  6/13/17  2017-06-13T22:58:24.752Z   NTNX  10/17(M)     20   Put   17.97   701  4.000000  SWEEP    701   3689.0  $280,400  ETF/ETN    True
    1  6/13/17  2017-06-13T22:54:53.948Z     KO  06/30/17     45  Call   45.01   769  0.430000  SWEEP   1228    832.0   $33,067  ETF/ETN   False
    2  6/13/17  2017-06-13T22:53:32.255Z     MU  06/16/17     32  Call   31.49   912  0.357774  SWEEP  17065  14769.0   $32,628  ETF/ETN   False
    3  6/13/17  2017-06-13T22:51:41.968Z     PE  07/17(M)     30  Call   28.61  2000  0.698850  SWEEP   2259   3054.0  $139,769  ETF/ETN   False
    4  6/13/17  2017-06-13T22:49:01.261Z    SPY  07/07/17  244.5  Call  244.45   908  1.270000  SWEEP   1932   1684.0  $115,316  ETF/ETN   False
    """
    print("cleaning and converting df")
    dates, expiries, rows, premium = [], [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # time had date in 2017 pulls
            if row.Time[:4] == "2017":
                date = arrow.get(arrow.get(row.Time))
            # combine date and time
            else:
                s = [int(x) for x in row.Date.split("/")]
                date = f"20{s[2]}-{s[0]:02}-{s[1]:02}"
                date = arrow.get(date)
                date = f"{date.format('YYYY-MM-DD')}T{row.Time}"
                date = arrow.get(date)

            # only month provided
            if row.Expiry[-3:] == "(M)":
                expiry = arrow.get(row.Expiry[:5], "MM/YY")
            elif row.Expiry[2] == "/":
                s = [int(x) for x in row.Date.split("/")]
                ds = f"20{s[2]}-{s[0]:02}-{s[1]:02}"
                expiry = arrow.get(ds)
            else:
                expiry = arrow.get(row.Expiry)

            premium.append(int(row.Premium[1:].replace(",", "")))
            dates.append(date)
            expiries.append(expiry)
            rows.append(row)
        except:
            pass

    df = pd.DataFrame(rows)
    df["Time"] = dates
    df["Expiry"] = expiries
    df["Premium"] = premium

    # sort by time
    index = df.index
    df = df.sort_values("Time")
    df.index = index

    df.to_pickle("hist_options.pkl")
    return df


# import data
data_dir = "hist_data"
frames = [pd.read_csv(f"{data_dir}/{filename}") for filename in os.listdir(data_dir)]
df = pd.concat(frames)
df = clean_df(df)
print("entries:", len(df))
print(df.head())