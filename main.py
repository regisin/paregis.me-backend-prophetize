from datetime import datetime, timedelta, date

import pandas as pd
import yfinance as yf

from prophet import Prophet

from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def download(symbols, start, end):
    data = yf.download(
        tickers = symbols,

        # use "period" instead of start/end
        # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        # (optional, default is '1mo')
        # period = "max",
        start=start,
        end=end,

        # fetch data by interval (including intraday if period < 60 days)
        # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        # (optional, default is '1d')
        interval = "1d",

        group_by = 'ticker',
        # auto_adjust = True,
        prepost = False,
        threads = 10,
        proxy = None
    )
    return data


class StockIn(BaseModel):
    symbol: str
    periods: Optional[int] = 10
    start: Optional[datetime] = datetime.fromtimestamp(0/1000.0)
    end: Optional[datetime] = datetime.now()

    class Config:
        schema_extra = {
            "symbol": "MSFT",
            "periods": 365,
            "start": "2021-07-10 00:00:00.000000",
            "end": "2021-07-20 00:00:00.000000"
        }


class StockOut(StockIn):
    symbol: str
    requested_at: datetime = datetime.now()
    forecast: dict

    class Config:
        schema_extra = {
            "symbol": "MSFT",
            "forecast": {
                "ds": [0],
                "y": [0],
                "yhat": [0],
                "yhat_upper": [0],
                "yhat_lower": [0],
            }
        }


@app.post("/prophetize", response_model=StockOut, status_code=200)
async def prophetize(payload: StockIn):
    requested_at = datetime.now()

    symbol = payload.symbol
    periods = payload.periods
    start = payload.start
    end = payload.end
    
    future = end + timedelta(days=periods)

    df = download(symbol, start, end)

    begining_of_time = df.index.min()

    df.reset_index(inplace=True)
    df['ds'] = df['Date']
    df['y'] = df['Adj Close']
    df = df[["ds", "y"]]

    # df = df[df.ds <= today]

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    print(forecast[['ds', 'yhat']])

    return {
            "symbol": "MSFT",
            "requested_at": requested_at,
            "forecast": {
                "ds": forecast['ds'],
                "y": df['y'],
                "yhat": forecast['yhat'],
                "yhat_upper": forecast['yhat_upper'],
                "yhat_lower": forecast['yhat_lower'],
            }
        }

