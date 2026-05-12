import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route("/audit", methods=["GET"])
def audit():
    ticker_str = request.args.get("tickers", "")
    if not ticker_str:
        return jsonify({"error": "Keine Ticker angegeben"}), 400
    
    ticker_list = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    
    try:
        # Daten laden
        raw_data = yf.download(ticker_list, start="2010-01-01", progress=False)
        
        if isinstance(raw_data.columns, pd.MultiIndex):
            data = raw_data['Adj Close']
        else:
            if 'Adj Close' in raw_data.columns:
                data = raw_data[['Adj Close']]
                data.columns = ticker_list
            else:
                data = raw_data
        
        data = data.resample('M').last()
        results = []
        
        for ticker in ticker_list:
            if ticker not in data.columns:
                continue
                
            prices = data[ticker].dropna()
            if len(prices) < 24:
                continue
            
            rets = prices.pct_change().dropna()
            persistence = rets.autocorr(lag=1)
            
            std = rets.std()
            snr = (rets.mean() * 12) / (std * np.sqrt(12)) if std > 0 else 0
            
            mom9 = prices.pct_change(9)
            bull_signal = mom9.shift(1) > 0
            valid_months = rets[bull_signal]
            
            hit_rate = 0
            if not valid_months.empty:
                hit_rate = valid_months[valid_months > 0].count() / valid_months.count()
            
            status = "ROBUST" if persistence > 0.10 and snr > 0.4 else "HEKTISCH"
            if persistence < 0: status = "GEFÄHRLICH"

            results.append({
                "ticker": ticker,
                "persistence": round(float(persistence), 3) if not np.isnan(persistence) else 0,
                "snr": round(float(snr), 3) if not np.isnan(snr) else 0,
                "frosch_score": round(float(hit_rate), 2),
                "status": status
            })
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
