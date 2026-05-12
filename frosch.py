import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) # Wichtig, damit WordPress mit Render kommunizieren darf

@app.route("/audit", methods=["GET"])
def audit():
    # Ticker aus der URL lesen, z.B. /audit?tickers=SMH,URA,GDX
    ticker_str = request.args.get("tickers", "")
    if not ticker_str:
        return jsonify({"error": "Keine Ticker angegeben"}), 400
    
    ticker_list = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    
    try:
        # Daten laden
        data = yf.download(ticker_list, start="2010-01-01", progress=False)['Adj Close']
        if len(ticker_list) == 1:
            data = data.to_frame(name=ticker_list[0])
        
        data = data.resample('M').last()
        results = []
        
        for ticker in ticker_list:
            if ticker not in data.columns: continue
            prices = data[ticker].dropna()
            if len(prices) < 24: continue
            
            rets = prices.pct_change().dropna()
            persistence = rets.autocorr(lag=1)
            snr = (rets.mean() * 12) / (rets.std() * np.sqrt(12))
            
            mom9 = prices.pct_change(9)
            bull_signal = mom9.shift(1) > 0
            valid_months = rets[bull_signal]
            hit_rate = valid_months[valid_months > 0].count() / valid_months.count() if valid_months.count() > 0 else 0
            
            status = "ROBUST" if persistence > 0.10 and snr > 0.4 else "HEKTISCH"
            if persistence < 0: status = "GEFÄHRLICH"

            results.append({
                "ticker": ticker,
                "persistence": round(persistence, 3),
                "snr": round(snr, 3),
                "frosch_score": round(hit_rate, 2),
                "status": status
            })
        
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Port wird von Render vorgegeben
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
