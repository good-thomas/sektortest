import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time

app = Flask(__name__)
CORS(app)

def fetch_with_retry(ticker, retries=3):
    for i in range(retries):
        try:
            df = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)
            if not df.empty:
                return df
        except Exception:
            time.sleep(2)
            continue
    return pd.DataFrame()

def get_hit_rate(prices, returns, months_back=None):
    """Berechnet die Trefferquote für die 30/70 Regel in einem Zeitfenster."""
    # Wir brauchen Puffer für das 12M Momentum
    if months_back:
        p_sub = prices.tail(months_back + 13) 
        r_sub = returns.tail(months_back)
    else:
        p_sub = prices
        r_sub = returns
    
    if len(p_sub) < 14: return 0.0
    
    m9 = p_sub.pct_change(9)
    m12 = p_sub.pct_change(12)
    combined = (0.3 * m9 + 0.7 * m12)
    
    # Signal des Vormonats für den aktuellen Monat
    sig = combined.shift(1) > 0
    # Abgleich der Längen
    sig = sig.tail(len(r_sub))
    valid = r_sub[sig]
    
    if valid.empty: return 0.0
    return float(valid[valid > 0].count() / valid.count())

@app.route("/audit", methods=["GET"])
def audit():
    ticker_str = request.args.get("tickers", "")
    if not ticker_str:
        return jsonify({"error": "No tickers"}), 400
    
    ticker_list = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    results = []
    
    for ticker in ticker_list:
        try:
            data_raw = fetch_with_retry(ticker)
            if data_raw.empty: continue
            
            prices = data_raw['Close'] if 'Close' in data_raw.columns else data_raw.iloc[:, 0]
            prices = prices.resample('ME').last().dropna()
            if len(prices) < 24: continue
            
            rets = prices.pct_change().dropna()

            # Die 3 Frosch-Ebenen
            h_score = get_hit_rate(prices, rets)          # Gesamt (seit 2010)
            r_score = get_hit_rate(prices, rets, 36)      # Recent (3 Jahre)
            f_score = get_hit_rate(prices, rets, 18)      # Fresh (1.5 Jahre)

            # Dynamischer Status
            status = "HOT" if f_score > h_score and f_score >= 0.6 else "STABLE"
            if f_score < 0.5: status = "WEAK"

            results.append({
                "ticker": ticker,
                "h_score": round(h_score, 2),
                "r_score": round(r_score, 2),
                "f_score": round(f_score, 2),
                "status": status
            })
        except Exception as e:
            print(f"Error {ticker}: {e}")
            continue
            
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
