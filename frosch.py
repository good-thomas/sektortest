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
            # Einzelabfrage ist oft stabiler gegen Format-Fehler
            df = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)
            if not df.empty:
                return df
        except Exception:
            time.sleep(2)
    return pd.DataFrame()

def get_hit_rate(prices, returns, months_back=None):
    """Berechnet die Trefferquote als reinen Zahlenwert."""
    try:
        if months_back:
            p_sub = prices.tail(months_back + 13) 
            r_sub = returns.tail(months_back)
        else:
            p_sub = prices
            r_sub = returns
        
        if len(p_sub) < 14: return 0.0
        
        # Momentum berechnen
        m9 = p_sub.pct_change(9)
        m12 = p_sub.pct_change(12)
        combined = (0.3 * m9 + 0.7 * m12)
        
        # Signal des Vormonats
        sig = combined.shift(1) > 0
        sig_aligned = sig.reindex(r_sub.index).fillna(False)
        
        valid_returns = r_sub[sig_aligned]
        
        if len(valid_returns) == 0: 
            return 0.0
            
        # Wir nutzen .values, um sicherzustellen, dass wir keine Pandas-Objekte mehr haben
        hits = np.sum(valid_returns.values > 0)
        total = len(valid_returns)
        
        return float(hits) / float(total)
    except:
        return 0.0

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
            
            # Spalten-Extraktion extrem sicher
            if 'Close' in data_raw.columns:
                prices = data_raw['Close']
            else:
                prices = data_raw.iloc[:, 0]
            
            # Sicherstellen, dass es eine Series ist, kein DataFrame
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            prices = prices.resample('ME').last().dropna()
            if len(prices) < 24: continue
            
            rets = prices.pct_change().dropna()

            # Scores berechnen
            h = get_hit_rate(prices, rets)
            r = get_hit_rate(prices, rets, 36)
            f = get_hit_rate(prices, rets, 18)

            # Status bestimmen
            status = "STABLE"
            if f > h and f >= 0.6: status = "HOT"
            elif f < 0.5: status = "WEAK"

            results.append({
                "ticker": str(ticker),
                "h_score": float(round(h, 2)),
                "r_score": float(round(r, 2)),
                "f_score": float(round(f, 2)),
                "status": str(status)
            })
        except Exception as e:
            print(f"Fehler bei {ticker}: {str(e)}")
            continue
            
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
