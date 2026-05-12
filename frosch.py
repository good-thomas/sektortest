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
    """Versucht Daten zu laden und wartet bei Rate Limits kurz."""
    for i in range(retries):
        try:
            # auto_adjust=True sorgt für saubere Kurse ohne 'Adj Close' Probleme
            df = yf.download(ticker, start="2010-01-01", progress=False, auto_adjust=True)
            if not df.empty:
                return df
        except Exception as e:
            if "Too Many Requests" in str(e):
                time.sleep(2)
                continue
    return pd.DataFrame()

@app.route("/audit", methods=["GET"])
def audit():
    ticker_str = request.args.get("tickers", "")
    if not ticker_str:
        return jsonify({"error": "Keine Ticker angegeben"}), 400
    
    ticker_list = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    results = []
    
    for ticker in ticker_list:
        try:
            data_raw = fetch_with_retry(ticker)
            
            if data_raw.empty:
                continue
            
            # WICHTIG: Sicherstellen, dass wir eine Series (Spalte) haben, nicht ein DataFrame
            if 'Close' in data_raw.columns:
                prices = data_raw['Close']
            else:
                prices = data_raw.iloc[:, 0]
            
            # Dein Fix: 'ME' statt 'M' für neue Pandas-Versionen
            prices = prices.resample('ME').last().dropna()
            
            # Konvertierung zu Series erzwingen, damit .autocorr() sicher existiert
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            if len(prices) < 24:
                continue
            
            rets = prices.pct_change().dropna()
            
            # Jetzt ist rets sicher eine Series -> .autocorr() funktioniert
            persistence_val = rets.autocorr(lag=1)
            
            std_val = rets.std()
            snr_val = (rets.mean() * 12) / (std_val * np.sqrt(12)) if std_val > 0 else 0
            
            mom9 = prices.pct_change(9)
            bull_signal = mom9.shift(1) > 0
            valid_months = rets[bull_signal]
            hit_rate_val = valid_months[valid_months > 0].count() / valid_months.count() if not valid_months.empty else 0
            
            status_text = "ROBUST" if persistence_val > 0.10 and snr_val > 0.4 else "HEKTISCH"
            if persistence_val < 0: status_text = "GEFÄHRLICH"

            results.append({
                "ticker": str(ticker),
                "persistence": float(round(persistence_val, 3)) if pd.notnull(persistence_val) else 0.0,
                "snr": float(round(snr_val, 3)) if pd.notnull(snr_val) else 0.0,
                "frosch_score": float(round(hit_rate_val, 2)),
                "status": str(status_text)
            })
        except Exception as e:
            # Protokolliert den Fehler im Render-Log, bricht aber nicht ab
            print(f"Fehler bei {ticker}: {e}")
            continue
            
    if not results:
        # Wenn Yahoo blockiert, geben wir eine klare Meldung an das WordPress-Frontend
        return jsonify({"error": "Yahoo Rate Limit. Bitte in 1 Min erneut versuchen."}), 429
        
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
