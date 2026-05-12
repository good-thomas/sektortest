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
            
            # Preis-Spalte extrahieren
            if 'Close' in data_raw.columns:
                prices = data_raw['Close']
            else:
                prices = data_raw.iloc[:, 0]
            
            # Monatliche Daten (ME = Month End)
            prices = prices.resample('ME').last().dropna()
            
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            if len(prices) < 24:
                continue
            
            # Renditen berechnen
            rets = prices.pct_change().dropna()
            
            # --- GEWICHTETE MOMENTUM-LOGIK (30% 9M / 70% 12M) ---
            mom9 = prices.pct_change(9)
            mom12 = prices.pct_change(12)
            
            # Kombiniertes Signal berechnen
            combined_signal = (0.3 * mom9) + (0.7 * mom12)
            
            # Signal um einen Monat verschieben (Entscheidung am Monatsende für den Folgemonat)
            bull_signal = combined_signal.shift(1) > 0
            
            # Nur Monate betrachten, in denen das kombinierte Signal "Go" gesagt hätte
            valid_months = rets[bull_signal]
            
            # Frosch-Score (Trefferquote im Folgemonat)
            hit_rate_val = 0
            if not valid_months.empty:
                hit_rate_val = valid_months[valid_months > 0].count() / valid_months.count()
            
            # Persistenz (Autokorrelation)
            persistence_val = rets.autocorr(lag=1)
            
            # Signal-to-Noise Ratio (SNR)
            std_val = rets.std()
            snr_val = (rets.mean() * 12) / (std_val * np.sqrt(12)) if std_val > 0 else 0
            
            # --- VERFEINERTE STATUS-LOGIK ---
            if persistence_val < 0:
                status_text = "GEFÄHRLICH"
            elif persistence_val < 0.05:
                status_text = "HEKTISCH"
            elif persistence_val < 0.10:
                status_text = "GRENZWERTIG"
            else:
                status_text = "ROBUST"

            results.append({
                "ticker": str(ticker),
                "persistence": float(round(persistence_val, 3)) if pd.notnull(persistence_val) else 0.0,
                "snr": float(round(snr_val, 3)) if pd.notnull(snr_val) else 0.0,
                "frosch_score": float(round(hit_rate_val, 2)),
                "status": str(status_text)
            })
        except Exception as e:
            print(f"Fehler bei {ticker}: {e}")
            continue
            
    if not results:
        return jsonify({"error": "Keine Ergebnisse. Möglicherweise Rate Limit."}), 429
        
    return jsonify(results)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
