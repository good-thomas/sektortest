import pandas as pd
import numpy as np
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import traceback

app = Flask(__name__)
CORS(app)

@app.route("/audit", methods=["GET"])
def audit():
    ticker_str = request.args.get("tickers", "")
    if not ticker_str:
        return jsonify({"error": "Keine Ticker angegeben"}), 400
    
    ticker_list = [t.strip().upper() for t in ticker_str.split(",") if t.strip()]
    
    try:
        # 1. Daten laden (Wir fangen Fehler beim Download ab)
        raw_data = yf.download(ticker_list, start="2010-01-01", progress=False)
        
        if raw_data.empty:
            return jsonify({"error": "Keine Daten von Yahoo Finance erhalten"}), 404

        # 2. Struktur-Bereinigung
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
            try:
                if ticker not in data.columns:
                    continue
                    
                prices = data[ticker].dropna()
                if len(prices) < 24:
                    continue
                
                rets = prices.pct_change().dropna()
                
                # Persistenz
                persistence_val = rets.autocorr(lag=1)
                
                # SNR
                std_val = rets.std()
                snr_val = (rets.mean() * 12) / (std_val * np.sqrt(12)) if std_val > 0 else 0
                
                # Frosch-Score
                mom9 = prices.pct_change(9)
                bull_signal = mom9.shift(1) > 0
                valid_months = rets[bull_signal]
                
                hit_rate_val = 0
                if not valid_months.empty:
                    hit_rate_val = valid_months[valid_months > 0].count() / valid_months.count()
                
                status_text = "ROBUST" if persistence_val > 0.10 and snr_val > 0.4 else "HEKTISCH"
                if persistence_val < 0: status_text = "GEFÄHRLICH"

                # Konvertierung in Standard-Typen für JSON
                results.append({
                    "ticker": str(ticker),
                    "persistence": float(round(persistence_val, 3)) if pd.notnull(persistence_val) else 0.0,
                    "snr": float(round(snr_val, 3)) if pd.notnull(snr_val) else 0.0,
                    "frosch_score": float(round(hit_rate_val, 2)) if pd.notnull(hit_rate_val) else 0.0,
                    "status": str(status_text)
                })
            except Exception as e_inner:
                print(f"Fehler bei Ticker {ticker}: {e_inner}")
                continue
        
        return jsonify(results)
    
    except Exception as e:
        # Das gibt uns den vollen Fehler-Pfad im Render-Log aus
        print("KRITISCHER FEHLER:")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
