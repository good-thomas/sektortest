import pandas as pd
import numpy as np
import yfinance as yf
import sys

def analyze_frosch_tickers(ticker_list):
    if not ticker_list:
        print("Bitte gib mindestens einen Ticker ein (z.B. python frosch.py SMH AAPL URA)")
        return

    print(f"--- Frosch-Audit läuft für: {', '.join(ticker_list)} ---")
    
    # Daten laden (15 Jahre für robuste Statistik)
    data = yf.download(ticker_list, start="2010-01-01", progress=False)['Adj Close']
    
    # Falls nur ein Ticker eingegeben wurde, Series zu DataFrame wandeln
    if isinstance(data, pd.Series):
        data = data.to_frame(name=ticker_list[0])
    
    # Auf monatliche Basis setzen (Schlusskurse)
    data = data.resample('M').last()
    
    results = []
    
    for ticker in ticker_list:
        if ticker not in data.columns:
            continue
            
        prices = data[ticker].dropna()
        if len(prices) < 24: # Mindestens 2 Jahre Daten nötig
            continue
            
        rets = prices.pct_change().dropna()
        
        # 1. PERSISTENZ (Autokorrelation Lag-1)
        # Misst die Tendenz: Folgt auf einen grünen Monat ein grüner Monat?
        persistence = rets.autocorr(lag=1)
        
        # 2. SIGNAL-TO-NOISE RATIO (SNR)
        # Trendstärke relativ zur "Hektik" (Volatilität)
        snr = (rets.mean() * 12) / (rets.std() * np.sqrt(12)) if rets.std() != 0 else 0
        
        # 3. FROSCH-SCORE (9M Momentum Trefferquote)
        # Wie oft war der Folgemonat positiv, wenn das 9M Momentum positiv war?
        mom9 = prices.pct_change(9)
        bull_signal = mom9.shift(1) > 0
        hit_rate = (rets[bull_signal] > 0).mean() if bull_signal.sum() > 0 else 0
        
        # Einstufung
        if persistence > 0.10 and snr > 0.4:
            status = "✅ ROBUST"
        elif persistence < 0:
            status = "❌ HEKTISCH (Whipsaw)"
        else:
            status = "⚠️ GRENZWERTIG"

        results.append({
            "Ticker": ticker,
            "Persistenz": round(persistence, 3),
            "SNR": round(snr, 3),
            "Frosch-Score": round(hit_rate, 2),
            "Bewertung": status
        })
    
    # Ergebnis als Tabelle ausgeben
    df = pd.DataFrame(results).sort_values("Persistenz", ascending=False)
    print("\nERGEBNIS:")
    print(df.to_string(index=False))
    print("\nLegende: Persistenz > 0.1 = Gute Trendfolge | < 0 = Gift für Momentum-Algos.")

if __name__ == "__main__":
    # Nimmt Ticker aus der Kommandozeile entgegen
    user_tickers = sys.argv[1:] 
    analyze_frosch_tickers(user_tickers)
