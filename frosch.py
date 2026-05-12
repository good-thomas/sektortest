import pandas as pd
import numpy as np
import yfinance as yf
import sys
from datetime import datetime

def analyze_frosch_tickers(ticker_list):
    """
    Analysiert Ticker auf ihre Eignung für 9/12-Monats-Momentum (Frosch-Konformität).
    Speichert die Ergebnisse als CSV und gibt eine Tabelle aus.
    """
    if not ticker_list:
        print("Fehler: Keine Ticker übergeben.")
        print("Nutzung: python frosch.py TICKER1 TICKER2 ...")
        print("Beispiel: python frosch.py SMH URA ITA")
        return

    print(f"\n--- Frosch-Audit läuft für {len(ticker_list)} Assets ---")
    print(f"Zeitraum: Seit 2010 | Frequenz: Monatlich\n")
    
    try:
        # Daten laden
        # Wir laden 'Adj Close', um Dividenden und Splits korrekt zu berücksichtigen
        data = yf.download(ticker_list, start="2010-01-01", progress=False)['Adj Close']
        
        # Falls nur ein Ticker angefragt wurde, wandelt yf das oft in eine Series
        if len(ticker_list) == 1:
            data = data.to_frame(name=ticker_list[0])
            
        # Resampling auf Monatsendkurse
        data = data.resample('M').last()
        
        results = []
        
        for ticker in ticker_list:
            if ticker not in data.columns:
                print(f"Warnung: {ticker} konnte nicht geladen werden.")
                continue
                
            prices = data[ticker].dropna()
            
            # Sicherheitscheck: Genug Daten vorhanden?
            if len(prices) < 24:
                print(f"Warnung: {ticker} hat zu wenig Historie ({len(prices)} Monate).")
                continue
                
            # Monatliche Renditen berechnen
            rets = prices.pct_change().dropna()
            
            # 1. PERSISTENZ (Autokorrelation Lag-1)
            # > 0.10 ist gut für Trendfolge, < 0 ist gefährlich (Mean-Reversion)
            persistence = rets.autocorr(lag=1)
            
            # 2. SIGNAL-TO-NOISE RATIO (SNR)
            # Annualisierte Rendite / Annualisierte Volatilität
            snr = (rets.mean() * 12) / (rets.std() * np.sqrt(12)) if rets.std() != 0 else 0
            
            # 3. FROSCH-SCORE (Hit Rate des 9M-Momentums)
            # Wie oft war der Folgemonat positiv, wenn das 9M-Momentum positiv war?
            mom9 = prices.pct_change(9)
            bull_signal = mom9.shift(1) > 0
            valid_months = rets[bull_signal]
            hit_rate = valid_months[valid_months > 0].count() / valid_months.count() if valid_months.count() > 0 else 0
            
            # Einstufung Logik
            if persistence > 0.10 and snr > 0.4:
                status = "✅ ROBUST"
            elif persistence < 0:
                status = "❌ GEFÄHRLICH (Whipsaw)"
            else:
                status = "⚠️ GRENZWERTIG"

            results.append({
                "Ticker": ticker,
                "Persistenz": round(persistence, 3),
                "SNR": round(snr, 3),
                "Frosch-Score": round(hit_rate, 2),
                "Bewertung": status
            })
        
        # Ergebnis-DataFrame erstellen
        df = pd.DataFrame(results).sort_values("Persistenz", ascending=False)
        
        # Tabelle in der Konsole anzeigen
        print("ERGEBNIS-TABELLE:")
        print(df.to_string(index=False))
        
        # Als CSV speichern für deine Historie
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"audit_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nErgebnisse gespeichert unter: {filename}")
        
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")

if __name__ == "__main__":
    # Ticker werden aus den Kommandozeilen-Argumenten gelesen
    # sys.argv[0] ist der Dateiname, daher ab Index 1
    input_tickers = [t.upper() for t in sys.argv[1:]]
    analyze_frosch_tickers(input_tickers)
