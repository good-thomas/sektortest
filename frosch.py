import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def run_bucket_audit(tickers_dict):
    """
    Analysiert das Universum auf Trend-Persistenz und Signal-Rausch-Verhältnis.
    Ziel: Hektische Sektoren (Frosch-Falle) identifizieren.
    """
    print(f"Starte Audit für {len(tickers_dict)} Ticker...")
    
    # Daten laden (ab 2010 für ausreichend Historie)
    symbols = list(tickers_dict.values())
    raw_data = yf.download(symbols, start="2010-01-01", progress=False)['Adj Close']
    
    # Auf monatliche Basis resamplen (wie im Haupt-Algorithmus)
    data = raw_data.resample('M').last()
    
    audit_results = []
    
    for name, ticker in tickers_dict.items():
        if ticker not in data.columns:
            continue
            
        # 1. Monatliche Renditen
        rets = data[ticker].pct_change().dropna()
        
        # 2. METRIK: Persistenz (Autokorrelation Lag-1)
        # Misst, ob auf einen guten Monat statistisch ein weiterer guter folgt.
        persistence = rets.autocorr(lag=1)
        
        # 3. METRIK: Signal-to-Noise Ratio (SNR)
        # Je höher, desto "glatter" der Trend.
        snr = (rets.mean() * 12) / (rets.std() * np.sqrt(12)) if rets.std() != 0 else 0
        
        # 4. METRIK: Frosch-Score (Realitätscheck 9M Momentum)
        # Wie oft war der Folgemonat positiv, wenn das 9M Momentum positiv war?
        mom9 = data[ticker].pct_change(9)
        # Wir schauen nur Monate an, in denen das Momentum positiv war
        bull_signal = mom9.shift(1) > 0
        hit_rate = (rets[bull_signal] > 0).mean() if bull_signal.sum() > 0 else 0
        
        # Bewertung
        status = "ROBUST" if persistence > 0.10 and snr > 0.5 else "HEKTISCH"
        if persistence < 0: status = "GEFÄHRLICH (Whipsaw)"

        audit_results.append({
            "Sektor": name,
            "Ticker": ticker,
            "Persistenz": round(persistence, 3),
            "SNR": round(snr, 3),
            "Frosch-Score": round(hit_rate, 3),
            "Status": status
        })
    
    # Ergebnis-Tabelle
    df_audit = pd.DataFrame(audit_results).sort_values("Persistenz", ascending=False)
    return df_audit

# --- DEIN UNIVERSUM ZUM TESTEN ---
if __name__ == "__main__":
    my_universe = {
        "Semis": "SMH", "Defense": "ITA", "Uranium": "URA", 
        "Biotech": "XBI", "Software": "IGV", "Gold Miners": "GDX",
        "Transport": "IYT", "Copper": "COPX", "Cyber": "HACK"
    }
    
    audit_report = run_bucket_audit(my_universe)
    
    print("\n--- UNIVERSUM AUDIT REPORT ---")
    print(audit_report.to_string(index=False))
    
    print("\nEMPFFEHLUNG FÜR THOMAS:")
    print("- Sektoren mit Status 'ROBUST' bilden den Kern des AIF.")
    print("- 'HEKTISCHE' Sektoren nur mit erhöhter Hürde (y_pa) zulassen.")
    print("- 'GEFÄHRLICHE' Sektoren komplett aus dem Universum streichen.")
