import os
import json
import requests
import numpy as np

class AIExecutiveAnalyst:
    """
    Ultra-lightweight decision intelligence agent.
    Synthesizes group multi-judge AHP and multicriteria results into concise, executive-level insights.
    Optimized for minimal token usage (~180 prompt tokens, ~150 completion tokens).
    """

    @staticmethod
    def compact_payload(results, criteria_names, judge_names=None):
        """
        Condenses multi-judge results into minimal summary facts to minimize token costs.
        """
        weights = results.get("weights", {})
        cr = float(results.get("consistency_ratio", 0.0))
        is_consistent = results.get("is_consistent", True)
        diag = results.get("individual_diagnostics", [])
        
        # Sort criteria by weight
        sorted_crit = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        top_crit, top_w = sorted_crit[0] if sorted_crit else ("N/A", 0.0)
        
        # Highly aligned judges (>80%) and dissident judges
        aligned_judges = []
        dissidents = []
        inconsistent_judges = []
        
        for d in diag:
            name = d["judge_name"]
            align = d["alignment_pct"]
            cr_j = d["consistency_ratio"]
            top_j = d["top_criterion"]
            
            if align >= 80.0:
                aligned_judges.append(name)
            else:
                dissidents.append(f"{name} (priorizó '{top_j}', alineación {align:.0f}%)")
                
            if cr_j > 0.20:
                inconsistent_judges.append(f"{name} (CR: {cr_j:.2f})")

        return {
            "top_criterion": top_crit,
            "top_weight_pct": f"{top_w * 100:.1f}%",
            "ranking_summary": [f"{c} ({w*100:.1f}%)" for c, w in sorted_crit],
            "cr_group": f"{cr:.3f}",
            "is_consistent": is_consistent,
            "aligned_judges": aligned_judges,
            "dissident_judges": dissidents,
            "inconsistent_judges": inconsistent_judges
        }

    @staticmethod
    def generate_heuristic_fallback(compact_data):
        """
        Deterministic, zero-token, zero-cost executive synthesis if no API key is provided.
        """
        top_c = compact_data["top_criterion"]
        top_w = compact_data["top_weight_pct"]
        aligned = ", ".join(compact_data["aligned_judges"]) if compact_data["aligned_judges"] else "Distribución variada"
        dissidents = "; ".join(compact_data["dissident_judges"]) if compact_data["dissident_judges"] else "Ninguna discrepancia relevante"
        incons = ", ".join(compact_data["inconsistent_judges"]) if compact_data["inconsistent_judges"] else "Todos los jueces presentaron coherencia lógica adecuada"
        
        return (
            f"🎯 **Consenso Principal:** El criterio prioritario indiscutible del grupo es **{top_c}** con una ponderación de **{top_w}**, "
            f"respaldado fuertemente por: *{aligned}*.\n\n"
            f"👥 **Posturas Disidentes / Enfoques Alternativos:** Se observan inclinaciones divergentes en: *{dissidents}*.\n\n"
            f"⚠️ **Calidad y Consistencia de la Decisión:** El Ratio de Consistencia Grupal es {compact_data['cr_group']}. "
            f"Observaciones metodológicas individuales: *{incons}*."
        )

    @classmethod
    def analyze(cls, results, criteria_names, judge_names=None, api_key=None, provider="gemini"):
        """
        Calls Gemini 1.5 Flash (or OpenAI mini) with ultra-concise prompt.
        Falls back to local heuristic synthesis if no key is provided or on network error.
        """
        compact_data = cls.compact_payload(results, criteria_names, judge_names)
        
        # Resolve API Key from argument or environment variables
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        if not key:
            return cls.generate_heuristic_fallback(compact_data), "Motor Heurístico Local (Sin costo)"

        prompt = (
            "Actúa como consultor senior en toma de decisiones estratégicas (AHP). "
            "Genera una interpretación ejecutiva en exactamente 3 viñetas concisas con emojis:\n"
            "1. 🎯 Consenso y Fortaleza: criterio ganador, porcentaje y quiénes lo respaldan con mayor fuerza.\n"
            "2. 👥 Matices y Disidencias: qué evaluadores discrepan y qué prioridades específicas defienden.\n"
            "3. ⚠️ Recomendación Metodológica: diagnóstico del Ratio de Consistencia (CR), alertando si hubo contradicciones lógicas.\n\n"
            f"Datos del caso:\n{json.dumps(compact_data, ensure_ascii=False, indent=2)}\n"
            "Responde en español, profesional, directo y sin introducciones ni despedidas."
        )

        try:
            # 1. Google Gemini Flash (Recommended: Ultra-cheap / Free tier)
            if provider.lower() == "gemini" or "AIza" in key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={key}"
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300}
                }
                resp = requests.post(url, json=payload, timeout=12)
                if resp.status_code == 200:
                    data = resp.json()
                    text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                    return text, "Google Gemini 1.5 Flash"
                else:
                    err_msg = resp.text
                    return f"{cls.generate_heuristic_fallback(compact_data)}\n\n*(Aviso: Gemini API respondió: {resp.status_code})*", "Motor Local (Fallback)"
            
            # 2. OpenAI (GPT-4o-mini)
            else:
                url = "https://api.openai.com/v1/chat/completions"
                headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
                payload = {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 300,
                    "temperature": 0.2
                }
                resp = requests.post(url, headers=headers, json=payload, timeout=12)
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"].strip()
                    return text, "OpenAI GPT-4o-mini"
                else:
                    return f"{cls.generate_heuristic_fallback(compact_data)}\n\n*(Aviso: OpenAI API respondió: {resp.status_code})*", "Motor Local (Fallback)"

        except Exception as e:
            # Safe graceful degradation to local heuristic
            return f"{cls.generate_heuristic_fallback(compact_data)}\n\n*(Modo sin conexión: {str(e)})*", "Motor Local (Fallback)"
