import json
import numpy as np
import pandas as pd
from datetime import datetime
from io import BytesIO

class ScenarioManager:
    """Manages saving, loading, and exporting decision scenarios."""
    
    VERSION = "1.0"
    
    @staticmethod
    def save_scenario(session_state, scenario_name):
        """
        Convert session state to JSON-serializable dictionary.
        
        Args:
            session_state: Streamlit session state model
            scenario_name: Name for the scenario
            
        Returns:
            dict: JSON-serializable scenario data
        """
        scenario = {
            "version": ScenarioManager.VERSION,
            "scenario_name": scenario_name,
            "timestamp": datetime.now().isoformat(),
            "num_criteria": session_state.get("num_criteria", 3),
            "num_alternatives": session_state.get("num_alternatives", 3),
            "criteria_names": session_state.get("criteria_names", []),
            "alternative_names": session_state.get("alternative_names", []),
            "ahp_pairwise": {},
            "decision_matrix": {},
            "weights": session_state.get("weights", {}),
            "impacts": session_state.get("impacts", {}),
            "ahp_weights": session_state.get("ahp_weights", {}),
            "fuzzy_ahp_weights": session_state.get("fuzzy_ahp_weights", {}),
            "topsis_results": session_state.get("topsis_results", {}),
            "promethee_config": {
                "pref_funcs": session_state.get("pref_funcs", {}),
                "p_params": session_state.get("p_params", {}),
                "q_params": session_state.get("q_params", {})
            }
        }
        
        # Convert AHP pairwise comparisons (tuple keys to string)
        ahp_pairwise = session_state.get("ahp_pairwise", {})
        for key, value in ahp_pairwise.items():
            if isinstance(key, tuple):
                str_key = f"{key[0]}_{key[1]}"
                scenario["ahp_pairwise"][str_key] = value
        
        # Convert decision matrix (tuple keys to string)
        decision_matrix = session_state.get("decision_matrix", {})
        for key, value in decision_matrix.items():
            if isinstance(key, tuple):
                str_key = f"{key[0]}_{key[1]}"
                scenario["decision_matrix"][str_key] = float(value) if isinstance(value, (int, float, np.number)) else value
        
        return scenario
    
    @staticmethod
    def load_scenario(json_data):
        """
        Validate and convert JSON data to session state format.
        
        Args:
            json_data: Dictionary loaded from JSON
            
        Returns:
            dict: Session state model
        """
        # Validate version
        version = json_data.get("version", "1.0")
        if version != ScenarioManager.VERSION:
            raise ValueError(f"Incompatible version: {version}. Expected {ScenarioManager.VERSION}")
        
        model = {
            "num_criteria": json_data.get("num_criteria", 3),
            "num_alternatives": json_data.get("num_alternatives", 3),
            "criteria_names": json_data.get("criteria_names", []),
            "alternative_names": json_data.get("alternative_names", []),
            "ahp_pairwise": {},
            "decision_matrix": {},
            "weights": json_data.get("weights", {}),
            "impacts": json_data.get("impacts", {}),
            "ahp_weights": json_data.get("ahp_weights", {}),
            "fuzzy_ahp_weights": json_data.get("fuzzy_ahp_weights", {}),
            "topsis_results": json_data.get("topsis_results", {}),
        }
        
        # Restore PROMETHEE config
        promethee_config = json_data.get("promethee_config", {})
        model["pref_funcs"] = promethee_config.get("pref_funcs", {})
        model["p_params"] = promethee_config.get("p_params", {})
        model["q_params"] = promethee_config.get("q_params", {})
        
        # Convert AHP pairwise (string keys back to tuples)
        ahp_pairwise = json_data.get("ahp_pairwise", {})
        for str_key, value in ahp_pairwise.items():
            parts = str_key.split("_")
            if len(parts) == 2:
                key = (int(parts[0]), int(parts[1]))
                model["ahp_pairwise"][key] = value
        
        # Convert decision matrix (string keys back to tuples)
        decision_matrix = json_data.get("decision_matrix", {})
        for str_key, value in decision_matrix.items():
            parts = str_key.split("_")
            if len(parts) == 2:
                key = (int(parts[0]), int(parts[1]))
                model["decision_matrix"][key] = value
        
        return model
    
    @staticmethod
    def export_results_csv(results, result_type="weights"):
        """
        Export results to CSV format.
        
        Args:
            results: Dictionary of results
            result_type: Type of results ("weights", "topsis", "promethee")
            
        Returns:
            str: CSV string
        """
        if result_type == "weights":
            df = pd.DataFrame(list(results.items()), columns=["Criterion", "Weight"])
        elif result_type == "topsis":
            df = pd.DataFrame(results)
        elif result_type == "promethee":
            df = pd.DataFrame(results)
        else:
            df = pd.DataFrame(results)
        
        return df.to_csv(index=False)
    
    @staticmethod
    def export_results_excel(scenario_data, results_data):
        """
        Export complete scenario and results to Excel with multiple sheets.
        
        Args:
            scenario_data: Scenario metadata
            results_data: Dictionary with results from different methods
            
        Returns:
            bytes: Excel file as bytes
        """
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Metadata sheet
            metadata = pd.DataFrame({
                "Property": ["Scenario Name", "Date", "Criteria Count", "Alternatives Count"],
                "Value": [
                    scenario_data.get("scenario_name", "Unnamed"),
                    scenario_data.get("timestamp", ""),
                    scenario_data.get("num_criteria", 0),
                    scenario_data.get("num_alternatives", 0)
                ]
            })
            metadata.to_excel(writer, sheet_name="Metadata", index=False)
            
            # Criteria sheet
            if scenario_data.get("criteria_names"):
                criteria_df = pd.DataFrame({
                    "Criterion": scenario_data["criteria_names"]
                })
                criteria_df.to_excel(writer, sheet_name="Criteria", index=False)
            
            # Weights sheet
            if results_data.get("weights"):
                weights_df = pd.DataFrame(
                    list(results_data["weights"].items()),
                    columns=["Criterion", "Weight"]
                )
                weights_df.to_excel(writer, sheet_name="Weights", index=False)
            
            # Results sheet
            ranking = results_data.get("ranking")
            if ranking is not None and not ranking.empty:
                results_df = pd.DataFrame(ranking)
                results_df.to_excel(writer, sheet_name="Results", index=False)
        
        output.seek(0)
        return output.getvalue()

    @staticmethod
    def export_group_analysis_excel(results, criteria_names, judge_names=None):
        """
        Exports full multi-judge AHP analysis into a structured multi-tab Excel workbook.
        Tabs:
          1. Consenso Grupal (Group Weights, Consistency Ratio, Dispersion)
          2. Diagnóstico Jueces (Individual CR, Status, Alignment)
          3. Comparativa Criterios (Table of weights per judge vs Group)
        """
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Tab 1: Consenso Grupal
            group_summary = []
            group_weights = results.get("weights", {})
            cr_group = results.get("consistency_ratio", 0.0)
            is_cons = results.get("is_consistent", True)
            
            summary_info = [
                {"Métrica": "Número de Evaluadores Incluidos", "Valor": results.get("num_judges_pairwise", 0)},
                {"Métrica": "Ratio de Consistencia Grupal (CR)", "Valor": f"{cr_group:.4f}"},
                {"Métrica": "Estado de Consistencia Grupal", "Valor": "Consistente (CR < 0.10)" if is_cons else "Inconsistente (CR >= 0.10)"},
                {"Métrica": "Método de Agregación", "Valor": "Media Geométrica Ponderada (Weighted AIJ)"}
            ]
            pd.DataFrame(summary_info).to_excel(writer, sheet_name="Consenso Grupal", index=False, startrow=0)
            
            # Group Weights Table
            disp = results.get("criteria_dispersion", {})
            rows_w = []
            for c in criteria_names:
                w_val = group_weights.get(c, 0.0)
                disp_info = disp.get(c, {})
                rows_w.append({
                    "Criterio": c,
                    "Peso Grupal": w_val,
                    "Peso Grupal (%)": f"{w_val * 100:.2f}%",
                    "Desviación Estándar": round(disp_info.get("std", 0.0), 4),
                    "Nivel de Acuerdo": disp_info.get("agreement_level", "N/A")
                })
            df_group_w = pd.DataFrame(rows_w)
            df_group_w.to_excel(writer, sheet_name="Consenso Grupal", index=False, startrow=len(summary_info) + 2)
            
            # Tab 2: Diagnóstico de Jueces
            diag = results.get("individual_diagnostics", [])
            if diag:
                diag_rows = []
                for d in diag:
                    diag_rows.append({
                        "Evaluador": d["judge_name"],
                        "Ratio de Consistencia (CR)": round(d["consistency_ratio"], 4),
                        "Diagnóstico de Consistencia": d["badge"],
                        "Nivel de Alineación con el Consenso": f"{d['alignment_pct']:.1f}%",
                        "Distancia al Grupo": round(d["distance_to_group"], 4),
                        "Criterio Más Valorado": d["top_criterion"]
                    })
                pd.DataFrame(diag_rows).to_excel(writer, sheet_name="Diagnóstico Evaluadores", index=False)
                
            # Tab 3: Comparativa Criterios
            if diag:
                comp_dict = {"Criterio": criteria_names}
                for d in diag:
                    comp_dict[d["judge_name"]] = [f"{d['weights'].get(c, 0.0)*100:.2f}%" for c in criteria_names]
                comp_dict["CONSENSO GRUPAL"] = [f"{group_weights.get(c, 0.0)*100:.2f}%" for c in criteria_names]
                pd.DataFrame(comp_dict).to_excel(writer, sheet_name="Comparativa Criterios", index=False)
                
        output.seek(0)
        return output.getvalue()
