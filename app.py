import streamlit as st
import pandas as pd
import numpy as np
import os
from decision_engine.ahp import AHPEngine
from decision_engine.topsis import TOPSISEngine
from decision_engine.fuzzy_ahp import FuzzyAHPEngine
from decision_engine.promethee import PrometheeEngine
from decision_engine.scenario_manager import ScenarioManager
from decision_engine.group_ahp import GroupDecisionEngine
from decision_engine.anp import ANPEngine
import json

st.set_page_config(page_title="Decision Engine Suite", layout="wide")

# Inject Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }

    h1, h2, h3 {
        text-transform: uppercase;
        font-weight: 800 !important;
        color: #1E1E1E;
    }
    
    /* Button Styling */
    .stButton > button {
        background-color: #F58634 !important;
        color: white !important;
        border-radius: 0px !important; /* Square/Sharp edges */
        border: none !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #D9752B !important; /* Darker orange on hover */
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1E1E1E;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #FFFFFF !important;
    }
    
    /* Sidebar input fields should have dark text */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea {
        color: #1E1E1E !important;
        background-color: #FFFFFF !important;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        border-radius: 0px;
    }
    .stNumberInput > div > div > input {
        border-radius: 0px;
    }
    
    /* Dataframe Headers */
    [data-testid="stDataFrame"] th {
        background-color: #1E1E1E !important;
        color: white !important;
    }

</style>
""", unsafe_allow_html=True)

st.title("Decision Engine Suite")
st.markdown("### AHP & TOPSIS for Decision Making")

# Initialize Session State Model
if "model" not in st.session_state:
    st.session_state.model = {
        "num_criteria": 3,
        "criteria_names": ["C1", "C2", "C3"],
        "ahp_pairwise": {}, # Key: (i,j), Value: {'winner': name, 'intensity': val}
        "ahp_weights": None,
        "num_alternatives": 3,
        "alternatives": ["A1", "A2", "A3"],
        "topsis_weights": [1/3]*3,
        "topsis_impacts": ["Benefit (+)"]*3,
        "promethee_funcs": ["Usual"]*3,
        "promethee_p": [0.0]*3,
        "promethee_q": [0.0]*3,
        "decision_matrix": {} # Key: (alt_idx, crit_idx), Value: float
    }

# Helper to update model from widgets
def update_model(key, index=None, index2=None):
    val = st.session_state[key]
    if key.startswith("c_name_"):
        st.session_state.model["criteria_names"][index] = val
    elif key == "num_criteria":
        # Resize lists if needed
        old_n = st.session_state.model["num_criteria"]
        new_n = val
        st.session_state.model["num_criteria"] = new_n
        if new_n > old_n:
            for i in range(old_n, new_n):
                st.session_state.model["criteria_names"].append(f"C{i+1}")
                st.session_state.model["topsis_weights"].append(1.0/new_n)
                st.session_state.model["topsis_impacts"].append("Benefit (+)")
                st.session_state.model["promethee_funcs"].append("Usual")
                st.session_state.model["promethee_p"].append(0.0)
                st.session_state.model["promethee_q"].append(0.0)
        elif new_n < old_n:
            st.session_state.model["criteria_names"] = st.session_state.model["criteria_names"][:new_n]
            st.session_state.model["topsis_weights"] = st.session_state.model["topsis_weights"][:new_n]
            st.session_state.model["topsis_impacts"] = st.session_state.model["topsis_impacts"][:new_n]
            st.session_state.model["promethee_funcs"] = st.session_state.model["promethee_funcs"][:new_n]
            st.session_state.model["promethee_p"] = st.session_state.model["promethee_p"][:new_n]
            st.session_state.model["promethee_q"] = st.session_state.model["promethee_q"][:new_n]
            
    elif key.startswith("win_") or key.startswith("int_"):
        # These are handled in the loop directly or via specific logic
        pass
    elif key == "num_alternatives":
        old_n = st.session_state.model["num_alternatives"]
        new_n = val
        st.session_state.model["num_alternatives"] = new_n
        if new_n > old_n:
            for i in range(old_n, new_n):
                st.session_state.model["alternatives"].append(f"A{i+1}")
        elif new_n < old_n:
            st.session_state.model["alternatives"] = st.session_state.model["alternatives"][:new_n]
            
    elif key.startswith("alt_name_"):
        st.session_state.model["alternatives"][index] = val
    elif key.startswith("w_"):
        st.session_state.model["topsis_weights"][index] = val
    elif key.startswith("imp_"):
        st.session_state.model["topsis_impacts"][index] = val
    elif key.startswith("pf_"):
        st.session_state.model["promethee_funcs"][index] = val
    elif key.startswith("pp_"):
        st.session_state.model["promethee_p"][index] = val
    elif key.startswith("pq_"):
        st.session_state.model["promethee_q"][index] = val
    elif key.startswith("dm_"):
        st.session_state.model["decision_matrix"][(index, index2)] = val

# Sidebar for navigation
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", width='stretch')
else:
    st.sidebar.markdown("## PROVOKERS")

mode = st.sidebar.selectbox("Select Mode", [
    "AHP (Weights)", 
    "Fuzzy AHP (Weights)", 
    "TOPSIS (Ranking)", 
    "PROMETHEE (Ranking)",
    "Combined (AHP + TOPSIS)", 
    "Combined (Fuzzy AHP + TOPSIS)",
    "Combined (AHP + PROMETHEE)",
    "Group Decision Aggregator",
    "ANP (Network)"
])

# Scenario Management Section
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Scenario Management")

# Save Scenario
scenario_name = st.sidebar.text_input("Scenario Name", value="My Scenario")
if st.sidebar.button("Save Scenario"):
    try:
        scenario_data = ScenarioManager.save_scenario(st.session_state.model, scenario_name)
        json_str = json.dumps(scenario_data, indent=2)
        st.sidebar.download_button(
            label="📥 Download JSON",
            data=json_str,
            file_name=f"{scenario_name.replace(' ', '_')}.json",
            mime="application/json"
        )
        st.sidebar.success("Scenario ready for download!")
    except Exception as e:
        st.sidebar.error(f"Error saving: {e}")

# Load Scenario
uploaded_file = st.sidebar.file_uploader("Load Scenario", type="json", key="scenario_uploader")
if uploaded_file is not None:
    # Use a flag to prevent infinite rerun loop
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state.get("last_loaded_file") != file_id:
        try:
            json_data = json.load(uploaded_file)
            loaded_model = ScenarioManager.load_scenario(json_data)
            st.session_state.model = loaded_model
            st.session_state.last_loaded_file = file_id
            st.sidebar.success(f"Loaded: {json_data.get('scenario_name', 'Unnamed')}")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading: {e}")

st.sidebar.markdown("---")

def render_ahp(fuzzy=False):
    if fuzzy:
        st.header("Fuzzy Analytic Hierarchy Process")
        st.info("Uses Triangular Fuzzy Numbers to handle uncertainty in judgments.")
    else:
        st.header("Analytic Hierarchy Process (AHP)")
    
    # Fuzzy Input Mode Selection
    fuzzy_mode = "Standard Scale (1-9)"
    if fuzzy:
        fuzzy_mode = st.radio(
            "Input Mode", 
            ["Standard Scale (1-9)", "Manual TFN Input (Advanced)"],
            horizontal=True,
            key="fuzzy_input_mode"
        )
    
    # Number of Criteria
    st.number_input(
        "Number of Criteria", 
        min_value=2, max_value=10, 
        value=st.session_state.model["num_criteria"],
        key="num_criteria",
        on_change=update_model, args=("num_criteria",)
    )
    
    num_criteria = st.session_state.model["num_criteria"]
    
    # Criteria Names
    criteria_names = st.session_state.model["criteria_names"]
    cols = st.columns(num_criteria)
    for i in range(num_criteria):
        cols[i].text_input(
            f"Criterion {i+1} Name", 
            value=criteria_names[i], 
            key=f"c_name_{i}",
            on_change=update_model, args=(f"c_name_{i}", i)
        )
    
    st.subheader("Pairwise Comparison Matrix")
    if fuzzy and fuzzy_mode == "Manual TFN Input (Advanced)":
        st.write("For each pair, define the Triangular Fuzzy Number (Lower, Middle, Upper).")
        matrix = np.zeros((num_criteria, num_criteria, 3))
        # Initialize diagonal with (1,1,1)
        for i in range(num_criteria):
            matrix[i, i] = (1, 1, 1)
    else:
        st.write("For each pair, select which criterion is more important and by how much (1-9).")
        matrix = np.ones((num_criteria, num_criteria))
    
    for i in range(num_criteria):
        for j in range(i + 1, num_criteria):
            col1, col2, col3 = st.columns([2, 2, 1])
            
            pair_key = (i, j)
            saved_pair = st.session_state.model["ahp_pairwise"].get(pair_key, {'winner': 'Equal', 'intensity': 1})
            
            # Ensure saved winner is valid (in case name changed)
            valid_options = [criteria_names[i], criteria_names[j], "Equal"]
            current_winner = saved_pair['winner']
            if current_winner not in valid_options and current_winner != "Equal":
                # Try to map by index if name changed? Or just reset to Equal
                current_winner = "Equal"

            with col1:
                winner = st.radio(
                    f"Comparison: {criteria_names[i]} vs {criteria_names[j]}",
                    valid_options,
                    index=valid_options.index(current_winner),
                    key=f"win_{i}_{j}",
                    horizontal=True,
                    label_visibility="collapsed"
                )
                
            with col2:
                if winner == "Equal":
                    intensity = 1
                    tfn = (1.0, 1.0, 1.0)
                    st.write("Equal Importance")
                else:
                    if fuzzy and fuzzy_mode == "Manual TFN Input (Advanced)":
                        # Manual TFN Input
                        saved_tfn = saved_pair.get('tfn', (1.0, 1.0, 1.0))
                        c1, c2, c3 = st.columns(3)
                        l = c1.number_input("L", value=float(saved_tfn[0]), key=f"l_{i}_{j}", step=0.1)
                        m = c2.number_input("M", value=float(saved_tfn[1]), key=f"m_{i}_{j}", step=0.1)
                        u = c3.number_input("U", value=float(saved_tfn[2]), key=f"u_{i}_{j}", step=0.1)
                        
                        if l > m or m > u:
                            st.warning("Ensure L <= M <= U")
                        
                        intensity = m # Use middle for intensity placeholder
                        tfn = (l, m, u)
                    else:
                        # Standard Slider
                        intensity = st.slider(
                            f"Intensity for {winner}",
                            min_value=1, max_value=9, 
                            value=int(saved_pair['intensity']),
                            key=f"int_{i}_{j}"
                        )
                        tfn = None # Not used in crisp mode
            
            # Update model immediately
            st.session_state.model["ahp_pairwise"][pair_key] = {'winner': winner, 'intensity': intensity, 'tfn': tfn}
            
            # Update local matrix for calculation
            if fuzzy and fuzzy_mode == "Manual TFN Input (Advanced)":
                if winner == "Equal":
                     matrix[i, j] = (1, 1, 1)
                     matrix[j, i] = (1, 1, 1)
                elif winner == criteria_names[i]:
                    matrix[i, j] = tfn
                    # Reciprocal: (1/u, 1/m, 1/l)
                    if tfn[0] != 0 and tfn[1] != 0 and tfn[2] != 0:
                        matrix[j, i] = (1/tfn[2], 1/tfn[1], 1/tfn[0])
                    else:
                         matrix[j, i] = (0, 0, 0) # Handle zero division risk?
                elif winner == criteria_names[j]:
                    # Reciprocal logic inverted
                    matrix[j, i] = tfn
                    if tfn[0] != 0 and tfn[1] != 0 and tfn[2] != 0:
                        matrix[i, j] = (1/tfn[2], 1/tfn[1], 1/tfn[0])
            else:
                # Standard Crisp Logic
                if winner == criteria_names[i]:
                    matrix[i, j] = intensity
                    matrix[j, i] = 1 / intensity
                elif winner == criteria_names[j]:
                    matrix[i, j] = 1 / intensity
                    matrix[j, i] = intensity
                else:
                    matrix[i, j] = 1.0
                    matrix[j, i] = 1.0
    
    if st.button("Calculate Weights"):
        try:
            if fuzzy:
                if fuzzy_mode == "Manual TFN Input (Advanced)":
                    engine = FuzzyAHPEngine(matrix, criteria_names, input_type="fuzzy")
                else:
                    engine = FuzzyAHPEngine(matrix, criteria_names, input_type="crisp")
            else:
                engine = AHPEngine(matrix, criteria_names)
            
            results = engine.get_results()
            
            st.success("Weights Calculated!")
            st.session_state.model["ahp_weights"] = results['weights']
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Weights")
                weights_df = pd.DataFrame(list(results['weights'].items()), columns=['Criterion', 'Weight'])
                st.dataframe(weights_df)
            
            with col2:
                if not fuzzy:
                    st.subheader("Consistency")
                    st.metric("Consistency Ratio (CR)", f"{results['consistency_ratio']:.4f}")
                    if results['is_consistent']:
                        st.success("The matrix is consistent.")
                    else:
                        st.error("The matrix is inconsistent (CR > 0.1). Please revise comparisons.")
            
            # Export Results
            st.markdown("---")
            st.subheader("📊 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_data = ScenarioManager.export_results_csv(results['weights'], "weights")
                st.download_button(
                    "📄 Download CSV",
                    data=csv_data,
                    file_name="ahp_weights.csv",
                    mime="text/csv"
                )
            with col_exp2:
                excel_data = ScenarioManager.export_results_excel(
                    {"scenario_name": scenario_name, "num_criteria": num_criteria, "criteria_names": criteria_names},
                    {"weights": results['weights']}
                )
                st.download_button(
                    "📊 Download Excel",
                    data=excel_data,
                    file_name="ahp_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            if fuzzy:
                st.subheader("Method Info")
                st.write("Method: Buckley's Geometric Mean")
                st.write("Defuzzification: Center of Area")
                    
        except Exception as e:
            st.error(f"Error: {e}")

def render_topsis(use_ahp_weights=False):
    st.header("TOPSIS Ranking")
    
    if use_ahp_weights and not st.session_state.model["ahp_weights"]:
        st.warning("Please run AHP first to calculate weights.")
        return

    # Configuration
    if use_ahp_weights:
        criteria_names = list(st.session_state.model["ahp_weights"].keys())
        num_criteria = len(criteria_names)
        st.write(f"Using {num_criteria} criteria from AHP: {', '.join(criteria_names)}")
        # Sync num_criteria in model just in case
        # st.session_state.model["num_criteria"] = num_criteria
    else:
        # Use shared criteria model
        st.number_input(
            "Number of Criteria", 
            min_value=2, max_value=10, 
            value=st.session_state.model["num_criteria"],
            key="num_criteria_t", # Different key to avoid conflict if both rendered? No, only one mode at a time usually.
            on_change=update_model, args=("num_criteria",)
        )
        num_criteria = st.session_state.model["num_criteria"]
        criteria_names = st.session_state.model["criteria_names"]
        
        cols = st.columns(num_criteria)
        for i in range(num_criteria):
            cols[i].text_input(
                f"Criterion {i+1} Name", 
                value=criteria_names[i], 
                key=f"c_name_{i}", # Same key as AHP to share state? Streamlit warns about duplicate keys if both rendered.
                # Since render_ahp and render_topsis are mutually exclusive in standalone modes, same key is fine.
                # BUT in Combined mode, render_ahp is called, then render_topsis.
                # If render_topsis uses same keys, it crashes.
                # So we must use different keys or conditional keys.
                # However, in Combined mode, use_ahp_weights=True, so this block is skipped!
                # So we are safe.
                on_change=update_model, args=(f"c_name_{i}", i)
            )
            
    # Alternatives
    st.number_input(
        "Number of Alternatives", 
        min_value=2, max_value=20, 
        value=st.session_state.model["num_alternatives"],
        key="num_alternatives",
        on_change=update_model, args=("num_alternatives",)
    )
    num_alternatives = st.session_state.model["num_alternatives"]
    alternatives = st.session_state.model["alternatives"]
    
    for i in range(num_alternatives):
        st.text_input(
            f"Alternative {i+1} Name", 
            value=alternatives[i], 
            key=f"alt_name_{i}",
            on_change=update_model, args=(f"alt_name_{i}", i)
        )

    # Weights and Impacts
    st.subheader("Weights and Impacts")
    weights = []
    impacts = []
    
    cols = st.columns(num_criteria)
    for i in range(num_criteria):
        with cols[i]:
            st.markdown(f"**{criteria_names[i]}**")
            if use_ahp_weights:
                w = st.session_state.model["ahp_weights"][criteria_names[i]]
                st.write(f"Weight: {w:.4f}")
                weights.append(w)
            else:
                w = st.number_input(
                    f"Weight", 
                    min_value=0.0, max_value=1.0, 
                    value=st.session_state.model["topsis_weights"][i], 
                    key=f"w_{i}",
                    on_change=update_model, args=(f"w_{i}", i)
                )
                weights.append(w)
            
            imp_val = st.session_state.model["topsis_impacts"][i]
            imp = st.selectbox(
                "Impact", 
                ["Benefit (+)", "Cost (-)"], 
                index=0 if "Benefit" in imp_val else 1,
                key=f"imp_{i}",
                on_change=update_model, args=(f"imp_{i}", i)
            )
            impacts.append('+' if "Benefit" in imp else '-')

    # Decision Matrix
    st.subheader("Decision Matrix")
    st.info("Enter the values for each alternative against each criterion.")
    
    decision_matrix = np.zeros((num_alternatives, num_criteria))
    
    for i in range(num_alternatives):
        st.markdown(f"**{alternatives[i]}**")
        cols = st.columns(num_criteria)
        for j in range(num_criteria):
            val = st.session_state.model["decision_matrix"].get((i, j), 0.0)
            new_val = cols[j].number_input(
                f"{criteria_names[j]}", 
                value=float(val), 
                key=f"dm_{i}_{j}",
                on_change=update_model, args=(f"dm_{i}_{j}", i, j)
            )
            decision_matrix[i, j] = new_val

    if st.button("Calculate Ranking"):
        try:
            # Normalize weights if manual
            if not use_ahp_weights:
                total_weight = sum(weights)
                if total_weight == 0:
                    st.error("Total weight cannot be zero.")
                    return
                weights = [w/total_weight for w in weights]
            
            df = pd.DataFrame(decision_matrix, index=alternatives, columns=criteria_names)
            
            topsis = TOPSISEngine(df, weights, impacts)
            results = topsis.rank()
            
            st.success("Ranking Calculated!")
            st.dataframe(results.style.highlight_max(axis=0, subset=['Score']), use_container_width=True)
            
            # Chart
            st.bar_chart(results.set_index('Alternative')['Score'])
            
            # Export Results
            st.markdown("---")
            st.subheader("📊 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_data = ScenarioManager.export_results_csv(results, "topsis")
                st.download_button(
                    "📄 Download CSV",
                    data=csv_data,
                    file_name="topsis_results.csv",
                    mime="text/csv"
                )
            with col_exp2:
                excel_data = ScenarioManager.export_results_excel(
                    {"scenario_name": scenario_name, "num_criteria": num_criteria, "num_alternatives": num_alternatives, "criteria_names": criteria_names},
                    {"ranking": results}
                )
                st.download_button(
                    "📊 Download Excel",
                    data=excel_data,
                    file_name="topsis_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
        except Exception as e:
            st.error(f"Error: {e}")

def render_promethee(use_ahp_weights=False):
    st.header("PROMETHEE Ranking")
    
    if use_ahp_weights and not st.session_state.model["ahp_weights"]:
        st.warning("Please run AHP first to calculate weights.")
        return

    # Configuration
    if use_ahp_weights:
        criteria_names = list(st.session_state.model["ahp_weights"].keys())
        num_criteria = len(criteria_names)
        st.write(f"Using {num_criteria} criteria from AHP: {', '.join(criteria_names)}")
    else:
        st.number_input(
            "Number of Criteria", 
            min_value=2, max_value=10, 
            value=st.session_state.model["num_criteria"],
            key="num_criteria_p",
            on_change=update_model, args=("num_criteria",)
        )
        num_criteria = st.session_state.model["num_criteria"]
        criteria_names = st.session_state.model["criteria_names"]
        
        cols = st.columns(num_criteria)
        for i in range(num_criteria):
            cols[i].text_input(
                f"Criterion {i+1} Name", 
                value=criteria_names[i], 
                key=f"c_name_{i}",
                on_change=update_model, args=(f"c_name_{i}", i)
            )
            
    # Alternatives
    st.number_input(
        "Number of Alternatives", 
        min_value=2, max_value=20, 
        value=st.session_state.model["num_alternatives"],
        key="num_alternatives_p",
        on_change=update_model, args=("num_alternatives",)
    )
    num_alternatives = st.session_state.model["num_alternatives"]
    alternatives = st.session_state.model["alternatives"]
    
    for i in range(num_alternatives):
        st.text_input(
            f"Alternative {i+1} Name", 
            value=alternatives[i], 
            key=f"alt_name_{i}",
            on_change=update_model, args=(f"alt_name_{i}", i)
        )

    # Weights, Impacts, and Preference Functions
    st.subheader("Criteria Configuration")
    weights = []
    impacts = []
    pref_funcs = []
    p_params = []
    q_params = []
    
    for i in range(num_criteria):
        with st.expander(f"Configuration for {criteria_names[i]}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if use_ahp_weights:
                    w = st.session_state.model["ahp_weights"][criteria_names[i]]
                    st.write(f"Weight: {w:.4f}")
                    weights.append(w)
                else:
                    w = st.number_input(
                        f"Weight", 
                        min_value=0.0, max_value=1.0, 
                        value=st.session_state.model["topsis_weights"][i], 
                        key=f"w_p_{i}",
                        on_change=update_model, args=(f"w_p_{i}", i)
                    )
                    weights.append(w)
                
                imp_val = st.session_state.model["topsis_impacts"][i]
                imp = st.selectbox(
                    "Impact", 
                    ["Benefit (+)", "Cost (-)"], 
                    index=0 if "Benefit" in imp_val else 1,
                    key=f"imp_p_{i}",
                    on_change=update_model, args=(f"imp_p_{i}", i)
                )
                impacts.append('+' if "Benefit" in imp else '-')
            
            with col2:
                pf_val = st.session_state.model["promethee_funcs"][i]
                pf = st.selectbox(
                    "Preference Function",
                    ["Usual", "Linear", "Linear (q, p)"],
                    index=["Usual", "Linear", "Linear (q, p)"].index(pf_val) if pf_val in ["Usual", "Linear", "Linear (q, p)"] else 0,
                    key=f"pf_{i}",
                    on_change=update_model, args=(f"pf_{i}", i)
                )
                pref_funcs.append(pf)
            
            with col3:
                if pf != "Usual":
                    p_val = st.number_input(
                        "Preference Threshold (p)",
                        min_value=0.0, value=float(st.session_state.model["promethee_p"][i]),
                        key=f"pp_{i}",
                        on_change=update_model, args=(f"pp_{i}", i)
                    )
                    p_params.append(p_val)
                    
                    if pf == "Linear (q, p)":
                        q_val = st.number_input(
                            "Indifference Threshold (q)",
                            min_value=0.0, value=float(st.session_state.model["promethee_q"][i]),
                            key=f"pq_{i}",
                            on_change=update_model, args=(f"pq_{i}", i)
                        )
                        q_params.append(q_val)
                        if q_val > p_val:
                            st.warning(f"Warning: q ({q_val}) should be <= p ({p_val})")
                    else:
                        q_params.append(0.0)
                else:
                    p_params.append(0.0)
                    q_params.append(0.0)

    # Decision Matrix
    st.subheader("Decision Matrix")
    decision_matrix = np.zeros((num_alternatives, num_criteria))
    
    for i in range(num_alternatives):
        st.markdown(f"**{alternatives[i]}**")
        cols = st.columns(num_criteria)
        for j in range(num_criteria):
            val = st.session_state.model["decision_matrix"].get((i, j), 0.0)
            new_val = cols[j].number_input(
                f"{criteria_names[j]}", 
                value=float(val), 
                key=f"dm_p_{i}_{j}",
                on_change=update_model, args=(f"dm_p_{i}_{j}", i, j)
            )
            decision_matrix[i, j] = new_val

    if st.button("Calculate PROMETHEE Ranking"):
        try:
            if not use_ahp_weights:
                total_weight = sum(weights)
                if total_weight == 0:
                    st.error("Total weight cannot be zero.")
                    return
                weights = [w/total_weight for w in weights]
            
            df = pd.DataFrame(decision_matrix, index=alternatives, columns=criteria_names)
            
            promethee = PrometheeEngine(df, weights, impacts, pref_funcs, p_params, q_params)
            results = promethee.calculate_flows()
            
            st.success("Ranking Calculated!")
            st.dataframe(results.style.highlight_max(axis=0, subset=['Net Phi']), width='stretch')
            
            st.bar_chart(results.set_index('Alternative')['Net Phi'])
            
            # Export Results
            st.markdown("---")
            st.subheader("📊 Export Results")
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                csv_data = ScenarioManager.export_results_csv(results, "promethee")
                st.download_button(
                    "📄 Download CSV",
                    data=csv_data,
                    file_name="promethee_results.csv",
                    mime="text/csv"
                )
            with col_exp2:
                excel_data = ScenarioManager.export_results_excel(
                    {"scenario_name": scenario_name, "num_criteria": num_criteria, "num_alternatives": num_alternatives, "criteria_names": criteria_names},
                    {"ranking": results}
                )
                st.download_button(
                    "📊 Download Excel",
                    data=excel_data,
                    file_name="promethee_results.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
        except Exception as e:
            st.error(f"Error: {e}")

def render_anp():
    st.header("Analytic Network Process (ANP)")
    st.info("Model complex decisions with dependencies and feedback loops.")
    
    # Initialize ANP State
    if "anp_clusters" not in st.session_state:
        st.session_state.anp_clusters = ["Goal", "Criteria", "Alternatives"]
    
    # Sync with Main Model (Always update to reflect changes in other tabs)
    # We use setdefault to ensure structure exists, but overwrite content for sync
    if "anp_nodes" not in st.session_state:
        st.session_state.anp_nodes = {}
        
    # Force sync of Criteria and Alternatives from main model
    st.session_state.anp_nodes["Goal"] = ["Best Choice"]
    st.session_state.anp_nodes["Criteria"] = st.session_state.model["criteria_names"]
    st.session_state.anp_nodes["Alternatives"] = st.session_state.model["alternatives"]
    if "anp_connections" not in st.session_state:
        st.session_state.anp_connections = [] # List of (source, target)
        
    tab1, tab2, tab3 = st.tabs(["1. Network Definition", "2. Supermatrix Input", "3. Results"])
    
    with tab1:
        st.subheader("Define Clusters & Nodes")
        
        # Cluster Management
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            new_cluster = st.text_input("New Cluster Name")
        with col_c2:
            if st.button("Add Cluster"):
                if new_cluster and new_cluster not in st.session_state.anp_clusters:
                    st.session_state.anp_clusters.append(new_cluster)
                    st.session_state.anp_nodes[new_cluster] = []
                    st.rerun()
        
        # Node Management
        selected_cluster = st.selectbox("Select Cluster to Edit", st.session_state.anp_clusters)
        
        col_n1, col_n2 = st.columns([3, 1])
        with col_n1:
            new_node = st.text_input("New Node Name")
        with col_n2:
            if st.button("Add Node"):
                if new_node and new_node not in st.session_state.anp_nodes[selected_cluster]:
                    st.session_state.anp_nodes[selected_cluster].append(new_node)
                    st.rerun()
                    
        st.write(f"**Nodes in {selected_cluster}:**", ", ".join(st.session_state.anp_nodes[selected_cluster]))
        
        st.markdown("---")
        st.subheader("Define Influences (Connections)")
        st.caption("Select a Target Node and check the Source Nodes that influence it.")
        
        # Flatten nodes for selection
        all_nodes = []
        for c in st.session_state.anp_clusters:
            all_nodes.extend(st.session_state.anp_nodes[c])
            
        target_node = st.selectbox("Target Node (Who is influenced?)", all_nodes)
        
        # Pre-select existing connections
        current_sources = [src for src, tgt in st.session_state.anp_connections if tgt == target_node]
        
        with st.form("connections_form"):
            sources = st.multiselect("Source Nodes (Who influences the target?)", all_nodes, default=current_sources)
            if st.form_submit_button("Save Connections"):
                # Remove old connections for this target
                st.session_state.anp_connections = [c for c in st.session_state.anp_connections if c[1] != target_node]
                # Add new
                for src in sources:
                    st.session_state.anp_connections.append((src, target_node))
                st.success(f"Connections for {target_node} updated!")
                
    with tab2:
        st.subheader("Unweighted Supermatrix")
        st.caption("Enter the influence weights directly. Columns must sum to 1 (Stochastic).")
        
        # Build DataFrame for Supermatrix
        n = len(all_nodes)
        if n > 0:
            # Initialize if not exists or size changed
            if "anp_supermatrix" not in st.session_state or st.session_state.anp_supermatrix.shape != (n, n):
                st.session_state.anp_supermatrix = np.zeros((n, n))
            
            # Create a DataFrame for editing
            df_sm = pd.DataFrame(
                st.session_state.anp_supermatrix,
                index=all_nodes,
                columns=all_nodes
            )
            
            # Highlight valid connections (optional visual cue)
            edited_df = st.data_editor(df_sm, key="sm_editor", height=400)
            
            # Update session state
            st.session_state.anp_supermatrix = edited_df.values
            
            # Validation
            col_sums = edited_df.sum(axis=0)
            if not np.allclose(col_sums, 1.0) and not np.allclose(col_sums, 0.0):
                st.warning("Warning: Some columns do not sum to 1. Results may be inaccurate.")
        else:
            st.info("Define nodes first.")
            
    with tab3:
        st.subheader("Results")
        if st.button("Calculate ANP"):
            try:
                engine = ANPEngine(
                    st.session_state.anp_clusters,
                    st.session_state.anp_nodes,
                    st.session_state.anp_connections
                )
                
                # Set the matrix from Step 2
                engine.set_unweighted_supermatrix(st.session_state.anp_supermatrix)
                
                # Calculate Limit Matrix
                limit_matrix = engine.calculate_limit_matrix()
                priorities = engine.get_priorities()
                
                st.success("Calculation Complete!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Global Priorities")
                    # Sort by priority
                    sorted_priorities = sorted(priorities.items(), key=lambda x: x[1], reverse=True)
                    df_p = pd.DataFrame(sorted_priorities, columns=["Node", "Priority"])
                    st.dataframe(df_p, use_container_width=True)
                    
                with col2:
                    st.markdown("### Limit Matrix")
                    st.dataframe(pd.DataFrame(limit_matrix, index=all_nodes, columns=all_nodes))
                    
            except Exception as e:
                st.error(f"Error: {e}")

def render_group_ahp():
    st.header("Group Decision Aggregator")
    st.info("Upload multiple JSON scenario files to aggregate judgments (AHP/Fuzzy) and/or decision matrices (TOPSIS/PROMETHEE).")
    
    uploaded_files = st.file_uploader("Upload Judge Files (JSON)", type="json", accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) < 2:
            st.warning("Please upload at least 2 files to perform aggregation.")
            return
            
        pairwise_matrices = []
        decision_matrices = []
        judge_names = []
        criteria_names = None
        alternatives = None
        
        try:
            for file in uploaded_files:
                data = json.load(file)
                judge_names.append(data.get("scenario_name", file.name))
                
                # Check Metadata
                c_names = data.get("criteria_names", [])
                alt_names = data.get("alternative_names", []) # Assuming this exists or we infer
                
                if criteria_names is None:
                    criteria_names = c_names
                elif c_names != criteria_names:
                    st.error(f"Criteria mismatch in file {file.name}.")
                    return
                    
                # Extract Pairwise Matrix (if exists)
                pairwise = data.get("ahp_pairwise", {})
                if pairwise:
                    n = data.get("num_criteria", 3)
                    # Check if fuzzy by looking at first item
                    is_fuzzy_input = False
                    first_val = list(pairwise.values())[0]['intensity'] if pairwise else 1
                    if isinstance(first_val, list) or (isinstance(first_val, (int, float)) and False): 
                        # Simplified check, real check below
                        pass
                        
                    matrix = np.empty((n, n), dtype=object) if "fuzzy" in str(data) or isinstance(list(pairwise.values())[0]['intensity'], list) else np.ones((n, n))
                    
                    # Fill diagonal
                    for i in range(n):
                        matrix[i, i] = (1, 1, 1) if matrix.dtype == object else 1.0
                        
                    for key, val in pairwise.items():
                        parts = key.split("_")
                        if len(parts) == 2:
                            i, j = int(parts[0]), int(parts[1])
                            winner = val['winner']
                            intensity = val['intensity']
                            
                            # Handle Fuzzy TFN
                            if isinstance(intensity, list):
                                tfn = tuple(intensity)
                                if winner == criteria_names[i]:
                                    matrix[i, j] = tfn
                                    if tfn[0]!=0 and tfn[1]!=0 and tfn[2]!=0:
                                        matrix[j, i] = (1/tfn[2], 1/tfn[1], 1/tfn[0])
                                    else:
                                        matrix[j, i] = (0,0,0)
                                elif winner == criteria_names[j]:
                                    matrix[j, i] = tfn
                                    if tfn[0]!=0 and tfn[1]!=0 and tfn[2]!=0:
                                        matrix[i, j] = (1/tfn[2], 1/tfn[1], 1/tfn[0])
                                    else:
                                        matrix[i, j] = (0,0,0)
                            # Handle Crisp
                            else:
                                if winner == criteria_names[i]:
                                    matrix[i, j] = intensity
                                    matrix[j, i] = 1 / intensity
                                elif winner == criteria_names[j]:
                                    matrix[i, j] = 1 / intensity
                                    matrix[j, i] = intensity
                    
                    pairwise_matrices.append(matrix)
                
                # Extract Decision Matrix (if exists)
                dm = data.get("decision_matrix", {})
                if dm:
                    num_alt = data.get("num_alternatives", 3)
                    num_crit = data.get("num_criteria", 3)
                    d_mat = np.zeros((num_alt, num_crit))
                    
                    for key, val in dm.items():
                        parts = key.split("_")
                        if len(parts) == 2:
                            r, c = int(parts[0]), int(parts[1])
                            d_mat[r, c] = val
                    
                    decision_matrices.append(d_mat)
            
            st.success(f"Loaded files from: {', '.join(judge_names)}")
            st.info(f"Found {len(pairwise_matrices)} Pairwise Matrices and {len(decision_matrices)} Decision Matrices.")
            
            if st.button("Aggregate & Calculate"):
                group_engine = GroupDecisionEngine(
                    matrices=pairwise_matrices if pairwise_matrices else None,
                    decision_matrices=decision_matrices if decision_matrices else None,
                    criteria_names=criteria_names
                )
                results = group_engine.get_results()
                
                st.subheader("Group Results")
                
                # Display Weights
                if "weights" in results:
                    st.markdown("### Aggregated Weights (Geometric Mean)")
                    if results.get("is_fuzzy"):
                        st.caption("Derived from Fuzzy AHP aggregation")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        weights_df = pd.DataFrame(list(results['weights'].items()), columns=['Criterion', 'Weight'])
                        st.dataframe(weights_df)
                        csv_data = ScenarioManager.export_results_csv(results['weights'], "weights")
                        st.download_button("Download Group Weights (CSV)", csv_data, "group_weights.csv", "text/csv")
                    with col2:
                        st.metric("Group Consistency Ratio", f"{results['consistency_ratio']:.4f}")

                # Display Decision Matrix
                if "aggregated_decision_matrix" in results:
                    st.markdown("---")
                    st.markdown("### Aggregated Decision Matrix (Arithmetic Mean)")
                    adm = results["aggregated_decision_matrix"]
                    df_adm = pd.DataFrame(adm, columns=criteria_names)
                    st.dataframe(df_adm)
                    
                    # Option to use this data
                    if st.button("Use Aggregated Data for Analysis"):
                        st.session_state.model["ahp_weights"] = results.get("weights")
                        # Flatten decision matrix to session state format
                        for r in range(adm.shape[0]):
                            for c in range(adm.shape[1]):
                                st.session_state.model["decision_matrix"][(r, c)] = adm[r, c]
                        st.success("Data loaded into session! Go to TOPSIS or PROMETHEE mode to see results.")

        except Exception as e:
            st.error(f"Error processing files: {e}")

if mode == "AHP (Weights)":
    render_ahp(fuzzy=False)
elif mode == "Fuzzy AHP (Weights)":
    render_ahp(fuzzy=True)
elif mode == "TOPSIS (Ranking)":
    render_topsis(use_ahp_weights=False)
elif mode == "PROMETHEE (Ranking)":
    render_promethee(use_ahp_weights=False)
elif mode == "Combined (AHP + TOPSIS)":
    st.markdown("### Step 1: Calculate Weights with AHP")
    render_ahp(fuzzy=False)
    st.markdown("---")
    st.markdown("### Step 2: Rank Alternatives with TOPSIS")
    if st.session_state.model["ahp_weights"]:
        render_topsis(use_ahp_weights=True)
    else:
        st.info("Please calculate weights in Step 1 to proceed.")
elif mode == "Combined (Fuzzy AHP + TOPSIS)":
    st.markdown("### Step 1: Calculate Weights with Fuzzy AHP")
    render_ahp(fuzzy=True)
    st.markdown("---")
    st.markdown("### Step 2: Rank Alternatives with TOPSIS")
    if st.session_state.model["ahp_weights"]:
        render_topsis(use_ahp_weights=True)
    else:
        st.info("Please calculate weights in Step 1 to proceed.")
elif mode == "Combined (AHP + PROMETHEE)":
    st.markdown("### Step 1: Calculate Weights with AHP")
    render_ahp(fuzzy=False)
    st.markdown("---")
    st.markdown("### Step 2: Rank Alternatives with PROMETHEE")
    if st.session_state.model["ahp_weights"]:
        render_promethee(use_ahp_weights=True)
    else:
        st.info("Please calculate weights in Step 1 to proceed.")
elif mode == "Group Decision Aggregator":
    render_group_ahp()
elif mode == "ANP (Network)":
    render_anp()
