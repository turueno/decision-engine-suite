import streamlit as st
import pandas as pd
import numpy as np
from decision_engine.ahp import AHPEngine
from decision_engine.topsis import TOPSISEngine
from decision_engine.fuzzy_ahp import FuzzyAHPEngine
from decision_engine.promethee import PrometheeEngine

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
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
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
mode = st.sidebar.selectbox("Select Mode", [
    "AHP (Weights)", 
    "Fuzzy AHP (Weights)", 
    "TOPSIS (Ranking)", 
    "PROMETHEE (Ranking)",
    "Combined (AHP + TOPSIS)", 
    "Combined (Fuzzy AHP + TOPSIS)",
    "Combined (AHP + PROMETHEE)"
])

def render_ahp(fuzzy=False):
    if fuzzy:
        st.header("Fuzzy Analytic Hierarchy Process")
        st.info("Uses Triangular Fuzzy Numbers to handle uncertainty in judgments.")
    else:
        st.header("Analytic Hierarchy Process (AHP)")
    
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
                    st.write("Equal Importance")
                else:
                    intensity = st.slider(
                        f"Intensity for {winner}",
                        min_value=1, max_value=9, 
                        value=int(saved_pair['intensity']),
                        key=f"int_{i}_{j}"
                    )
            
            # Update model immediately
            st.session_state.model["ahp_pairwise"][pair_key] = {'winner': winner, 'intensity': intensity}
            
            # Update local matrix for calculation
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
                engine = FuzzyAHPEngine(matrix, criteria_names)
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
                else:
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
            st.dataframe(results.style.highlight_max(axis=0, subset=['Net Phi']), use_container_width=True)
            
            st.bar_chart(results.set_index('Alternative')['Net Phi'])
            
        except Exception as e:
            st.error(f"Error: {e}")

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
