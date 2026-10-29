import os
import pytest
from decision_engine.case_repository import CaseRepository

def test_case_creation_and_retrieval(tmp_path):
    db_file = str(tmp_path / "test_cases.db")
    repo = CaseRepository(db_path=db_file)
    
    config = {
        "criteria": ["Costo", "Calidad", "Tiempo"],
        "alternatives": ["Proveedor A", "Proveedor B"]
    }
    case_id = repo.create_case(
        title="Selección de Proveedor",
        evaluation_type="AHP",
        config_data=config,
        description="Evaluar proveedores para el nuevo proyecto"
    )
    
    assert case_id is not None
    assert len(case_id) == 8
    
    case = repo.get_case(case_id)
    assert case is not None
    assert case["title"] == "Selección de Proveedor"
    assert case["evaluation_type"] == "AHP"
    assert case["config"]["criteria"] == ["Costo", "Calidad", "Tiempo"]

def test_submit_and_list_evaluations(tmp_path):
    db_file = str(tmp_path / "test_cases.db")
    repo = CaseRepository(db_path=db_file)
    
    case_id = repo.create_case("Caso 1", "AHP", {"criteria": ["C1", "C2"]})
    
    # Judge 1 submits
    eval1 = {"ahp_pairwise": {"0_1": {"winner": "C1", "intensity": 3}}}
    repo.submit_evaluation(case_id, "Juez Ana", eval1)
    
    # Judge 2 submits
    eval2 = {"ahp_pairwise": {"0_1": {"winner": "C2", "intensity": 2}}}
    repo.submit_evaluation(case_id, "Juez Carlos", eval2)
    
    evaluations = repo.get_case_evaluations(case_id)
    assert len(evaluations) == 2
    assert evaluations[0]["judge_name"] == "Juez Ana"
    assert evaluations[1]["judge_name"] == "Juez Carlos"
    
    # Judge 1 updates evaluation
    eval1_updated = {"ahp_pairwise": {"0_1": {"winner": "C1", "intensity": 5}}}
    repo.submit_evaluation(case_id, "Juez Ana", eval1_updated)
    
    evaluations_after = repo.get_case_evaluations(case_id)
    assert len(evaluations_after) == 2
    # Find Juez Ana's evaluation
    ana_eval = next(e for e in evaluations_after if e["judge_name"] == "Juez Ana")
    assert ana_eval["evaluation"]["ahp_pairwise"]["0_1"]["intensity"] == 5
