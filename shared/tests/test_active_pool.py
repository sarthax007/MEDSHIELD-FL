import os
import tempfile
import pytest
from shared.medshield.active.pool import DataPoolManager

def test_pool_initialization():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        state_path = tmp.name
        
    try:
        initial_items = ["img_1", "img_2", "img_3"]
        initial_labelled = ["img_1"]
        
        manager = DataPoolManager(state_path, initial_items=initial_items, initial_labelled=initial_labelled)
        
        sizes = manager.get_pool_sizes()
        assert sizes["labelled"] == 1
        assert sizes["unlabeled"] == 2
        
        assert "img_1" in manager.get_labelled_pool()
        assert "img_2" in manager.get_unlabeled_pool()
        assert "img_3" in manager.get_unlabeled_pool()
        
    finally:
        if os.path.exists(state_path):
            os.remove(state_path)

def test_submit_label():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        state_path = tmp.name
        
    try:
        initial_items = ["img_1", "img_2", "img_3", "img_4"]
        manager = DataPoolManager(state_path, initial_items=initial_items)
        
        # Move items
        manager.submit_label("img_2", label=1, user_id="doc1")
        manager.submit_label("img_4", label=0, user_id="doc2")
        
        sizes = manager.get_pool_sizes()
        assert sizes["labelled"] == 2
        assert sizes["unlabeled"] == 2
        
        assert set(manager.get_labelled_pool()) == {"img_2", "img_4"}
        assert set(manager.get_unlabeled_pool()) == {"img_1", "img_3"}
        
        # Check labels and audit log
        labels = manager.get_labels()
        assert labels["img_2"] == 1
        assert labels["img_4"] == 0
        
        audit_log = manager.get_audit_log()
        assert len(audit_log) == 2
        assert audit_log[0]["item_id"] == "img_2"
        assert audit_log[0]["user_id"] == "doc1"
        assert audit_log[1]["item_id"] == "img_4"
        assert audit_log[1]["user_id"] == "doc2"
        assert "timestamp" in audit_log[0]
        
    finally:
        if os.path.exists(state_path):
            os.remove(state_path)

def test_state_persistence():
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        state_path = tmp.name
        
    try:
        initial_items = ["img_1", "img_2", "img_3"]
        
        # First instance
        manager1 = DataPoolManager(state_path, initial_items=initial_items)
        manager1.submit_label("img_1", label=2, user_id="doc3")
        
        # Second instance (should load state from disk)
        manager2 = DataPoolManager(state_path)
        
        assert manager2.get_pool_sizes()["labelled"] == 1
        assert manager2.get_pool_sizes()["unlabeled"] == 2
        assert "img_1" in manager2.get_labelled_pool()
        assert manager2.get_labels()["img_1"] == 2
        assert manager2.get_audit_log()[0]["user_id"] == "doc3"
        
    finally:
        if os.path.exists(state_path):
            os.remove(state_path)
