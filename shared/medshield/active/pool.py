import json
import os
import datetime
from typing import List, Dict, Optional

class DataPoolManager:
    def __init__(self, state_file_path: str, initial_items: Optional[List[str]] = None, initial_labelled: Optional[List[str]] = None):
        """
        Initialize the DataPoolManager for a hospital.
        
        Args:
            state_file_path: Path to the JSON file where the pool state will be saved.
            initial_items: List of all item IDs available. Used if creating state for the first time.
            initial_labelled: List of item IDs that are initially labelled (optional).
        """
        self.state_file_path = state_file_path
        self.unlabeled_pool = set()
        self.labelled_pool = set()
        self.labels = {}
        self.audit_log = []
        
        self._load_or_initialize_state(initial_items, initial_labelled)

    def _load_or_initialize_state(self, initial_items: Optional[List[str]], initial_labelled: Optional[List[str]]):
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, 'r') as f:
                    state = json.load(f)
                self.unlabeled_pool = set(state.get("unlabeled_pool", []))
                self.labelled_pool = set(state.get("labelled_pool", []))
                self.labels = state.get("labels", {})
                self.audit_log = state.get("audit_log", [])
            except json.JSONDecodeError:
                # If file is corrupt, re-initialize if possible
                self._initialize_from_scratch(initial_items, initial_labelled)
        else:
            self._initialize_from_scratch(initial_items, initial_labelled)

    def _initialize_from_scratch(self, initial_items: Optional[List[str]], initial_labelled: Optional[List[str]]):
        if initial_items is None:
            initial_items = []
        if initial_labelled is None:
            initial_labelled = []
            
        initial_set = set(initial_items)
        init_labelled_set = set(initial_labelled)
        
        # Ensure initial labelled items are actually part of the items
        self.labelled_pool = init_labelled_set.intersection(initial_set)
        self.unlabeled_pool = initial_set - self.labelled_pool
        self.labels = {}
        self.audit_log = []
        
        self.save_state()

    def get_unlabeled_pool(self) -> List[str]:
        return list(self.unlabeled_pool)

    def get_labelled_pool(self) -> List[str]:
        return list(self.labelled_pool)
        
    def get_labels(self) -> Dict[str, int]:
        return self.labels
        
    def get_audit_log(self) -> List[Dict]:
        return self.audit_log

    def submit_label(self, item_id: str, label: int, user_id: str, timestamp: Optional[str] = None):
        """
        Move item from unlabeled to labelled pool, store the label and audit record.
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            
        if item_id in self.unlabeled_pool:
            self.unlabeled_pool.remove(item_id)
            self.labelled_pool.add(item_id)
            
        self.labels[item_id] = label
        self.audit_log.append({
            "item_id": item_id,
            "label": label,
            "user_id": user_id,
            "timestamp": timestamp
        })
        
        self.save_state()

    def label_items(self, item_ids: List[str]):
        """
        Legacy method. Used mostly in older tests. Moves items from unlabeled to labelled pool.
        For proper label tracking, use submit_label.
        """
        for item_id in item_ids:
            if item_id in self.unlabeled_pool:
                self.unlabeled_pool.remove(item_id)
                self.labelled_pool.add(item_id)
        
        self.save_state()

    def get_pool_sizes(self) -> Dict[str, int]:
        return {
            "labelled": len(self.labelled_pool),
            "unlabeled": len(self.unlabeled_pool)
        }

    def save_state(self):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.state_file_path)), exist_ok=True)
        
        state = {
            "labelled_pool": list(self.labelled_pool),
            "unlabeled_pool": list(self.unlabeled_pool),
            "labels": self.labels,
            "audit_log": self.audit_log
        }
        
        with open(self.state_file_path, 'w') as f:
            json.dump(state, f, indent=4)
