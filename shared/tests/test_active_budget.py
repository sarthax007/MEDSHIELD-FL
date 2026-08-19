import pytest
from shared.medshield.active.budget import BudgetManager

def test_budget_initialization():
    manager = BudgetManager(10)
    assert manager.get_remaining_budget() == 10
    
    with pytest.raises(ValueError):
        BudgetManager(-1)

def test_get_allowed_batch_size():
    manager = BudgetManager(10)
    assert manager.get_allowed_batch_size(5) == 5
    assert manager.get_allowed_batch_size(15) == 10

def test_consume_budget():
    manager = BudgetManager(10)
    
    # Consume partial
    manager.consume_budget(4)
    assert manager.get_remaining_budget() == 6
    assert manager.get_allowed_batch_size(10) == 6
    
    # Consume exactly remaining
    manager.consume_budget(6)
    assert manager.get_remaining_budget() == 0
    assert manager.get_allowed_batch_size(5) == 0
    
    # Over consume
    with pytest.raises(ValueError):
        manager.consume_budget(1)

def test_reset_budget():
    manager = BudgetManager(15)
    manager.consume_budget(10)
    assert manager.get_remaining_budget() == 5
    
    manager.reset_budget()
    assert manager.get_remaining_budget() == 15
