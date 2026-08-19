class BudgetManager:
    def __init__(self, initial_budget: int):
        if initial_budget < 0:
            raise ValueError("Budget cannot be negative.")
        self._initial_budget = initial_budget
        self._remaining_budget = initial_budget

    def get_allowed_batch_size(self, requested_size: int) -> int:
        if requested_size < 0:
            return 0
        return min(requested_size, self._remaining_budget)

    def consume_budget(self, amount: int):
        if amount < 0:
            raise ValueError("Amount to consume cannot be negative.")
        if amount > self._remaining_budget:
            raise ValueError(f"Cannot consume {amount}, only {self._remaining_budget} remaining.")
        self._remaining_budget -= amount

    def get_remaining_budget(self) -> int:
        return self._remaining_budget

    def reset_budget(self):
        self._remaining_budget = self._initial_budget
