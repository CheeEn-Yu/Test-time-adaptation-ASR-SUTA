from strategies.choose_ln import ChooseLNStrategy
# 導入您未來會創建的其他策略

class StrategyRegistry:
    """Registry for adaptation strategies."""
    
    def __init__(self):
        self.strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """Register built-in strategies."""
        self.register("choose_ln", ChooseLNStrategy())
        # 註冊其他策略
    
    def register(self, name, strategy):
        """Register a new strategy."""
        self.strategies[name] = strategy
    
    def get(self, name):
        """Get a strategy by name."""
        if name not in self.strategies:
            raise ValueError(f"Strategy '{name}' not found. Available strategies: {list(self.strategies.keys())}")
        return self.strategies[name]
    
    def list_strategies(self):
        """List all available strategies."""
        return list(self.strategies.keys())

# 創建一個單例實例
strategy_registry = StrategyRegistry()