"""Semantic parser for autoformalization: NL → formal logic (Datalog, Z3, etc.)."""

from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import re

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    print("Warning: Z3 not available. Install with: pip install z3-solver")


@dataclass
class FormalExpression:
    """Formal logic representation of a statement."""
    expression: str  # The formal expression
    formal_type: str  # "datalog" | "z3" | "fol" | "lean" | "coq"
    confidence: float = 0.5
    parse_success: bool = False


class SemanticParser:
    """Parser that converts natural language to formal logic."""
    
    def __init__(self, use_llm: bool = False, llm_model=None):
        self.use_llm = use_llm
        self.llm_model = llm_model
    
    def parse(self, text: str, target_type: str = "datalog") -> FormalExpression:
        """
        Parse natural language text to formal expression.
        
        Args:
            text: Natural language statement
            target_type: Target formal type ("datalog", "z3", "fol")
        
        Returns:
            FormalExpression with parsed result
        """
        if self.use_llm and self.llm_model:
            return self._llm_parse(text, target_type)
        else:
            return self._rule_based_parse(text, target_type)
    
    def _rule_based_parse(self, text: str, target_type: str) -> FormalExpression:
        """Rule-based parsing for common patterns."""
        text_lower = text.lower().strip()
        
        # Pattern: "X is Y" -> Datalog: is(X, Y) or Z3: x == y
        if target_type == "datalog":
            # Simple pattern matching
            match = re.match(r"(.+?)\s+is\s+(.+?)\.?$", text_lower)
            if match:
                x, y = match.groups()
                x_clean = x.strip().replace(" ", "_").replace("-", "_")
                y_clean = y.strip().replace(" ", "_").replace("-", "_")
                expr = f"is({x_clean}, {y_clean})"
                return FormalExpression(expr, "datalog", confidence=0.7, parse_success=True)
        
        elif target_type == "z3":
            # Try to extract numeric/arithmetic patterns
            # Pattern: "X is greater than Y" -> x > y
            if "greater than" in text_lower or ">" in text:
                vars_match = re.findall(r'\b(\d+|[a-z]+\w*)\b', text_lower)
                if len(vars_match) >= 2:
                    x, y = vars_match[0], vars_match[1]
                    expr = f"{x} > {y}"
                    return FormalExpression(expr, "z3", confidence=0.6, parse_success=True)
            
            # Pattern: "X equals Y" -> x == y
            if "equals" in text_lower or "==" in text or "is equal to" in text_lower:
                vars_match = re.findall(r'\b(\d+|[a-z]+\w*)\b', text_lower)
                if len(vars_match) >= 2:
                    x, y = vars_match[0], vars_match[1]
                    expr = f"{x} == {y}"
                    return FormalExpression(expr, "z3", confidence=0.6, parse_success=True)
        
        # Fallback: return as-is with low confidence
        return FormalExpression(text, target_type, confidence=0.3, parse_success=False)
    
    def _llm_parse(self, text: str, target_type: str) -> FormalExpression:
        """Use LLM for parsing (more accurate but slower)."""
        if not self.llm_model:
            return self._rule_based_parse(text, target_type)
        
        prompt = f"""Convert the following natural language statement to {target_type} format.

Statement: {text}

Format the output as a {target_type} expression. For example:
- Datalog: predicate(subject, object)
- Z3: x > 5 or And(x > 0, y < 10)
- FOL: ∀x. P(x) → Q(x)

Output only the {target_type} expression, nothing else."""
        
        # Generate with LLM (placeholder - would need actual LLM call)
        # For now, fallback to rule-based
        return self._rule_based_parse(text, target_type)
    
    def parse_with_multiple_targets(self, text: str) -> List[FormalExpression]:
        """Try parsing to multiple formal types and return best match."""
        results = []
        for target_type in ["datalog", "z3", "fol"]:
            result = self.parse(text, target_type)
            if result.parse_success:
                results.append(result)
        
        return sorted(results, key=lambda x: x.confidence, reverse=True)


class Z3Checker:
    """Z3 SMT solver integration for verification."""
    
    def __init__(self):
        if not Z3_AVAILABLE:
            raise ImportError("Z3 not available. Install with: pip install z3-solver")
        self.solver = z3.Solver()
    
    def check_entailment(
        self,
        premises: List[str],
        conclusion: str,
        variable_map: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if conclusion follows from premises using Z3.
        
        Args:
            premises: List of Z3 expressions (as strings or z3 objects)
            conclusion: Conclusion expression
            variable_map: Mapping of variable names to z3 variables
        
        Returns:
            (is_valid, counterexample_or_proof)
        """
        try:
            solver = z3.Solver()
            
            # Parse premises
            premise_constraints = []
            for prem in premises:
                try:
                    # Try to parse as Z3 expression
                    if isinstance(prem, str):
                        # Simple parsing (in production, use proper parser)
                        if ">" in prem:
                            parts = prem.split(">")
                            if len(parts) == 2:
                                x, y = parts[0].strip(), parts[1].strip()
                                var_x = z3.Int(x) if not y.isdigit() else z3.Int(x)
                                var_y = z3.IntVal(int(y)) if y.isdigit() else z3.Int(y)
                                premise_constraints.append(var_x > var_y)
                        elif "==" in prem or "=" in prem:
                            parts = prem.split("==") if "==" in prem else prem.split("=")
                            if len(parts) == 2:
                                x, y = parts[0].strip(), parts[1].strip()
                                var_x = z3.Int(x)
                                var_y = z3.IntVal(int(y)) if y.isdigit() else z3.Int(y)
                                premise_constraints.append(var_x == var_y)
                        else:
                            # Add as assumption (simplified)
                            continue
                    else:
                        premise_constraints.append(prem)
                except:
                    continue
            
            # Add premises to solver
            for constraint in premise_constraints:
                solver.add(constraint)
            
            # Negate conclusion and check for satisfiability
            # If premises + !conclusion is UNSAT, then conclusion follows
            try:
                if isinstance(conclusion, str):
                    # Parse conclusion similarly
                    if ">" in conclusion:
                        parts = conclusion.split(">")
                        if len(parts) == 2:
                            x, y = parts[0].strip(), parts[1].strip()
                            var_x = z3.Int(x)
                            var_y = z3.IntVal(int(y)) if y.isdigit() else z3.Int(y)
                            negated_conclusion = z3.Not(var_x > var_y)
                    elif "==" in conclusion or "=" in conclusion:
                        parts = conclusion.split("==") if "==" in conclusion else conclusion.split("=")
                        if len(parts) == 2:
                            x, y = parts[0].strip(), parts[1].strip()
                            var_x = z3.Int(x)
                            var_y = z3.IntVal(int(y)) if y.isdigit() else z3.Int(y)
                            negated_conclusion = z3.Not(var_x == var_y)
                    else:
                        return False, "Could not parse conclusion"
                    
                    solver.add(negated_conclusion)
                else:
                    solver.add(z3.Not(conclusion))
            except:
                return False, "Could not parse conclusion"
            
            # Check
            result = solver.check()
            
            if result == z3.unsat:
                # UNSAT means premises entail conclusion
                return True, "Proof: premises + !conclusion is UNSAT"
            elif result == z3.sat:
                # SAT means we found a counterexample
                model = solver.model()
                return False, f"Counterexample: {model}"
            else:
                return False, "Unknown"
        
        except Exception as e:
            return False, f"Z3 error: {str(e)}"


class DatalogChecker:
    """Datalog engine for rule-based verification (simplified)."""
    
    def __init__(self):
        self.rules = []
        self.facts = []
    
    def add_rule(self, rule: str):
        """Add a Datalog rule."""
        self.rules.append(rule)
    
    def add_fact(self, fact: str):
        """Add a Datalog fact."""
        self.facts.append(fact)
    
    def check_entailment(self, query: str) -> Tuple[bool, Optional[str]]:
        """Check if query is entailed by facts and rules."""
        # Simplified Datalog evaluation
        # In production, use a proper Datalog engine like Soufflé
        
        # For now, simple pattern matching
        query_predicate = query.split("(")[0] if "(" in query else query
        
        # Check facts
        for fact in self.facts:
            if fact.startswith(query_predicate):
                return True, f"Fact found: {fact}"
        
        # Check rules (simplified)
        # In real implementation, would do proper resolution
        
        return False, "Not entailed"

