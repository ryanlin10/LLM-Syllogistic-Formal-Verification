"""Generate atomic propositions using Anthropic Claude API with pooling."""

import os
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class AtomicProposition:
    """A single atomic proposition."""
    text: str
    topic: str
    is_predicate: bool = False  # True if it's a predicate like "is tall", False if full statement


@dataclass
class EntityPool:
    """Pool of entity names for first-order logic."""
    names: List[str] = field(default_factory=list)

    def sample(self, n: int = 1, exclude: Optional[List[str]] = None) -> List[str]:
        """Sample n distinct entities, excluding specified ones."""
        available = [e for e in self.names if not exclude or e not in exclude]
        return random.sample(available, min(n, len(available)))


@dataclass
class PropositionPool:
    """Pool of atomic propositions for efficient generation."""
    # Full propositions for propositional logic (e.g., "it is raining")
    propositions: List[AtomicProposition] = field(default_factory=list)
    # Predicates for first-order logic (e.g., "tall", "red", "made of metal")
    predicates: List[AtomicProposition] = field(default_factory=list)
    # Relations for first-order logic (e.g., "loves", "is taller than")
    relations: List[AtomicProposition] = field(default_factory=list)
    # Categories/types (e.g., "mammals", "vehicles", "fruits")
    categories: List[AtomicProposition] = field(default_factory=list)
    # Entity names
    entities: EntityPool = field(default_factory=EntityPool)

    def sample_propositions(self, n: int, exclude: Optional[List[str]] = None) -> List[AtomicProposition]:
        """Sample n distinct propositions."""
        available = [p for p in self.propositions if not exclude or p.text not in exclude]
        return random.sample(available, min(n, len(available)))

    def sample_predicates(self, n: int, exclude: Optional[List[str]] = None) -> List[AtomicProposition]:
        """Sample n distinct predicates."""
        available = [p for p in self.predicates if not exclude or p.text not in exclude]
        return random.sample(available, min(n, len(available)))

    def sample_relations(self, n: int = 1) -> List[AtomicProposition]:
        """Sample n distinct relations."""
        return random.sample(self.relations, min(n, len(self.relations)))

    def sample_categories(self, n: int = 1) -> List[AtomicProposition]:
        """Sample n distinct categories."""
        return random.sample(self.categories, min(n, len(self.categories)))


@dataclass
class GeneratorConfig:
    """Configuration for atomic proposition generation."""
    api_key: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.9
    retry_attempts: int = 3
    retry_delay: float = 1.0
    # Pool sizes
    propositions_per_topic: int = 20
    predicates_per_topic: int = 15
    relations_count: int = 30
    categories_count: int = 20
    entities_count: int = 50


# Topic categories for diverse generation
TOPIC_CATEGORIES = [
    "everyday life and routines",
    "science and nature",
    "technology and computing",
    "sports and fitness",
    "food and cooking",
    "travel and geography",
    "history and culture",
    "business and economics",
    "health and medicine",
    "arts and entertainment",
    "education and learning",
    "animals and wildlife",
    "weather and climate",
    "transportation and vehicles",
    "relationships and social dynamics",
]


class AtomicPropositionGenerator:
    """Generate pools of atomic propositions using Claude API."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.client = None
        self.pool: Optional[PropositionPool] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize Anthropic client."""
        if anthropic is None:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Set via config or environment variable.")

        self.client = anthropic.Anthropic(api_key=api_key)

    def generate_pool(
        self,
        topics: Optional[List[str]] = None,
        verbose: bool = True
    ) -> PropositionPool:
        """
        Generate a pool of atomic propositions with a single API call per category.

        This makes only a few API calls total, then all examples are generated
        by sampling from this pool.
        """
        topics = topics or TOPIC_CATEGORIES
        pool = PropositionPool()

        if verbose:
            print("Generating proposition pool...")

        # 1. Generate full propositions (for propositional logic)
        if verbose:
            print("  Generating propositions...")
        propositions = self._generate_propositions(topics)
        pool.propositions = propositions
        if verbose:
            print(f"    Generated {len(propositions)} propositions")

        # 2. Generate predicates (for first-order logic)
        if verbose:
            print("  Generating predicates...")
        predicates = self._generate_predicates(topics)
        pool.predicates = predicates
        if verbose:
            print(f"    Generated {len(predicates)} predicates")

        # 3. Generate relations (for relational reasoning)
        if verbose:
            print("  Generating relations...")
        relations = self._generate_relations()
        pool.relations = relations
        if verbose:
            print(f"    Generated {len(relations)} relations")

        # 4. Generate categories (for universal statements)
        if verbose:
            print("  Generating categories...")
        categories = self._generate_categories()
        pool.categories = categories
        if verbose:
            print(f"    Generated {len(categories)} categories")

        # 5. Generate entity names
        if verbose:
            print("  Generating entity names...")
        entities = self._generate_entities()
        pool.entities = EntityPool(names=entities)
        if verbose:
            print(f"    Generated {len(entities)} entity names")

        self.pool = pool
        if verbose:
            print("Pool generation complete!")

        return pool

    def _generate_propositions(self, topics: List[str]) -> List[AtomicProposition]:
        """Generate full propositions for propositional logic."""
        prompt = f"""Generate a diverse list of simple, atomic propositions that can be true or false.
These will be used for logic reasoning examples.

Topics to cover: {', '.join(topics[:5])}

Requirements:
- Each proposition should be a simple declarative statement
- Propositions should be concrete and specific, not abstract
- Include cause-effect pairs (e.g., "it is raining" and "the ground is wet")
- Keep each proposition under 10 words
- Generate at least {self.config.propositions_per_topic * len(topics[:5])} propositions

Respond with a JSON array of strings only:
["proposition 1", "proposition 2", ...]"""

        content = self._call_api(prompt)
        propositions = self._parse_string_array(content)

        return [
            AtomicProposition(text=p, topic="mixed", is_predicate=False)
            for p in propositions
        ]

    def _generate_predicates(self, topics: List[str]) -> List[AtomicProposition]:
        """Generate predicates for first-order logic (e.g., 'tall', 'red', 'edible')."""
        prompt = f"""Generate a diverse list of predicates (properties/adjectives) that can describe things.
These will be used in first-order logic statements like "X is [predicate]".

Topics to cover: {', '.join(topics[:5])}

Requirements:
- Each predicate should be a single property or short phrase
- Mix of physical properties, states, categories, and abstract qualities
- Generate predicates that can form logical chains (e.g., "made of metal" implies "conducts electricity")
- Keep each predicate under 5 words
- Generate at least {self.config.predicates_per_topic * len(topics[:5])} predicates

Respond with a JSON array of strings only:
["predicate 1", "predicate 2", ...]"""

        content = self._call_api(prompt)
        predicates = self._parse_string_array(content)

        return [
            AtomicProposition(text=p, topic="mixed", is_predicate=True)
            for p in predicates
        ]

    def _generate_relations(self) -> List[AtomicProposition]:
        """Generate relations for relational reasoning (e.g., 'loves', 'is taller than')."""
        prompt = f"""Generate a list of binary relations that can hold between two entities.
These will be used in statements like "A [relation] B".

Requirements:
- Include various types: social, physical, temporal, causal
- Each relation should be a verb or verb phrase
- Some should imply properties (e.g., "teaches" implies the first entity is knowledgeable)
- Generate at least {self.config.relations_count} relations

Respond with a JSON array of strings only:
["loves", "is taller than", "teaches", ...]"""

        content = self._call_api(prompt)
        relations = self._parse_string_array(content)

        return [
            AtomicProposition(text=r, topic="relations", is_predicate=False)
            for r in relations
        ]

    def _generate_categories(self) -> List[AtomicProposition]:
        """Generate categories for universal statements (e.g., 'mammals', 'vehicles')."""
        prompt = f"""Generate a list of category/type nouns for universal statements.
These will be used in statements like "All [category] are..." or "X is a [category]".

Requirements:
- Include natural kinds (animals, plants), artifacts (vehicles, tools), abstract categories
- Each should be a plural noun or noun phrase
- Generate at least {self.config.categories_count} categories

Respond with a JSON array of strings only:
["mammals", "vehicles", "students", ...]"""

        content = self._call_api(prompt)
        categories = self._parse_string_array(content)

        return [
            AtomicProposition(text=c, topic="categories", is_predicate=False)
            for c in categories
        ]

    def _generate_entities(self) -> List[str]:
        """Generate entity names for first-order logic."""
        prompt = f"""Generate a list of specific entity names for logic examples.
These will be used as specific instances in statements.

Requirements:
- Include person names (diverse), place names, object descriptions
- Mix of proper nouns and definite descriptions ("the red car", "John", "Mount Everest")
- Generate at least {self.config.entities_count} entity names

Respond with a JSON array of strings only:
["John", "the red car", "Paris", ...]"""

        content = self._call_api(prompt)
        return self._parse_string_array(content)

    def _call_api(self, prompt: str) -> str:
        """Make API call with retries."""
        for attempt in range(self.config.retry_attempts):
            try:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            except Exception as e:
                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    raise e
        return "[]"

    def _parse_string_array(self, content: str) -> List[str]:
        """Parse JSON array of strings from API response."""
        try:
            content = content.strip()
            # Find JSON array in content
            start_idx = content.find("[")
            end_idx = content.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                content = content[start_idx:end_idx]
            return json.loads(content)
        except json.JSONDecodeError:
            return []

    def get_propositions_for_template(
        self,
        template_name: str,
        premise_slots: List[str],
        entity_slots: List[str],
        logic_type: str
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        Get propositions for a specific template by sampling from the pool.

        Returns:
            (propositions_dict, entities_dict)
        """
        if self.pool is None:
            raise ValueError("Pool not generated. Call generate_pool() first.")

        propositions = {}
        entities = {}

        if logic_type == "propositional":
            # For propositional logic, sample full propositions
            sampled = self.pool.sample_propositions(len(premise_slots))
            for slot, prop in zip(premise_slots, sampled):
                propositions[slot] = prop.text
        else:
            # For first-order logic, handle based on template type
            if template_name == "universal_instantiation":
                # Need: X (category), P (predicate), a (entity)
                cat = self.pool.sample_categories(1)[0] if self.pool.categories else None
                pred = self.pool.sample_predicates(1)[0] if self.pool.predicates else None
                propositions["X"] = cat.text if cat else "things"
                propositions["P"] = pred.text if pred else "special"
            elif template_name == "relational_inference":
                # Need: R (relation), S (predicate)
                rel = self.pool.sample_relations(1)[0] if self.pool.relations else None
                pred = self.pool.sample_predicates(1)[0] if self.pool.predicates else None
                propositions["R"] = rel.text if rel else "knows"
                propositions["S"] = pred.text if pred else "wise"
            else:
                # Default: sample predicates for P, Q, R, etc.
                sampled = self.pool.sample_predicates(len(premise_slots))
                for slot, prop in zip(premise_slots, sampled):
                    propositions[slot] = prop.text

            # Sample entities
            if entity_slots:
                entity_names = self.pool.entities.sample(len(entity_slots))
                for slot, name in zip(entity_slots, entity_names):
                    entities[slot] = name

        return propositions, entities


# Fallback pool for when API is not available
def create_fallback_pool() -> PropositionPool:
    """Create a basic fallback pool without API calls."""
    pool = PropositionPool()

    # Basic propositions
    pool.propositions = [
        AtomicProposition("it is raining", "weather", False),
        AtomicProposition("the ground is wet", "weather", False),
        AtomicProposition("the sun is shining", "weather", False),
        AtomicProposition("it is cold outside", "weather", False),
        AtomicProposition("the alarm is ringing", "everyday", False),
        AtomicProposition("I wake up early", "everyday", False),
        AtomicProposition("the coffee is hot", "everyday", False),
        AtomicProposition("the door is open", "everyday", False),
        AtomicProposition("the light is on", "everyday", False),
        AtomicProposition("the car is running", "transport", False),
        AtomicProposition("the road is clear", "transport", False),
        AtomicProposition("traffic is heavy", "transport", False),
        AtomicProposition("the store is open", "everyday", False),
        AtomicProposition("the phone is charging", "tech", False),
        AtomicProposition("the battery is full", "tech", False),
        AtomicProposition("the wifi is connected", "tech", False),
        AtomicProposition("the water is boiling", "science", False),
        AtomicProposition("the ice is melting", "science", False),
        AtomicProposition("the plant is growing", "nature", False),
        AtomicProposition("the bird is singing", "nature", False),
    ]

    # Basic predicates
    pool.predicates = [
        AtomicProposition("tall", "physical", True),
        AtomicProposition("short", "physical", True),
        AtomicProposition("heavy", "physical", True),
        AtomicProposition("light", "physical", True),
        AtomicProposition("red", "color", True),
        AtomicProposition("blue", "color", True),
        AtomicProposition("made of metal", "material", True),
        AtomicProposition("made of wood", "material", True),
        AtomicProposition("edible", "property", True),
        AtomicProposition("dangerous", "property", True),
        AtomicProposition("valuable", "property", True),
        AtomicProposition("fragile", "property", True),
        AtomicProposition("warm-blooded", "biology", True),
        AtomicProposition("able to fly", "ability", True),
        AtomicProposition("able to swim", "ability", True),
    ]

    # Basic relations
    pool.relations = [
        AtomicProposition("loves", "social", False),
        AtomicProposition("knows", "social", False),
        AtomicProposition("teaches", "social", False),
        AtomicProposition("is taller than", "physical", False),
        AtomicProposition("is older than", "temporal", False),
        AtomicProposition("works with", "social", False),
        AtomicProposition("lives near", "spatial", False),
        AtomicProposition("owns", "possession", False),
    ]

    # Basic categories
    pool.categories = [
        AtomicProposition("mammals", "biology", False),
        AtomicProposition("birds", "biology", False),
        AtomicProposition("vehicles", "artifacts", False),
        AtomicProposition("fruits", "food", False),
        AtomicProposition("students", "social", False),
        AtomicProposition("teachers", "social", False),
        AtomicProposition("buildings", "artifacts", False),
        AtomicProposition("metals", "material", False),
    ]

    # Basic entities
    pool.entities = EntityPool(names=[
        "John", "Mary", "Alice", "Bob", "Charlie",
        "the red car", "the old house", "the big dog",
        "Paris", "London", "Tokyo", "New York",
        "Mount Everest", "the Pacific Ocean",
        "the morning sun", "the evening star",
    ])

    return pool
