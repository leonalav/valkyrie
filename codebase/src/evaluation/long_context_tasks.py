"""
Long-context task evaluation for HRM model.

Implements evaluation for long-context reasoning tasks including document retrieval,
cross-reference finding, and multi-document reasoning as specified in the PLAN.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import random

logger = logging.getLogger(__name__)


@dataclass
class LongContextTask:
    """Single long-context task instance."""
    name: str
    input_sequence: jnp.ndarray
    target_output: jnp.ndarray
    context_length: int
    query_position: int  # Position of query in context
    answer_position: int  # Position of answer in context
    max_cycles: int = 8
    max_steps: int = 10
    difficulty: str = "medium"


class LongContextEvaluator:
    """
    Evaluates HRM model on long-context reasoning tasks.
    
    Tasks include:
    - Document retrieval (find specific information)
    - Cross-reference finding (link related information)
    - Multi-document reasoning
    - Long-range dependency tracking
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        max_sequence_length: int = 32768,  # Support long contexts
        seed: int = 42,
    ):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.rng = jax.random.PRNGKey(seed)
        
        # Special tokens for long-context tasks
        self.special_tokens = {
            'DOC_START': vocab_size - 15,
            'DOC_END': vocab_size - 14,
            'QUERY_START': vocab_size - 13,
            'QUERY_END': vocab_size - 12,
            'FIND': vocab_size - 11,
            'RETRIEVE': vocab_size - 10,
            'REFERENCE': vocab_size - 9,
            'ANSWER': vocab_size - 8,
            'SEP': vocab_size - 7,
            'PARAGRAPH': vocab_size - 6,
            'SECTION': vocab_size - 5,
            'CHAPTER': vocab_size - 4,
        }
    
    def generate_synthetic_document(
        self,
        length: int,
        num_facts: int = 10,
        difficulty: str = "medium"
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Generate a synthetic document with embedded facts."""
        
        # Create vocabulary for synthetic text
        words = [
            "the", "and", "of", "to", "a", "in", "is", "it", "you", "that",
            "he", "was", "for", "on", "are", "as", "with", "his", "they", "i",
            "at", "be", "this", "have", "from", "or", "one", "had", "by", "word",
            "but", "not", "what", "all", "were", "we", "when", "your", "can", "said",
            "company", "product", "service", "customer", "business", "market", "data",
            "system", "technology", "information", "process", "management", "development",
            "research", "analysis", "report", "study", "project", "team", "work",
            "year", "time", "day", "week", "month", "number", "value", "price",
            "cost", "revenue", "profit", "growth", "increase", "decrease", "change"
        ]
        
        # Generate base document
        document_tokens = []
        facts = {}
        
        # Add document structure
        document_tokens.append(self.special_tokens['DOC_START'])
        
        # Generate sections with facts
        current_pos = 1
        for section_idx in range(max(1, num_facts // 3)):
            document_tokens.append(self.special_tokens['SECTION'])
            current_pos += 1
            
            # Add section content
            section_length = length // max(1, num_facts // 3)
            
            # Embed facts in this section
            facts_in_section = min(3, num_facts - len(facts))
            fact_positions = sorted(random.sample(range(section_length), facts_in_section))
            
            for i in range(section_length):
                if i in fact_positions:
                    # Insert a fact
                    fact_id = len(facts)
                    if difficulty == "easy":
                        fact = f"fact_{fact_id}_value_{fact_id * 10}"
                    elif difficulty == "medium":
                        fact = f"entity_{fact_id}_has_property_{fact_id * 7 + 3}"
                    else:  # hard
                        fact = f"complex_relation_{fact_id}_between_entity_{fact_id}_and_entity_{(fact_id + 1) % 5}_with_value_{fact_id * 13 + 7}"
                    
                    fact_tokens = [min(ord(c), self.vocab_size - 20) for c in fact]
                    document_tokens.extend(fact_tokens)
                    
                    facts[f"fact_{fact_id}"] = {
                        'content': fact,
                        'position': current_pos,
                        'tokens': fact_tokens
                    }
                    current_pos += len(fact_tokens)
                else:
                    # Add regular content
                    word = random.choice(words)
                    word_tokens = [min(ord(c), self.vocab_size - 20) for c in word]
                    document_tokens.extend(word_tokens)
                    document_tokens.append(32)  # space
                    current_pos += len(word_tokens) + 1
        
        document_tokens.append(self.special_tokens['DOC_END'])
        
        return document_tokens, facts
    
    def generate_retrieval_task(
        self,
        context_length: int = 4096,
        difficulty: str = "medium"
    ) -> LongContextTask:
        """Generate a document retrieval task."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        # Generate document with facts
        num_facts = 5 if difficulty == "easy" else 10 if difficulty == "medium" else 20
        document_tokens, facts = self.generate_synthetic_document(
            context_length - 200,  # Leave room for query
            num_facts,
            difficulty
        )
        
        # Select a fact to query
        fact_keys = list(facts.keys())
        if not fact_keys:
            # Fallback if no facts generated
            target_fact = "default_fact"
            target_tokens = [min(ord(c), self.vocab_size - 20) for c in target_fact]
        else:
            fact_key = random.choice(fact_keys)
            target_fact = facts[fact_key]['content']
            target_tokens = facts[fact_key]['tokens']
        
        # Create query
        if difficulty == "easy":
            query = f"find fact"
        elif difficulty == "medium":
            query = f"retrieve information about entity"
        else:  # hard
            query = f"locate complex relation involving multiple entities"
        
        query_tokens = [min(ord(c), self.vocab_size - 20) for c in query]
        
        # Construct input sequence
        input_seq = document_tokens + [
            self.special_tokens['QUERY_START']
        ] + query_tokens + [
            self.special_tokens['QUERY_END'],
            self.special_tokens['RETRIEVE'],
            self.special_tokens['SEP']
        ]
        
        # Truncate if too long
        if len(input_seq) > self.max_sequence_length - 100:
            input_seq = input_seq[:self.max_sequence_length - 100]
        
        input_seq = jnp.array(input_seq)
        target_output = jnp.array(target_tokens)
        
        return LongContextTask(
            name=f"retrieval_{difficulty}_{context_length}",
            input_sequence=input_seq,
            target_output=target_output,
            context_length=len(input_seq),
            query_position=len(document_tokens),
            answer_position=facts.get(fact_keys[0] if fact_keys else 'default', {}).get('position', 0),
            max_cycles=6 if difficulty == "easy" else 8,
            max_steps=8 if difficulty == "easy" else 12,
            difficulty=difficulty
        )
    
    def generate_cross_reference_task(
        self,
        context_length: int = 8192,
        difficulty: str = "medium"
    ) -> LongContextTask:
        """Generate a cross-reference finding task."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        # Generate document with related facts
        num_entities = 3 if difficulty == "easy" else 5 if difficulty == "medium" else 8
        
        # Create entities and relationships
        entities = [f"entity_{i}" for i in range(num_entities)]
        relationships = []
        
        # Create cross-references
        for i in range(num_entities - 1):
            for j in range(i + 1, min(i + 3, num_entities)):  # Limit connections
                rel_type = random.choice(["related_to", "connected_with", "references"])
                relationships.append((entities[i], rel_type, entities[j]))
        
        # Generate document with embedded relationships
        document_tokens = [self.special_tokens['DOC_START']]
        relationship_positions = {}
        
        # Distribute relationships throughout document
        section_length = (context_length - 300) // len(relationships)
        
        for idx, (ent1, rel, ent2) in enumerate(relationships):
            # Add section marker
            document_tokens.append(self.special_tokens['SECTION'])
            
            # Add some filler content
            filler_length = section_length // 2
            for _ in range(filler_length):
                document_tokens.append(random.randint(100, self.vocab_size - 50))
            
            # Add relationship
            rel_text = f"{ent1} {rel} {ent2}"
            rel_tokens = [min(ord(c), self.vocab_size - 20) for c in rel_text]
            
            relationship_positions[f"rel_{idx}"] = {
                'entities': (ent1, ent2),
                'relation': rel,
                'position': len(document_tokens),
                'tokens': rel_tokens
            }
            
            document_tokens.extend(rel_tokens)
            
            # Add more filler
            for _ in range(filler_length):
                document_tokens.append(random.randint(100, self.vocab_size - 50))
        
        document_tokens.append(self.special_tokens['DOC_END'])
        
        # Create cross-reference query
        if relationships:
            target_rel = random.choice(list(relationship_positions.keys()))
            target_info = relationship_positions[target_rel]
            query_entity = target_info['entities'][0]
            
            query = f"find references to {query_entity}"
            query_tokens = [min(ord(c), self.vocab_size - 20) for c in query]
            
            target_tokens = target_info['tokens']
        else:
            query = "find references"
            query_tokens = [min(ord(c), self.vocab_size - 20) for c in query]
            target_tokens = [self.vocab_size - 30]  # Fallback token
        
        # Construct input sequence
        input_seq = document_tokens + [
            self.special_tokens['QUERY_START']
        ] + query_tokens + [
            self.special_tokens['QUERY_END'],
            self.special_tokens['REFERENCE'],
            self.special_tokens['SEP']
        ]
        
        # Truncate if too long
        if len(input_seq) > self.max_sequence_length - 100:
            input_seq = input_seq[:self.max_sequence_length - 100]
        
        input_seq = jnp.array(input_seq)
        target_output = jnp.array(target_tokens)
        
        return LongContextTask(
            name=f"cross_ref_{difficulty}_{context_length}",
            input_sequence=input_seq,
            target_output=target_output,
            context_length=len(input_seq),
            query_position=len(document_tokens),
            answer_position=target_info.get('position', 0) if relationships else 0,
            max_cycles=8 if difficulty == "easy" else 10,
            max_steps=10 if difficulty == "easy" else 15,
            difficulty=difficulty
        )
    
    def generate_multi_document_task(
        self,
        context_length: int = 16384,
        num_documents: int = 3,
        difficulty: str = "medium"
    ) -> LongContextTask:
        """Generate a multi-document reasoning task."""
        
        self.rng, task_rng = jax.random.split(self.rng)
        
        # Generate multiple related documents
        all_tokens = []
        shared_entities = [f"shared_entity_{i}" for i in range(3)]
        document_facts = {}
        
        doc_length = (context_length - 200) // num_documents
        
        for doc_idx in range(num_documents):
            # Add document separator
            all_tokens.append(self.special_tokens['DOC_START'])
            
            # Generate document with shared entities
            doc_tokens, facts = self.generate_synthetic_document(
                doc_length,
                num_facts=3,
                difficulty=difficulty
            )
            
            # Inject shared entities
            entity = shared_entities[doc_idx % len(shared_entities)]
            entity_tokens = [min(ord(c), self.vocab_size - 20) for c in entity]
            
            # Insert entity at random position
            insert_pos = len(doc_tokens) // 2
            doc_tokens = doc_tokens[:insert_pos] + entity_tokens + doc_tokens[insert_pos:]
            
            document_facts[f"doc_{doc_idx}"] = {
                'entity': entity,
                'position': len(all_tokens) + insert_pos,
                'facts': facts
            }
            
            all_tokens.extend(doc_tokens)
            all_tokens.append(self.special_tokens['DOC_END'])
        
        # Create multi-document query
        target_entity = random.choice(shared_entities)
        query = f"find all mentions of {target_entity} across documents"
        query_tokens = [min(ord(c), self.vocab_size - 20) for c in query]
        
        # Target: entity tokens (simplified)
        target_tokens = [min(ord(c), self.vocab_size - 20) for c in target_entity]
        
        # Construct input sequence
        input_seq = all_tokens + [
            self.special_tokens['QUERY_START']
        ] + query_tokens + [
            self.special_tokens['QUERY_END'],
            self.special_tokens['FIND'],
            self.special_tokens['SEP']
        ]
        
        # Truncate if too long
        if len(input_seq) > self.max_sequence_length - 100:
            input_seq = input_seq[:self.max_sequence_length - 100]
        
        input_seq = jnp.array(input_seq)
        target_output = jnp.array(target_tokens)
        
        return LongContextTask(
            name=f"multi_doc_{difficulty}_{num_documents}_{context_length}",
            input_sequence=input_seq,
            target_output=target_output,
            context_length=len(input_seq),
            query_position=len(all_tokens),
            answer_position=list(document_facts.values())[0].get('position', 0),
            max_cycles=10 if difficulty == "easy" else 12,
            max_steps=12 if difficulty == "easy" else 18,
            difficulty=difficulty
        )
    
    def evaluate_model(
        self,
        model_fn,
        params: Dict[str, Any],
        tasks: List[LongContextTask],
        batch_size: int = 2,  # Smaller batches for long contexts
    ) -> Dict[str, float]:
        """
        Evaluate model on long-context tasks.
        
        Args:
            model_fn: Model forward function
            params: Model parameters
            tasks: List of long-context tasks
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        
        results = {
            "accuracy": 0.0,
            "exact_match": 0.0,
            "retrieval_success": 0.0,
            "avg_cycles_used": 0.0,
            "avg_steps_used": 0.0,
            "context_utilization": 0.0,
            "long_range_accuracy": 0.0,
        }
        
        total_tasks = len(tasks)
        if total_tasks == 0:
            return results
        
        correct_predictions = 0
        exact_matches = 0
        retrieval_successes = 0
        total_cycles = 0
        total_steps = 0
        long_range_correct = 0
        
        # Process tasks in batches
        for i in range(0, total_tasks, batch_size):
            batch_tasks = tasks[i:i + batch_size]
            
            # Prepare batch inputs
            batch_inputs = []
            
            for task in batch_tasks:
                # Pad sequences to max length
                padded_input = jnp.zeros(self.max_sequence_length, dtype=jnp.int32)
                input_len = min(len(task.input_sequence), self.max_sequence_length)
                padded_input = padded_input.at[:input_len].set(task.input_sequence[:input_len])
                batch_inputs.append(padded_input)
            
            batch_inputs = jnp.stack(batch_inputs)
            
            # Run model inference
            try:
                outputs = model_fn(params, batch_inputs)
                
                # Extract predictions and HRM metrics
                if isinstance(outputs, dict):
                    predictions = outputs.get('logits', outputs.get('predictions'))
                    cycles_used = outputs.get('cycles_used', 0)
                    steps_used = outputs.get('steps_used', 0)
                    attention_weights = outputs.get('attention_weights', None)
                else:
                    predictions = outputs
                    cycles_used = 0
                    steps_used = 0
                    attention_weights = None
                
                # Evaluate each task in batch
                for j, task in enumerate(batch_tasks):
                    pred = predictions[j] if predictions.ndim > 1 else predictions
                    target = task.target_output
                    
                    # Convert logits to predictions if needed
                    if pred.ndim > 1:  # Logits
                        pred = jnp.argmax(pred, axis=-1)
                    
                    # Truncate prediction to target length
                    pred = pred[:len(target)]
                    
                    # Check accuracy
                    if len(pred) == len(target):
                        accuracy = jnp.mean(pred == target)
                        correct_predictions += accuracy
                        
                        # Exact match
                        if jnp.all(pred == target):
                            exact_matches += 1
                            retrieval_successes += 1
                    
                    # Check long-range accuracy (query far from answer)
                    query_answer_distance = abs(task.query_position - task.answer_position)
                    if query_answer_distance > task.context_length * 0.5:  # Far apart
                        if len(pred) == len(target) and jnp.all(pred == target):
                            long_range_correct += 1
                    
                    # Track HRM efficiency
                    total_cycles += cycles_used
                    total_steps += steps_used
                    
            except Exception as e:
                logger.warning(f"Error evaluating batch {i}: {e}")
                continue
        
        # Compute final metrics
        if total_tasks > 0:
            results["accuracy"] = correct_predictions / total_tasks
            results["exact_match"] = exact_matches / total_tasks
            results["retrieval_success"] = retrieval_successes / total_tasks
            results["avg_cycles_used"] = total_cycles / total_tasks
            results["avg_steps_used"] = total_steps / total_tasks
            results["long_range_accuracy"] = long_range_correct / total_tasks
            
            # Context utilization (simplified metric)
            avg_context_length = sum(task.context_length for task in tasks) / total_tasks
            results["context_utilization"] = min(avg_context_length / self.max_sequence_length, 1.0)
        
        return results
    
    def create_evaluation_suite(
        self,
        num_tasks_per_type: int = 3,  # Fewer tasks due to computational cost
        difficulties: List[str] = ["easy", "medium", "hard"],
        context_lengths: List[int] = [4096, 8192, 16384]
    ) -> List[LongContextTask]:
        """Create a comprehensive long-context evaluation suite."""
        
        tasks = []
        
        for difficulty in difficulties:
            for context_length in context_lengths:
                # Skip very long contexts for easy tasks
                if difficulty == "easy" and context_length > 8192:
                    continue
                
                # Retrieval tasks
                for _ in range(num_tasks_per_type):
                    tasks.append(self.generate_retrieval_task(context_length, difficulty))
                
                # Cross-reference tasks
                for _ in range(num_tasks_per_type):
                    tasks.append(self.generate_cross_reference_task(context_length, difficulty))
                
                # Multi-document tasks (only for longer contexts)
                if context_length >= 8192:
                    for _ in range(num_tasks_per_type):
                        num_docs = 2 if difficulty == "easy" else 3 if difficulty == "medium" else 4
                        tasks.append(self.generate_multi_document_task(context_length, num_docs, difficulty))
        
        logger.info(f"Created long-context evaluation suite with {len(tasks)} tasks")
        return tasks