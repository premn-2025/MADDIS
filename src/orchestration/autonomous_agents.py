#!/usr/bin/env python3
"""
Autonomous Agent Base Class for True Multi-Agent Architecture
This transforms our system from centralized orchestration to peer-to-peer agent communication
"""

import asyncio
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Try to import redis, but make it optional
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Try to import jsonlogger, but make it optional
try:
    from pythonjsonlogger import jsonlogger
    JSONLOGGER_AVAILABLE = True
except ImportError:
    JSONLOGGER_AVAILABLE = False


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NEGOTIATION = "negotiation"
    CRITIQUE = "critique"
    VOTE = "vote"
    ERROR = "error"


@dataclass
class AgentMessage:
    """Standard message format for inter-agent communication"""
    id: str
    sender_id: str
    receiver_id: Optional[str]  # None for broadcasts
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: str
    requires_response: bool = False
    conversation_id: Optional[str] = None


@dataclass
class AgentCapability:
    """Describes what an agent can do"""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    confidence: float
    cost: float  # Computational cost (0.0-1.0)


class AgentLogger:
    """Structured logging for agent decisions and interactions"""

    def __init__(self, agent_id: str):
        self.logger = logging.getLogger(f"agent.{agent_id}")
        self.logger.setLevel(logging.INFO)

        # JSON formatter for structured logs
        import os
        os.makedirs('logs/agents', exist_ok=True)
        
        handler = logging.FileHandler(f'logs/agents/{agent_id}.jsonl')
        if JSONLOGGER_AVAILABLE:
            formatter = jsonlogger.JsonFormatter()
        else:
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_decision(self, decision: str, reasoning: str, confidence: float, context: Dict = None):
        """Log an autonomous decision made by the agent"""
        self.logger.info({
            'event': 'decision',
            'decision': decision,
            'reasoning': reasoning,
            'confidence': confidence,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })

    def log_interaction(self, message: AgentMessage, direction: str):
        """Log inter-agent communication"""
        self.logger.info({
            'event': 'interaction',
            'direction': direction,  # 'sent' or 'received'
            'message_id': message.id,
            'from': message.sender_id,
            'to': message.receiver_id,
            'type': message.message_type.value,
            'timestamp': message.timestamp
        })

    def log_error(self, error: Exception, context: Dict = None):
        """Log errors with context"""
        self.logger.error({
            'event': 'error',
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        })


class MessageBus:
    """Redis-based message broker for agent communication"""

    def __init__(self, redis_host='localhost', redis_port=6379):
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host, port=redis_port, decode_responses=True)
                self.redis_client.ping()
            except Exception:
                self.redis_client = None
        else:
            self.redis_client = None
        self.subscribers = {}
        self.local_messages = {}  # Fallback for when Redis is not available

    async def publish(self, message: AgentMessage):
        """Publish message to appropriate channel"""
        channel = f"agent.{message.receiver_id}" if message.receiver_id else "agent.broadcast"
        message_dict = asdict(message)
        message_dict['message_type'] = message.message_type.value
        message_json = json.dumps(message_dict)
        
        if self.redis_client:
            self.redis_client.publish(channel, message_json)
        else:
            # Local fallback
            if channel not in self.local_messages:
                self.local_messages[channel] = []
            self.local_messages[channel].append(message_json)

    async def subscribe(self, agent_id: str, callback: Callable):
        """Subscribe to messages for specific agent"""
        channel = f"agent.{agent_id}"
        
        if self.redis_client:
            pubsub = self.redis_client.pubsub()
            pubsub.subscribe(channel, "agent.broadcast")
            self.subscribers[agent_id] = pubsub

            # Listen for messages
            async def message_listener():
                for message in pubsub.listen():
                    if message['type'] == 'message':
                        data = json.loads(message['data'])
                        data['message_type'] = MessageType(data['message_type'])
                        agent_message = AgentMessage(**data)
                        await callback(agent_message)

            asyncio.create_task(message_listener())
        else:
            self.subscribers[agent_id] = callback


class AutonomousAgent(ABC):
    """Base class for truly autonomous agents with decision-making capabilities"""

    def __init__(self, agent_id: str, message_bus: MessageBus = None):
        self.agent_id = agent_id
        self.message_bus = message_bus or MessageBus()
        self.logger = AgentLogger(agent_id)
        self.state = {}
        self.capabilities = []
        self.reputation = 0.5  # Track performance over time
        self.workload = 0.0  # Current computational load

        # Decision-making parameters
        self.confidence_threshold = 0.7
        self.max_retries = 3
        self.collaboration_preference = 0.8  # How much to collaborate vs work alone

    async def _setup_message_handling(self):
        """Set up message subscription"""
        await self.message_bus.subscribe(self.agent_id, self._handle_incoming_message)

    async def _handle_incoming_message(self, message: AgentMessage):
        """Handle incoming messages from other agents"""
        self.logger.log_interaction(message, 'received')

        try:
            if message.message_type == MessageType.REQUEST:
                response = await self._handle_request(message)
                if response and message.requires_response:
                    await self.send_response(message.sender_id, response, message.id)

            elif message.message_type == MessageType.RESPONSE:
                await self._handle_response(message)

            elif message.message_type == MessageType.BROADCAST:
                await self._handle_broadcast(message)

            elif message.message_type == MessageType.NEGOTIATION:
                await self._handle_negotiation(message)

        except Exception as e:
            self.logger.log_error(e, {'message_id': message.id})
            await self.send_error(message.sender_id, str(e), message.id)

    @abstractmethod
    async def _handle_request(self, message: AgentMessage) -> Optional[Dict]:
        """Handle task requests - must be implemented by subclasses"""
        pass

    async def _handle_response(self, message: AgentMessage):
        """Handle responses from other agents"""
        # Store response for requester to process
        self.state[f"response_{message.conversation_id}"] = message.content

    async def _handle_broadcast(self, message: AgentMessage):
        """Handle broadcast messages"""
        # Default: ignore broadcasts unless specifically interested
        pass

    async def _handle_negotiation(self, message: AgentMessage):
        """Handle negotiation messages"""
        # Default negotiation strategy
        proposal = await self.make_proposal(message.content.get('topic'))
        await self.send_negotiation(message.sender_id, proposal)

    async def send_request(self, receiver_id: str, content: Dict, requires_response: bool = True) -> str:
        """Send request to another agent"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=MessageType.REQUEST,
            content=content,
            timestamp=datetime.now().isoformat(),
            requires_response=requires_response,
            conversation_id=str(uuid.uuid4())
        )

        await self.message_bus.publish(message)
        self.logger.log_interaction(message, 'sent')
        return message.conversation_id

    async def send_response(self, receiver_id: str, content: Dict, conversation_id: str):
        """Send response to a request"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=MessageType.RESPONSE,
            content=content,
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id
        )

        await self.message_bus.publish(message)
        self.logger.log_interaction(message, 'sent')

    async def broadcast(self, content: Dict):
        """Broadcast message to all agents"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=None,
            message_type=MessageType.BROADCAST,
            content=content,
            timestamp=datetime.now().isoformat()
        )

        await self.message_bus.publish(message)
        self.logger.log_interaction(message, 'sent')

    async def send_negotiation(self, receiver_id: str, proposal: Dict):
        """Send negotiation proposal"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=MessageType.NEGOTIATION,
            content={'proposal': proposal},
            timestamp=datetime.now().isoformat()
        )

        await self.message_bus.publish(message)

    async def send_error(self, receiver_id: str, error_message: str, conversation_id: str):
        """Send error message"""
        message = AgentMessage(
            id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=MessageType.ERROR,
            content={'error': error_message},
            timestamp=datetime.now().isoformat(),
            conversation_id=conversation_id
        )

        await self.message_bus.publish(message)

    def decide_next_action(self, context: Dict = None) -> str:
        """Autonomous decision-making about what to do next"""
        # Base decision logic - override in subclasses
        if self.workload < 0.3:
            action = "proactive_work"
            reasoning = f"Low workload ({self.workload:.2f}), seeking tasks"
        elif self.workload > 0.8:
            action = "defer_tasks"
            reasoning = f"High workload ({self.workload:.2f}), deferring new tasks"
        else:
            action = "continue_current"
            reasoning = f"Normal workload ({self.workload:.2f}), continuing"

        self.logger.log_decision(action, reasoning, self.reputation, context)
        return action

    def can_handle(self, task: Dict) -> bool:
        """Decide if agent can handle a specific task"""
        required_capability = task.get('requires_capability')
        if not required_capability:
            return True

        for capability in self.capabilities:
            if capability.name == required_capability:
                return capability.confidence > self.confidence_threshold

        return False

    async def delegate_or_refuse(self, task: Dict) -> Dict:
        """Either delegate task to better agent or refuse with reasoning"""
        # Find agents with required capability
        suitable_agents = await self._find_suitable_agents(task)

        if suitable_agents:
            # Delegate to best agent
            best_agent = max(suitable_agents, key=lambda x: x['confidence'])
            conversation_id = await self.send_request(
                best_agent['agent_id'],
                {'delegated_task': task},
                requires_response=True
            )

            reasoning = f"Delegated to {best_agent['agent_id']} (confidence: {best_agent['confidence']:.2f})"
            self.logger.log_decision("delegate", reasoning, best_agent['confidence'])

            return {'status': 'delegated', 'to': best_agent['agent_id'], 'conversation_id': conversation_id}
        else:
            # Refuse with clear reasoning
            reasoning = f"Cannot handle task requiring '{task.get('requires_capability')}'"
            self.logger.log_decision("refuse", reasoning, 0.0)

            return {'status': 'refused', 'reason': reasoning}

    async def _find_suitable_agents(self, task: Dict) -> List[Dict]:
        """Find other agents capable of handling task"""
        # Broadcast capability inquiry
        await self.broadcast({
            'type': 'capability_inquiry',
            'task': task,
            'requester': self.agent_id
        })

        # Wait for responses (simplified - in reality would use more sophisticated discovery)
        await asyncio.sleep(1)

        # Return mock data for now - would be replaced with actual agent registry
        return []

    async def make_proposal(self, topic: Dict) -> Dict:
        """Make a proposal during negotiations"""
        # Default proposal strategy - override in subclasses
        return {
            'agent_id': self.agent_id,
            'proposal': f"Default proposal for {topic}",
            'confidence': self.reputation
        }

    def update_reputation(self, feedback: float):
        """Update agent reputation based on performance feedback"""
        # Exponential moving average
        alpha = 0.1
        self.reputation = alpha * feedback + (1 - alpha) * self.reputation
        self.logger.log_decision("reputation_update", f"Updated to {self.reputation:.3f}", feedback)


class MultiAgentNegotiation:
    """Handles multi-round negotiations between agents"""

    def __init__(self, agents: List[AutonomousAgent], topic: Dict, max_rounds: int = 5):
        self.agents = agents
        self.topic = topic
        self.max_rounds = max_rounds
        self.proposals = {}
        self.critiques = {}

    async def negotiate(self) -> Dict:
        """Run multi-round negotiation process"""
        for round_num in range(self.max_rounds):
            print(f"🗣 Negotiation Round {round_num + 1}")

            # Each agent makes proposal
            for agent in self.agents:
                proposal = await agent.make_proposal(self.topic)
                self.proposals[agent.agent_id] = proposal

            # Exchange critiques
            await self._exchange_critiques()

            # Check for convergence
            if await self._has_converged():
                break

        # Final consensus
        return await self._reach_consensus()

    async def _exchange_critiques(self):
        """Agents critique each other's proposals"""
        for agent in self.agents:
            for other_agent_id, proposal in self.proposals.items():
                if other_agent_id != agent.agent_id:
                    if hasattr(agent, 'critique_proposal'):
                        critique = await agent.critique_proposal(proposal)
                        self.critiques[f"{agent.agent_id}_critiques_{other_agent_id}"] = critique

    async def _has_converged(self) -> bool:
        """Check if agents have reached sufficient agreement"""
        if len(self.proposals) < 2:
            return False

        # Simple convergence check - proposals are similar
        proposal_values = [p.get('confidence', 0) for p in self.proposals.values()]
        return max(proposal_values) - min(proposal_values) < 0.1

    async def _reach_consensus(self) -> Dict:
        """Reach final consensus through voting"""
        votes = {}

        for agent in self.agents:
            if hasattr(agent, 'vote_on_proposals'):
                vote = await agent.vote_on_proposals(self.proposals)
                votes[agent.agent_id] = vote

        # Weight votes by agent reputation
        weighted_scores = {}
        for agent_id, vote in votes.items():
            agent = next(a for a in self.agents if a.agent_id == agent_id)
            for proposal_id, score in vote.items():
                if proposal_id not in weighted_scores:
                    weighted_scores[proposal_id] = 0
                weighted_scores[proposal_id] += score * agent.reputation

        if not weighted_scores:
            return {'winner': None, 'proposal': None, 'final_score': 0, 'all_scores': {}}

        # Return winning proposal
        winner_id = max(weighted_scores, key=weighted_scores.get)
        return {
            'winner': winner_id,
            'proposal': self.proposals.get(winner_id),
            'final_score': weighted_scores[winner_id],
            'all_scores': weighted_scores
        }


# Example implementation of a specialized agent
class PropertyPredictorAgent(AutonomousAgent):
    """Autonomous agent for molecular property prediction"""

    def __init__(self, message_bus: MessageBus = None):
        super().__init__("property_predictor", message_bus)

        # Define capabilities
        self.capabilities = [
            AgentCapability(
                name="molecular_property_prediction",
                description="Predict molecular properties from SMILES",
                input_types=["smiles"],
                output_types=["properties", "confidence"],
                confidence=0.85,
                cost=0.3
            ),
            AgentCapability(
                name="drug_likeness_assessment",
                description="Assess drug-likeness using Lipinski rules",
                input_types=["smiles"],
                output_types=["drug_like", "violations"],
                confidence=0.95,
                cost=0.1
            )
        ]

        # Load actual models (simplified for now)
        self.models = self._load_models()

    def _load_models(self):
        """Load trained ML models"""
        return {
            'property_predictor': None,  # Would load actual model
            'drug_likeness': None
        }

    async def _handle_request(self, message: AgentMessage) -> Optional[Dict]:
        """Handle incoming task requests"""
        content = message.content
        task_type = content.get('task_type')

        if task_type == 'predict_properties':
            return await self._predict_properties(content.get('smiles'))
        elif task_type == 'assess_drug_likeness':
            return await self._assess_drug_likeness(content.get('smiles'))
        else:
            return {'error': f'Unknown task type: {task_type}'}

    async def _predict_properties(self, smiles: str) -> Dict:
        """Predict molecular properties"""
        # Simulate prediction (replace with actual model)
        import random

        self.workload += 0.2
        # Simulate computation
        await asyncio.sleep(0.5)

        properties = {
            'molecular_weight': random.uniform(200, 500),
            'logp': random.uniform(-2, 5),
            'tpsa': random.uniform(20, 140),
            'hbd': random.randint(0, 5),
            'hba': random.randint(0, 10)
        }

        confidence = random.uniform(0.7, 0.95)

        self.workload -= 0.2

        self.logger.log_decision(
            "property_prediction",
            f"Predicted properties for {smiles[:10]}...",
            confidence,
            {'smiles': smiles}
        )

        return {
            'properties': properties,
            'confidence': confidence,
            'agent_id': self.agent_id
        }

    async def _assess_drug_likeness(self, smiles: str) -> Dict:
        """Assess drug-likeness"""
        import random
        # Mock assessment
        violations = random.randint(0, 2)
        drug_like = violations <= 1

        return {
            'drug_like': drug_like,
            'violations': violations,
            'confidence': 0.95,
            'agent_id': self.agent_id
        }

    def decide_next_action(self, context: Dict = None) -> str:
        """Autonomous decision about next action"""
        if self.workload < 0.2:
            # Proactively look for prediction tasks
            asyncio.create_task(self._seek_proactive_work())
            return "seeking_work"
        elif self.workload > 0.8:
            return "defer_new_tasks"
        else:
            return "continue_current"

    async def _seek_proactive_work(self):
        """Proactively seek prediction work"""
        await self.broadcast({
            'type': 'capability_advertisement',
            'capabilities': [c.name for c in self.capabilities],
            'current_workload': self.workload,
            'estimated_response_time': 0.5
        })


# Example usage and testing
async def test_multi_agent_system():
    """Test the new multi-agent architecture"""

    # Set up message bus
    message_bus = MessageBus()

    # Create agents
    predictor = PropertyPredictorAgent(message_bus)

    # Simulate some agent communication
    # Let agents initialize
    await asyncio.sleep(1)

    # Test request-response
    conversation_id = await predictor.send_request(
        "literature_agent",  # Would exist in full system
        {
            'task_type': 'find_papers',
            'topic': 'aspirin toxicity'
        }
    )

    print(f"Sent request with conversation ID: {conversation_id}")

    # Test autonomous decision making
    action = predictor.decide_next_action({'current_time': datetime.now()})
    print(f"Agent decided: {action}")


if __name__ == "__main__":
    # Create logs directory
    import os
    os.makedirs('logs/agents', exist_ok=True)

    # Run test
    asyncio.run(test_multi_agent_system())
