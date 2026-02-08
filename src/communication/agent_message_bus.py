"""
Agent Message Bus - Inter-Agent Communication System

Enables direct agent-to-agent communication via:
- Pub/Sub messaging pattern
- Shared blackboard for data exchange
- Async message handling
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, List, Any, Callable, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"          # Agent requesting action from another
    RESPONSE = "response"        # Response to a request
    BROADCAST = "broadcast"      # Message to all agents
    DATA_UPDATE = "data_update"  # Update shared data on blackboard
    STATUS = "status"            # Agent status update


@dataclass
class AgentMessage:
    """Structured message for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    message_type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: str = ""  # Empty for broadcasts
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    reply_to: str = ""  # ID of message being replied to
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.message_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "topic": self.topic,
            "payload": self.payload,
            "timestamp": self.timestamp
        }


class Blackboard:
    """
    Shared memory space for agents to exchange data
    
    Agents can write intermediate results here for other agents to read.
    """
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._history: List[Dict] = []
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def write(self, key: str, value: Any, agent: str = "unknown") -> None:
        """Write data to blackboard"""
        async with self._lock:
            old_value = self._data.get(key)
            self._data[key] = value
            self._history.append({
                "action": "write",
                "key": key,
                "agent": agent,
                "timestamp": time.time()
            })
            logger.debug(f"Blackboard: {agent} wrote to '{key}'")
            
            # Notify subscribers
            for callback in self._subscribers.get(key, []):
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key, value, old_value)
                    else:
                        callback(key, value, old_value)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")
    
    async def read(self, key: str, default: Any = None) -> Any:
        """Read data from blackboard"""
        return self._data.get(key, default)
    
    async def read_all(self) -> Dict[str, Any]:
        """Read all blackboard data"""
        return dict(self._data)
    
    def subscribe(self, key: str, callback: Callable) -> None:
        """Subscribe to changes on a key"""
        self._subscribers[key].append(callback)
    
    def unsubscribe(self, key: str, callback: Callable) -> None:
        """Unsubscribe from key changes"""
        if callback in self._subscribers[key]:
            self._subscribers[key].remove(callback)
    
    async def clear(self) -> None:
        """Clear all data"""
        async with self._lock:
            self._data.clear()
            self._history.append({
                "action": "clear",
                "timestamp": time.time()
            })


class MessageBus:
    """
    Central message bus for agent-to-agent communication
    
    Features:
    - Publish/Subscribe pattern
    - Direct messaging between agents
    - Broadcast capability
    - Message history tracking
    """
    
    def __init__(self):
        self._agents: Dict[str, 'AgentInterface'] = {}
        self._topic_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self._message_handlers: Dict[str, Callable] = {}
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._message_history: List[AgentMessage] = []
        self._running = False
        self.blackboard = Blackboard()
        
        logger.info("MessageBus initialized")
    
    def register_agent(self, agent_id: str, handler: Callable) -> None:
        """Register an agent with a message handler"""
        self._message_handlers[agent_id] = handler
        logger.info(f"Agent registered: {agent_id}")
    
    def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent"""
        if agent_id in self._message_handlers:
            del self._message_handlers[agent_id]
            # Remove from all topic subscriptions
            for topic_subs in self._topic_subscribers.values():
                topic_subs.discard(agent_id)
            logger.info(f"Agent unregistered: {agent_id}")
    
    def subscribe_topic(self, agent_id: str, topic: str) -> None:
        """Subscribe agent to a topic"""
        self._topic_subscribers[topic].add(agent_id)
        logger.debug(f"{agent_id} subscribed to topic: {topic}")
    
    def unsubscribe_topic(self, agent_id: str, topic: str) -> None:
        """Unsubscribe agent from a topic"""
        self._topic_subscribers[topic].discard(agent_id)
    
    async def publish(self, message: AgentMessage) -> None:
        """
        Publish a message to the bus
        
        - Broadcasts go to all topic subscribers
        - Direct messages go to specific recipient
        """
        self._message_history.append(message)
        
        # Keep history bounded
        if len(self._message_history) > 1000:
            self._message_history = self._message_history[-500:]
        
        if message.message_type == MessageType.BROADCAST:
            # Send to all subscribers of the topic
            recipients = self._topic_subscribers.get(message.topic, set())
            for agent_id in recipients:
                if agent_id != message.sender:  # Don't send to self
                    await self._deliver(agent_id, message)
            logger.debug(f"Broadcast from {message.sender}: {message.topic} to {len(recipients)} agents")
        
        elif message.recipient:
            # Direct message
            await self._deliver(message.recipient, message)
            logger.debug(f"Direct message: {message.sender} -> {message.recipient}")
        
        else:
            logger.warning(f"Message has no recipient and is not broadcast: {message.id}")
    
    async def _deliver(self, agent_id: str, message: AgentMessage) -> None:
        """Deliver message to an agent"""
        handler = self._message_handlers.get(agent_id)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Message delivery error to {agent_id}: {e}")
        else:
            logger.warning(f"No handler for agent: {agent_id}")
    
    async def request(
        self, 
        sender: str, 
        recipient: str, 
        topic: str, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """
        Send a request and wait for response
        
        This is a synchronous request-response pattern.
        """
        request_msg = AgentMessage(
            message_type=MessageType.REQUEST,
            sender=sender,
            recipient=recipient,
            topic=topic,
            payload=payload
        )
        
        response_received = asyncio.Event()
        response_message: Optional[AgentMessage] = None
        
        async def response_handler(msg: AgentMessage):
            nonlocal response_message
            if msg.reply_to == request_msg.id:
                response_message = msg
                response_received.set()
        
        # Temporarily register response handler
        original_handler = self._message_handlers.get(sender)
        self._message_handlers[sender] = response_handler
        
        try:
            await self.publish(request_msg)
            await asyncio.wait_for(response_received.wait(), timeout=timeout)
            return response_message
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout: {sender} -> {recipient}")
            return None
        finally:
            # Restore original handler
            if original_handler:
                self._message_handlers[sender] = original_handler
    
    def get_registered_agents(self) -> List[str]:
        """Get list of registered agent IDs"""
        return list(self._message_handlers.keys())
    
    def get_message_history(self, limit: int = 50) -> List[Dict]:
        """Get recent message history"""
        return [m.to_dict() for m in self._message_history[-limit:]]


class CommunicatingAgent:
    """
    Base class for agents that can communicate via the message bus
    
    Extend this class to create agents with built-in communication.
    """
    
    def __init__(self, agent_id: str, bus: MessageBus):
        self.agent_id = agent_id
        self.bus = bus
        self._subscribed_topics: Set[str] = set()
        
        # Register with bus
        bus.register_agent(agent_id, self._handle_message)
        logger.info(f"CommunicatingAgent created: {agent_id}")
    
    def subscribe(self, topic: str) -> None:
        """Subscribe to a topic"""
        self.bus.subscribe_topic(self.agent_id, topic)
        self._subscribed_topics.add(topic)
    
    def unsubscribe(self, topic: str) -> None:
        """Unsubscribe from a topic"""
        self.bus.unsubscribe_topic(self.agent_id, topic)
        self._subscribed_topics.discard(topic)
    
    async def send(
        self, 
        recipient: str, 
        topic: str, 
        payload: Dict[str, Any]
    ) -> None:
        """Send a direct message to another agent"""
        msg = AgentMessage(
            message_type=MessageType.REQUEST,
            sender=self.agent_id,
            recipient=recipient,
            topic=topic,
            payload=payload
        )
        await self.bus.publish(msg)
    
    async def broadcast(self, topic: str, payload: Dict[str, Any]) -> None:
        """Broadcast a message to all topic subscribers"""
        msg = AgentMessage(
            message_type=MessageType.BROADCAST,
            sender=self.agent_id,
            topic=topic,
            payload=payload
        )
        await self.bus.publish(msg)
    
    async def request(
        self, 
        recipient: str, 
        topic: str, 
        payload: Dict[str, Any],
        timeout: float = 30.0
    ) -> Optional[AgentMessage]:
        """Send a request and wait for response"""
        return await self.bus.request(
            self.agent_id, recipient, topic, payload, timeout
        )
    
    async def reply(self, original_message: AgentMessage, payload: Dict[str, Any]) -> None:
        """Reply to a message"""
        response = AgentMessage(
            message_type=MessageType.RESPONSE,
            sender=self.agent_id,
            recipient=original_message.sender,
            topic=original_message.topic,
            payload=payload,
            reply_to=original_message.id
        )
        await self.bus.publish(response)
    
    async def write_to_blackboard(self, key: str, value: Any) -> None:
        """Write data to the shared blackboard"""
        await self.bus.blackboard.write(key, value, self.agent_id)
    
    async def read_from_blackboard(self, key: str, default: Any = None) -> Any:
        """Read data from the shared blackboard"""
        return await self.bus.blackboard.read(key, default)
    
    async def _handle_message(self, message: AgentMessage) -> None:
        """
        Handle incoming messages - override in subclass
        """
        logger.debug(f"{self.agent_id} received: {message.topic}")
        await self.on_message(message)
    
    async def on_message(self, message: AgentMessage) -> None:
        """
        Override this method to handle incoming messages
        """
        pass


# Global message bus instance
_global_bus: Optional[MessageBus] = None


def get_message_bus() -> MessageBus:
    """Get the global message bus instance"""
    global _global_bus
    if _global_bus is None:
        _global_bus = MessageBus()
    return _global_bus


def reset_message_bus() -> MessageBus:
    """Reset the global message bus (for testing)"""
    global _global_bus
    _global_bus = MessageBus()
    return _global_bus


# Example usage
if __name__ == "__main__":
    async def demo():
        print("=" * 60)
        print("Agent Message Bus Demo")
        print("=" * 60)
        
        # Create bus
        bus = MessageBus()
        
        # Create test agents
        class TestAgent(CommunicatingAgent):
            def __init__(self, agent_id: str, bus: MessageBus):
                super().__init__(agent_id, bus)
                self.received_messages = []
            
            async def on_message(self, message: AgentMessage):
                self.received_messages.append(message)
                print(f"  {self.agent_id} received: {message.topic} from {message.sender}")
                
                # Auto-reply to requests
                if message.message_type == MessageType.REQUEST:
                    await self.reply(message, {"status": "OK", "from": self.agent_id})
        
        # Create agents
        agent_a = TestAgent("PropertyPredictor", bus)
        agent_b = TestAgent("DockingAgent", bus)
        agent_c = TestAgent("RLGenerator", bus)
        
        # Subscribe to topics
        agent_a.subscribe("molecule_generated")
        agent_b.subscribe("molecule_generated")
        agent_b.subscribe("docking_request")
        agent_c.subscribe("optimization_feedback")
        
        print("\n📡 Test 1: Direct messaging")
        await agent_a.send("DockingAgent", "docking_request", {
            "smiles": "CCO",
            "protein": "EGFR"
        })
        await asyncio.sleep(0.1)
        
        print(f"  DockingAgent received {len(agent_b.received_messages)} messages")
        
        print("\n📢 Test 2: Broadcast messaging")
        await agent_c.broadcast("molecule_generated", {
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "score": -8.5
        })
        await asyncio.sleep(0.1)
        
        print(f"  PropertyPredictor received: {len(agent_a.received_messages)}")
        print(f"  DockingAgent received: {len(agent_b.received_messages)}")
        
        print("\n📝 Test 3: Blackboard shared memory")
        await agent_a.write_to_blackboard("best_molecule", {
            "smiles": "CCO",
            "qed": 0.85
        })
        
        result = await agent_b.read_from_blackboard("best_molecule")
        print(f"  DockingAgent read from blackboard: {result}")
        
        print("\n✅ Message bus demo complete!")
        print(f"   Registered agents: {bus.get_registered_agents()}")
        print(f"   Total messages: {len(bus.get_message_history())}")
    
    asyncio.run(demo())
