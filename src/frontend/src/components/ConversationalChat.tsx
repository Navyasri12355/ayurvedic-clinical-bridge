import React, { useState, useRef, useEffect } from 'react';

/**
 * Conversational LLM Chat Component
 * Maintains conversation history with multi-turn interactions
 * Integrates with clinical reasoning engine
 */

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface ConversationalChatProps {
  userId: string;
  userRole: 'general_user' | 'qualified_practitioner';
}

const ConversationalChat: React.FC<ConversationalChatProps> = ({
  userId,
  userRole,
}) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationSummary, setConversationSummary] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch conversation summary on mount
  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await fetch(
          `/api/llm/conversation/${userId}/summary`,
          {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
            },
          }
        );
        if (response.ok) {
          const data = await response.json();
          setConversationSummary(data.summary.context_summary);
        }
      } catch (error) {
        console.error('Failed to fetch conversation summary:', error);
      }
    };

    fetchSummary();
  }, [userId]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString(),
    };

    setMessages([...messages, userMessage]);
    setInputValue('');
    setLoading(true);

    try {
      const response = await fetch('/api/llm/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
        },
        body: JSON.stringify({
          message: inputValue,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date().toISOString(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Failed to get response:', error);
      const errorMessage: Message = {
        role: 'assistant',
        content:
          'Sorry, I encountered an error. Please try again or consult a healthcare provider.',
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearConversation = async () => {
    if (window.confirm('Clear entire conversation history?')) {
      try {
        await fetch(`/api/llm/conversation/${userId}/clear`, {
          method: 'DELETE',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('access_token')}`,
          },
        });
        setMessages([]);
        setConversationSummary('');
      } catch (error) {
        console.error('Failed to clear conversation:', error);
      }
    }
  };

  return (
    <div className="conversation-chat">
      <div className="chat-header">
        <h2>Ayurvedic Clinical Assistant</h2>
        <div className="user-info">
          <span className="role-badge">
            {userRole === 'qualified_practitioner' ? '👨‍⚕️ Practitioner' : '👤 General User'}
          </span>
          <button
            className="clear-btn"
            onClick={handleClearConversation}
            title="Clear conversation history"
          >
            🗑️ Clear History
          </button>
        </div>
      </div>

      <div className="messages-container">
        {messages.length === 0 && (
          <div className="welcome-message">
            <h3>Start a Clinical Conversation</h3>
            <p>
              Ask about Ayurvedic herbs, symptoms, disease management, herb-drug
              interactions, or any clinical queries. The system maintains context
              across our conversation.
            </p>
            {conversationSummary && (
              <div className="context-box">
                <strong>Previous Context:</strong>
                <p>{conversationSummary}</p>
              </div>
            )}
          </div>
        )}

        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? '👤' : '🤖'}
            </div>
            <div className="message-content">
              <p>{msg.content}</p>
              <span className="timestamp">
                {new Date(msg.timestamp).toLocaleTimeString()}
              </span>
            </div>
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
          placeholder="Ask a follow-up question..."
          disabled={loading}
        />
        <button
          onClick={handleSendMessage}
          disabled={loading || !inputValue.trim()}
          className="send-btn"
        >
          {loading ? '⏳' : '📤'} Send
        </button>
      </div>

      <style jsx>{`
        .conversation-chat {
          display: flex;
          flex-direction: column;
          height: 100%;
          max-width: 900px;
          margin: 0 auto;
          background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
          border-radius: 12px;
          box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
          overflow: hidden;
        }

        .chat-header {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 20px;
          display: flex;
          justify-content: space-between;
          align-items: center;
        }

        .chat-header h2 {
          margin: 0;
          font-size: 24px;
        }

        .user-info {
          display: flex;
          gap: 10px;
          align-items: center;
        }

        .role-badge {
          background: rgba(255, 255, 255, 0.2);
          padding: 6px 12px;
          border-radius: 20px;
          font-size: 14px;
        }

        .clear-btn {
          background: rgba(255, 255, 255, 0.2);
          border: none;
          color: white;
          padding: 8px 12px;
          border-radius: 6px;
          cursor: pointer;
          font-size: 14px;
          transition: all 0.3s;
        }

        .clear-btn:hover {
          background: rgba(255, 255, 255, 0.3);
        }

        .messages-container {
          flex: 1;
          overflow-y: auto;
          padding: 20px;
          display: flex;
          flex-direction: column;
          gap: 12px;
        }

        .welcome-message {
          text-align: center;
          padding: 40px 20px;
          color: #555;
        }

        .context-box {
          background: white;
          padding: 15px;
          border-left: 4px solid #667eea;
          border-radius: 6px;
          margin-top: 15px;
          text-align: left;
          font-size: 14px;
        }

        .message {
          display: flex;
          gap: 12px;
          margin-bottom: 8px;
        }

        .message.user {
          justify-content: flex-end;
        }

        .message-avatar {
          font-size: 24px;
          min-width: 32px;
          text-align: center;
        }

        .message-content {
          background: white;
          padding: 12px 16px;
          border-radius: 12px;
          max-width: 70%;
          box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }

        .message.user .message-content {
          background: #667eea;
          color: white;
        }

        .message-content p {
          margin: 0 0 6px 0;
          line-height: 1.4;
        }

        .timestamp {
          font-size: 12px;
          opacity: 0.6;
        }

        .typing-indicator {
          display: flex;
          gap: 4px;
        }

        .typing-indicator span {
          width: 8px;
          height: 8px;
          background: #667eea;
          border-radius: 50%;
          animation: typing 1.4s infinite;
        }

        .typing-indicator span:nth-child(2) {
          animation-delay: 0.2s;
        }

        .typing-indicator span:nth-child(3) {
          animation-delay: 0.4s;
        }

        @keyframes typing {
          0%,
          60%,
          100% {
            opacity: 0.5;
            transform: translateY(0);
          }
          30% {
            opacity: 1;
            transform: translateY(-10px);
          }
        }

        .input-container {
          display: flex;
          gap: 10px;
          padding: 20px;
          background: white;
          border-top: 1px solid #e0e0e0;
        }

        .input-container input {
          flex: 1;
          padding: 12px 16px;
          border: 2px solid #e0e0e0;
          border-radius: 24px;
          font-size: 14px;
          transition: border-color 0.3s;
        }

        .input-container input:focus {
          outline: none;
          border-color: #667eea;
        }

        .input-container input:disabled {
          background: #f5f5f5;
          color: #999;
        }

        .send-btn {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          padding: 12px 24px;
          border-radius: 24px;
          cursor: pointer;
          font-weight: 600;
          transition: all 0.3s;
        }

        .send-btn:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        }

        .send-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        @media (max-width: 768px) {
          .message-content {
            max-width: 85%;
          }

          .chat-header {
            flex-direction: column;
            gap: 10px;
            align-items: flex-start;
          }

          .user-info {
            width: 100%;
            justify-content: space-between;
          }
        }
      `}</style>
    </div>
  );
};

export default ConversationalChat;
