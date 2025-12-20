
import unittest
from unittest.mock import MagicMock, patch
from bot import build_joke_graph, JokeState, Joke
from langchain_core.messages import AIMessage

class TestJokeBot(unittest.TestCase):

    @patch('bot.ChatOpenAI')
    @patch('bot.os.getenv')
    def test_writer_critic_success_flow(self, mock_getenv, mock_chat_openai):
        # Setup mocks
        mock_getenv.return_value = "fake-api-key"
        
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Define side effects for the LLM to simulate Writer and Critic
        def llm_side_effect(messages):
            content = messages[0].content
            if "writer_prompt" in content or "comedy writer" in content:
                return AIMessage(content="Why did the chicken cross the road? To get to the other side!")
            elif "critic_prompt" in content or "comedy critic" in content:
                return AIMessage(content="APPROVE")
            return AIMessage(content="Unknown")
            
        mock_llm.invoke.side_effect = llm_side_effect

        # Run Graph
        graph = build_joke_graph()
        
        # We need to simulate user input for the menu. 
        # Since 'show_menu' uses input(), we should mock that too or bypass it.
        # However, testing the whole graph end-to-end with input() is tricky.
        # Let's test the nodes individually or mock input() to select 'n' (next joke) then 'q' (quit).
        
        with patch('builtins.input', side_effect=['n', 'q']): 
            initial_state = JokeState(category="neutral", language="en")
            final_state = graph.invoke(initial_state)
            
            # Assertions
            self.assertEqual(len(final_state['jokes']), 1)
            self.assertEqual(final_state['jokes'][0].text, "Why did the chicken cross the road? To get to the other side!")
            self.assertEqual(final_state['approval_status'], "PENDING") # Reset after delivery
            self.assertEqual(final_state['retry_count'], 0)

    @patch('bot.ChatOpenAI')
    @patch('bot.os.getenv')
    def test_writer_critic_retry_flow(self, mock_getenv, mock_chat_openai):
        # Setup mocks
        mock_getenv.return_value = "fake-api-key"
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Scenario: Reject once, then Approve
        # Writer called -> Joke 1
        # Critic called -> Reject
        # Writer called -> Joke 2
        # Critic called -> Approve
        
        self.call_count = 0
        
        def llm_side_effect(messages):
            content = messages[0].content
            if "comedy writer" in content:
                self.call_count += 1
                return AIMessage(content=f"Joke attempt {self.call_count}")
            elif "comedy critic" in content:
                if self.call_count == 1:
                    return AIMessage(content="REJECT Too boring")
                else:
                    return AIMessage(content="APPROVE")
            return AIMessage(content="Unknown")
            
        mock_llm.invoke.side_effect = llm_side_effect

        graph = build_joke_graph()
        
        with patch('builtins.input', side_effect=['n', 'q']):
            initial_state = JokeState()
            final_state = graph.invoke(initial_state)
            
            self.assertEqual(len(final_state['jokes']), 1)
            self.assertEqual(final_state['jokes'][0].text, "Joke attempt 2")
            # We expect 2 calls to writer
            self.assertEqual(self.call_count, 2)

    @patch('bot.ChatOpenAI')
    @patch('bot.os.getenv')
    def test_max_retries_flow(self, mock_getenv, mock_chat_openai):
        # Setup mocks
        mock_getenv.return_value = "fake-api-key"
        mock_llm = MagicMock()
        mock_chat_openai.return_value = mock_llm
        
        # Scenario: Always Reject
        
        def llm_side_effect(messages):
            content = messages[0].content
            if "comedy writer" in content:
                return AIMessage(content="Bad Joke")
            elif "comedy critic" in content:
                return AIMessage(content="REJECT Not funny")
            return AIMessage(content="Unknown")
            
        mock_llm.invoke.side_effect = llm_side_effect

        graph = build_joke_graph()
        
        with patch('builtins.input', side_effect=['n', 'q']):
            initial_state = JokeState()
            final_state = graph.invoke(initial_state)
            
            # Should deliver the last joke anyway after 5 retries
            self.assertEqual(len(final_state['jokes']), 1)
            self.assertEqual(final_state['jokes'][0].text, "Bad Joke")
            
            # Since we reset state after delivery, we can't check retry_count in final state directly 
            # unless we inspect the trace, but the fact we got a joke means it passed through.
            # We can verify we didn't get stuck in infinite loop (test would timeout or fail).

if __name__ == '__main__':
    unittest.main()
