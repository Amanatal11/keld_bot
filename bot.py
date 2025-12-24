import operator
import random
import os
from typing import Annotated, List, Literal, TypedDict, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pyjokes import get_joke
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from utils import PromptBuilder

load_dotenv()

# 1. Define the State
class Joke(BaseModel):
    text: str
    category: str
    rating: Optional[int] = None

def reduce_jokes(left: List[Joke], right: List[Joke]) -> List[Joke]:
    for joke in right:
        if joke.text == "RESET_HISTORY":
            return []
    return left + right

class JokeState(BaseModel):
    jokes: Annotated[List[Joke], reduce_jokes] = []
    jokes_choice: Literal["n", "c", "l", "r", "q"] = "n" # next joke, change category, change language, reset, or quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False
    
    # Writer-Critic Loop State
    current_joke: Optional[str] = None
    critique: Optional[str] = None
    approval_status: Literal["APPROVE", "REJECT", "PENDING"] = "PENDING"
    retry_count: int = 0

# 2. Write Your Node Functions

def show_menu(state: JokeState) -> dict:
    avg_rating = "N/A"
    rated_jokes = [j.rating for j in state.jokes if j.rating is not None]
    if rated_jokes:
        avg_rating = f"{sum(rated_jokes) / len(rated_jokes):.1f}⭐"

    print(f"\n============================================================")
    print(f"🎭 Menu | Category: {state.category.upper()} | Language: {state.language.upper()}")
    print(f"📊 Stats: {len(state.jokes)} jokes | Avg Rating: {avg_rating}")
    print(f"--------------------------------------------------")
    print(f"Pick an option:")
    print(f"[n] 🎭 Next Joke  [c] 📂 Change Category  [l] 🌐 Change Language  [r] 🔁 Reset History  [q] 🚪 Quit")
    while True:
        user_input = input("User Input: ").strip().lower()
        if user_input in ["n", "c", "l", "r", "q"]:
            return {"jokes_choice": user_input}
        print(f"Invalid input '{user_input}'. Please enter one of [n, c, l, r, q].")

def writer_node(state: JokeState) -> dict:
    prompt_builder = PromptBuilder()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("\n⚠️ OPENAI_API_KEY not found. Falling back to pyjokes.")
        joke_text = get_joke(language="en", category="neutral")
        return {
            "current_joke": joke_text,
            "approval_status": "APPROVE", # Skip critic if no API key
            "retry_count": 0
        }

    feedback = ""
    if state.critique:
        feedback = f"Previous attempt rejected. Critique: {state.critique}"
    
    prompt = prompt_builder.get_prompt(
        "writer_prompt", 
        category=state.category, 
        language=state.language,
        feedback=feedback
    )
    
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        response = llm.invoke([HumanMessage(content=prompt)])
        joke_text = response.content.strip()
        print(f"\n✍️  Writer generated: {joke_text}")
        return {"current_joke": joke_text, "retry_count": state.retry_count + 1}
    except Exception as e:
        print(f"\n⚠️ Writer API Error: {e}")
        print("🔄 Falling back to local pyjokes.")
        joke_text = get_joke(language="en", category="neutral")
        return {
            "current_joke": joke_text,
            "approval_status": "APPROVE",
            "retry_count": 0
        }

def critic_node(state: JokeState) -> dict:
    prompt_builder = PromptBuilder()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return {"approval_status": "APPROVE"} # Should be handled in writer, but safe guard

    prompt = prompt_builder.get_prompt(
        "critic_prompt",
        joke=state.current_joke,
        category=state.category
    )
    
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo")
        response = llm.invoke([HumanMessage(content=prompt)])
        result = response.content.strip()
        
        if result.startswith("APPROVE"):
            print(f"🕵️  Critic Approved!")
            return {"approval_status": "APPROVE", "critique": None}
        else:
            critique = result.replace("REJECT", "").strip()
            print(f"🕵️  Critic Rejected: {critique}")
            return {"approval_status": "REJECT", "critique": critique}
    except Exception as e:
        print(f"\n⚠️ Critic API Error: {e}")
        print("🔓 Critic failing open (approving).")
        return {"approval_status": "APPROVE"} # Fail open if critic dies

def deliver_joke(state: JokeState) -> dict:
    print(f"\n🎉 Final Joke: {state.current_joke}")
    return {}

def rate_joke(state: JokeState) -> dict:
    print(f"\n⭐ Rate this joke (1-5 stars, or press Enter to skip):")
    try:
        user_input = input("> ").strip()
        if not user_input:
            rating = None
        else:
            rating = int(user_input)
            if not (1 <= rating <= 5):
                print("Invalid rating, skipping.")
                rating = None
    except ValueError:
        print("Invalid input, skipping.")
        rating = None

    new_joke = Joke(text=state.current_joke, category=state.category, rating=rating)
    
    # Reset loop state for next time
    return {
        "jokes": [new_joke], 
        "current_joke": None, 
        "critique": None, 
        "approval_status": "PENDING", 
        "retry_count": 0
    }

def update_category(state: JokeState) -> dict:
    categories = ["neutral", "chuck", "all"]
    print(f"\nSelect category [0=neutral, 1=chuck, 2=all]: ")
    try:
        selection = int(input("> ").strip())
        if 0 <= selection < len(categories):
            # Reset loop state when category changes
            return {
                "category": categories[selection],
                "current_joke": None,
                "critique": None,
                "approval_status": "PENDING",
                "retry_count": 0
            }
        else:
            print("Invalid selection, keeping current category.")
            return {}
    except ValueError:
        print("Invalid input, keeping current category.")
        return {}

def update_language(state: JokeState) -> dict:
    languages = ["en", "de", "es", "am"]
    print(f"\nSelect language [0=en, 1=de, 2=es, 3=am]: ")
    try:
        selection = int(input("> ").strip())
        if 0 <= selection < len(languages):
            return {"language": languages[selection]}
        else:
            print("Invalid selection, keeping current language.")
            return {}
    except ValueError:
        print("Invalid input, keeping current language.")
        return {}

def reset_jokes(state: JokeState) -> dict:
    print(f"\n🧹 Joke history has been reset!")
    return {
        "jokes": [Joke(text="RESET_HISTORY", category="neutral")],
        "current_joke": None,
        "critique": None,
        "approval_status": "PENDING",
        "retry_count": 0
    }

def exit_bot(state: JokeState) -> dict:
    return {"quit": True}

def route_choice(state: JokeState) -> str:
    if state.jokes_choice == "n":
        return "writer_node"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "l":
        return "update_language"
    elif state.jokes_choice == "r":
        return "reset_jokes"
    elif state.jokes_choice == "q":
        return "exit_bot"
    return "exit_bot"

def route_critique(state: JokeState) -> str:
    if state.approval_status == "APPROVE":
        return "deliver_joke"
    elif state.retry_count >= 5:
        print(f"\n⚠️ Max retries reached. Delivering best effort.")
        return "deliver_joke"
    else:
        return "writer_node"

# Steps 3 & 4. Create the Graph and Add Nodes + Edges

def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)
    workflow.add_node("writer_node", writer_node)
    workflow.add_node("critic_node", critic_node)
    workflow.add_node("deliver_joke", deliver_joke)
    workflow.add_node("rate_joke", rate_joke)
    workflow.add_node("update_category", update_category)
    workflow.add_node("update_language", update_language)
    workflow.add_node("reset_jokes", reset_jokes)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "writer_node": "writer_node",
            "update_category": "update_category",
            "update_language": "update_language",
            "reset_jokes": "reset_jokes",
            "exit_bot": "exit_bot",
        }
    )

    workflow.add_edge("writer_node", "critic_node")
    
    workflow.add_conditional_edges(
        "critic_node",
        route_critique,
        {
            "deliver_joke": "deliver_joke",
            "writer_node": "writer_node"
        }
    )

    workflow.add_edge("deliver_joke", "rate_joke")
    workflow.add_edge("rate_joke", "show_menu")
    workflow.add_edge("update_category", "show_menu")
    workflow.add_edge("update_language", "show_menu")
    workflow.add_edge("reset_jokes", "show_menu")
    workflow.add_edge("exit_bot", END)

    return workflow.compile()

# 5. Run the Graph

def main():
    print("🎉==========================================================🎉")
    print("    WELCOME TO THE LANGGRAPH JOKE BOT!")
    print("    This example demonstrates agentic state flow without LLMs")
    print("============================================================")
    print("\n")
    print("🚀==========================================================🚀")
    print("    STARTING JOKE BOT SESSION...")
    print("============================================================")

    graph = build_joke_graph()
    final_state = graph.invoke(JokeState(), config={"recursion_limit": 100})

    print("\n🚪==========================================================🚪")
    print("    GOODBYE!")
    print("============================================================")
    print("\n🎊==========================================================🎊")
    print("    SESSION COMPLETE!")
    print("============================================================")
    print(f"    📈 You enjoyed {len(final_state['jokes'])} jokes during this session!")
    print(f"    📂 Final category: {final_state['category'].upper()}")
    print("    🙏 Thanks for using the LangGraph Joke Bot!")
    print("============================================================")

if __name__ == "__main__":
    main()
