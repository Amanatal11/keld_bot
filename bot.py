import operator
from typing import Annotated, List, Literal, TypedDict
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pyjokes import get_joke

# 1. Define the State
class Joke(BaseModel):
    text: str
    category: str

def add_jokes(left: List[Joke], right: List[Joke]) -> List[Joke]:
    return left + right

class JokeState(BaseModel):
    jokes: Annotated[List[Joke], add_jokes] = []
    jokes_choice: Literal["n", "c", "q"] = "n" # next joke, change category, or quit
    category: str = "neutral"
    language: str = "en"
    quit: bool = False

# 2. Write Your Node Functions

def show_menu(state: JokeState) -> dict:
    print(f"\n============================================================")
    print(f"🎭 Menu | Category: {state.category.upper()} | Jokes: {len(state.jokes)}")
    print(f"--------------------------------------------------")
    print(f"Pick an option:")
    print(f"[n] 🎭 Next Joke  [c] 📂 Change Category  [q] 🚪 Quit")
    user_input = input("User Input: ").strip().lower()
    return {"jokes_choice": user_input}

def fetch_joke(state: JokeState) -> dict:
    joke_text = get_joke(language=state.language, category=state.category)
    new_joke = Joke(text=joke_text, category=state.category)
    print(f"\n😂 {joke_text}")
    return {"jokes": [new_joke]}

def update_category(state: JokeState) -> dict:
    categories = ["neutral", "chuck", "all"]
    print(f"\nSelect category [0=neutral, 1=chuck, 2=all]: ")
    try:
        selection = int(input("> ").strip())
        if 0 <= selection < len(categories):
            return {"category": categories[selection]}
        else:
            print("Invalid selection, keeping current category.")
            return {}
    except ValueError:
        print("Invalid input, keeping current category.")
        return {}

def exit_bot(state: JokeState) -> dict:
    return {"quit": True}

def route_choice(state: JokeState) -> str:
    if state.jokes_choice == "n":
        return "fetch_joke"
    elif state.jokes_choice == "c":
        return "update_category"
    elif state.jokes_choice == "q":
        return "exit_bot"
    return "exit_bot"

# Steps 3 & 4. Create the Graph and Add Nodes + Edges

def build_joke_graph() -> CompiledStateGraph:
    workflow = StateGraph(JokeState)

    workflow.add_node("show_menu", show_menu)
    workflow.add_node("fetch_joke", fetch_joke)
    workflow.add_node("update_category", update_category)
    workflow.add_node("exit_bot", exit_bot)

    workflow.set_entry_point("show_menu")

    workflow.add_conditional_edges(
        "show_menu",
        route_choice,
        {
            "fetch_joke": "fetch_joke",
            "update_category": "update_category",
            "exit_bot": "exit_bot",
        }
    )

    workflow.add_edge("fetch_joke", "show_menu")
    workflow.add_edge("update_category", "show_menu")
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
