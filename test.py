from src.tasks.scruples import ScruplesMonitor

# Set OPENROUTER_API_KEY env var, or pass directly
monitor = ScruplesMonitor(
    variant="first_person",
    scope="all_messages",
    model="openai/gpt-5-thinking",  # default
)

# Analyze a model response
result = monitor.analyze_with_answer(
    thinking="Let me consider this situation...",
    answer="B",
    post_title="AITA for...",
    post_text="Story here...",
)

print(result["monitor_detected"])  # True if sycophancy detected
