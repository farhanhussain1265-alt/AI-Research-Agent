from backend.graph.router_node import router_node

test_state = {
    "topic": "Latest developments in Agentic AI",
    "as_of": "2026-11-12"
}

result = router_node(test_state)

print("\nRouter Output:")
print(result)

print("\nDecision:")
