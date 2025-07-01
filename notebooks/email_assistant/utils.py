"""Utility functions for the email assistant."""

def show_graph(app):
    """Display a simple text representation of the graph structure.
    
    Args:
        app: The compiled LangGraph application
    """
    try:
        # Try to use the built-in get_graph method if available
        graph = app.get_graph()
        print("Graph Structure:")
        print(f"Nodes: {list(graph.nodes.keys())}")
        print(f"Edges: {[(edge.source, edge.target) for edge in graph.edges]}")
        
        # Try to display more detailed information if available
        print("\nDetailed Node Information:")
        for node_id, node_data in graph.nodes.items():
            print(f"  {node_id}: {type(node_data).__name__}")
            
    except Exception as e:
        print("Basic graph visualization:")
        print(f"Graph object type: {type(app)}")
        try:
            # Alternative way to get some graph info
            if hasattr(app, 'nodes'):
                print(f"Available nodes: {list(app.nodes.keys()) if app.nodes else 'None'}")
            if hasattr(app, 'edges'):
                print(f"Available edges: {app.edges if app.edges else 'None'}")
        except:
            pass
        print(f"Note: Full graph visualization not available: {e}")
