import uuid


class Node:
    def __init__(self, text, layer, embedding, children_ids, metadata):
        self.node_id = str(uuid.uuid4())
        self.text = text
        self.layer = layer
        self.embedding = embedding
        self.children_ids = children_ids or []
        self.metadata = metadata or {}

    def append_child(self, id):
        self.children_ids.append(id)


class RaptorTree:
    def __init__(self):
        self.layers = dict()

    def add_node(self, node):
        key = node.layer
        if key not in self.layers:
            self.layers[key] = []
        self.layers[key].append(node)

    def nodes_at(self, layer):
        return self.layers.get(layer, [])

    def all_nodes(self):
        result = []
        for node_list in self.layers.values():
            result.extend(node_list)
        return result

    @property
    def depth(self):
        return max(self.layers.keys()) if self.layers else 0
