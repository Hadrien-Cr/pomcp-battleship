class State_Tiger:
    def __init__(self, name: str, n: int):
        self.name = name
        self.n = n
        if name not in [f"tiger-{s}" for s in range(self.n)]:
            raise ValueError("Invalid state: %s" % name)

    def __repr__(self):
        return f"State_{self.n}-Tiger({self.name})"  # self.name

    def __eq__(self, other):
        return self.name == other.name and type(self) == type(other)

    def __hash__(self):
        return hash(self.name)


class Action_Tiger:
    def __init__(self, name: str, n: int):
        self.name = name
        self.n = n
        if name not in [f"open-{s}" for s in range(self.n)] + ["listen"]:
            raise ValueError("Invalid action: %s" % name)

    def __repr__(self):
        return f"Act_{self.n}-Tiger({self.name})"  # self.name

    def __eq__(self, other):
        return self.name == other.name and type(self) == type(other)

    def __hash__(self):
        return hash(self.name)


class Observation_Tiger:
    def __init__(self, name: str, n: int):
        self.name = name
        self.n = n
        if name not in [f"tiger-{s}" for s in range(self.n)]:
            raise ValueError("Invalid action: %s" % name)

    def __repr__(self):
        return f"Obs_{self.n}-Tiger({self.name})"  # self.name

    def __eq__(self, other):
        return self.name == other.name and type(self) == type(other)

    def __hash__(self):
        return hash(self.name)
