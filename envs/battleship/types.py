from enum import Enum
import random
import pygame

CELL_SIZE = 40
MARGIN = 5
BOARD_SIZE = 10
WINDOW_SIZE = (
    BOARD_SIZE * (CELL_SIZE + MARGIN) + MARGIN,
    BOARD_SIZE * (CELL_SIZE + MARGIN) + MARGIN,
)


class Coord:
    x: int
    y: int

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __mul__(self, other):
        if isinstance(other, Coord):
            return Coord(self.x * other.x, self.y * other.y)

        elif isinstance(other, (int, float)):
            return Coord(self.x * other, self.y * other)
        else:
            return NotImplemented  # allows other types to handle multiplication

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        x = self.x + other.x
        y = self.y + other.y
        return Coord(x, y)

    def __eq__(self, value) -> bool:
        return self.x == value.x and self.y == value.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self):
        return "Coord({},{})".format(self.x, self.y)

    def is_valid(self):
        return self.x >= 0 and self.y >= 0


class Compass(Enum):
    North = Coord(0, 1)
    East = Coord(1, 0)
    South = Coord(0, -1)
    West = Coord(-1, 0)
    Null = Coord(0, 0)
    NorthEast = Coord(1, 1)
    SouthEast = Coord(1, -1)
    SouthWest = Coord(-1, -1)
    NorthWest = Coord(-1, 1)

    @staticmethod
    def get_coord(idx):
        return list(Compass)[idx].value


class Ship:
    def __init__(self, coord, length):
        self.pos = coord
        self.direction = random.randint(0, 3)
        self.length = length

    def __repr__(self):
        return f"Ship({(self.pos.x, self.pos.y)}, {self.direction}, {self.length})"


class State_Battleship:
    def __init__(self):
        self.ships = []

    def is_occupied(self, coord):
        for ship in self.ships:
            for i in range(ship.length):
                segment = ship.pos + i * Compass.get_coord(ship.direction)
                if segment == coord:
                    return True
        return False

    def get_all_occupied(self) -> list[Coord]:
        coords = []
        for ship in self.ships:
            for i in range(ship.length):
                segment = ship.pos + i * Compass.get_coord(ship.direction)
                coords.append(segment)
        return coords

    def ship_collision(self, ship):
        for i in range(ship.length):
            target_pos = ship.pos + i * Compass.get_coord(ship.direction)
            if not (0 <= target_pos.x < 10 and 0 <= target_pos.y < 10):
                return True
            if self.is_occupied(target_pos):
                return True
            for adj in range(8):
                adj_pos = target_pos + Compass.get_coord(adj)
                if 0 <= adj_pos.x < 10 and 0 <= adj_pos.y < 10:
                    if self.is_occupied(adj_pos):
                        return True
        return False

    def __repr__(self):
        return f"State_Battleship(ships={self.ships})"

    def __eq__(self, other):
        return isinstance(other, State_Battleship) and self.ships == other.ships

    def __hash__(self):
        return hash(tuple(self.ships))

    def init_window(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode(WINDOW_SIZE)
        pygame.display.set_caption("Battleship")
        self.clock = pygame.time.Clock()
        self.running = True

    def render(self, history: list):

        if not hasattr(self, "screen"):
            self.init_window()

        hits = {a.coord for a, o in history if o.name == "hit"}
        misses = {a.coord for a, o in history if o.name == "miss"}

        self.screen.fill(pygame.Color("white"))

        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                rect = pygame.Rect(
                    MARGIN + x * (CELL_SIZE + MARGIN),
                    MARGIN + y * (CELL_SIZE + MARGIN),
                    CELL_SIZE,
                    CELL_SIZE,
                )
                coord = Coord(x, y)

                color = pygame.Color("lightblue")
                if self.is_occupied(coord):
                    color = pygame.Color("gray")
                if coord in misses:
                    color = pygame.Color("blue")
                elif coord in hits:
                    color = pygame.Color("red")

                pygame.draw.rect(self.screen, color, rect)

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def quit(self):
        pygame.quit()


def get_random_state() -> State_Battleship:
    ship_set = State_Battleship()

    for length in reversed(range(2, 6)):
        while True:  # add one ship of each kind
            ship = Ship(
                coord=Coord(random.randint(0, 9), random.randint(0, 9)), length=length
            )
            if not ship_set.ship_collision(ship):
                break
        ship_set.ships.append(ship)
    return ship_set


class Action_Battleship:
    def __init__(self, coord: Coord):
        self.coord = coord

    def __repr__(self):
        return f"Action_Battleship({self.coord.x, self.coord.y})"

    def __eq__(self, other):
        return self.coord == other.coord and type(self) == type(other)

    def __hash__(self):
        return hash(self.coord)


class Observation_Battleship:
    def __init__(self, name: str):
        self.name = name
        if name not in ["hit", "miss"]:
            raise ValueError("Invalid state: %s" % name)

    def __repr__(self):
        return f"Obs-BattleShip({self.name})"  # self.name

    def __eq__(self, other):
        return self.name == other.name and type(self) == type(other)

    def __hash__(self):
        return hash(self.name)
