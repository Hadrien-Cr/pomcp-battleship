from enum import Enum
import random
import pygame
import copy
import numpy as np

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
        return self.x >= 0 and self.y >= 0 and self.x < 10 and self.y < 10


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


def get_occupation_coords(pos: Coord, direction: int, length: int) -> list[Coord]:
    return [pos + i * Compass.get_coord(direction) for i in range(length)]


class Ship:
    def __init__(self, coord: Coord, length: int, direction: int = 0):
        self.pos = coord
        self.direction = direction
        self.length = length

    def __repr__(self):
        return f"Ship({(self.pos.x, self.pos.y)}, {self.direction}, {self.length})"


class State_Battleship:
    def __init__(self):
        self.ships = []

    def is_occupied(self, coord):
        for ship in self.ships:
            for segment in get_occupation_coords(ship.pos, ship.direction, ship.length):
                if segment == coord:
                    return True
        return False

    def get_all_occupied(self) -> list[Coord]:
        coords = []
        for ship in self.ships:
            coords.extend(get_occupation_coords(ship.pos, ship.direction, ship.length))
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

    def _is_coherent_with_history(
        self,
        history: list,
    ) -> bool:
        for a, o in history:
            if o.name == "hit" and not self.is_occupied(a.coord):
                return False
            elif o.name == "miss" and self.is_occupied(a.coord):
                return False
        return True

    def _ship_swap(self) -> "State_Battleship":
        """
        2 ships of different sizes swapped location
        """
        while True:
            i, j = np.random.choice(len(self.ships), 2, replace=False)
            if all(
                [
                    coord.is_valid()
                    for coord in get_occupation_coords(
                        self.ships[j].pos, self.ships[j].direction, self.ships[i].length
                    )
                ]
            ) and all(
                [
                    coord.is_valid()
                    for coord in get_occupation_coords(
                        self.ships[i].pos, self.ships[i].direction, self.ships[j].length
                    )
                ]
            ):
                new_state = copy.deepcopy(self)
                new_state.ships[i].pos, new_state.ships[j].pos = (
                    self.ships[j].pos,
                    self.ships[i].pos,
                )
                return new_state

    def _ship_merge(self) -> "State_Battleship":
        """2 smaller ships were swapped into the location of 1 larger ship"""

        def triplet_merge(i, j, k) -> "State_Battleship":
            new_state = copy.deepcopy(self)

            len_i = self.ships[i].length
            len_k = self.ships[k].length

            new_state.ships[i].pos, new_state.ships[j].pos = self.ships[
                k
            ].pos, self.ships[k].pos + len_i * Compass.get_coord(
                self.ships[k].direction
            )
            new_state.ships[i].direction = self.ships[k].direction
            new_state.ships[j].direction = self.ships[k].direction

            if all(
                [
                    coord.is_valid()
                    for coord in get_occupation_coords(
                        self.ships[i].pos, self.ships[i].direction, len_k
                    )
                ]
            ) and not self.ship_collision(
                Ship(self.ships[i].pos, len_k, self.ships[i].direction)
            ):
                new_state.ships[k].pos = self.ships[i].pos
                new_state.ships[k].direction = self.ships[i].direction

            elif all(
                [
                    coord.is_valid()
                    for coord in get_occupation_coords(
                        self.ships[j].pos, self.ships[j].direction, len_k
                    )
                ]
            ) and not self.ship_collision(
                Ship(self.ships[j].pos, len_k, self.ships[j].direction)
            ):
                new_state.ships[k].pos = self.ships[j].pos
                new_state.ships[k].direction = self.ships[j].direction

            else:
                random_coord = Coord(
                    random.randint(0, 9),
                    random.randint(0, 9),
                )
                random_direction = random.randint(0, 3)

                while not all(
                    [
                        coord.is_valid()
                        for coord in get_occupation_coords(
                            random_coord, random_direction, len_k
                        )
                    ]
                ):
                    random_coord = Coord(
                        random.randint(0, 9),
                        random.randint(0, 9),
                    )
                    random_direction = random.randint(0, 3)

                new_state.ships[k].pos = random_coord
                new_state.ships[k].direction = random_direction
            return new_state

        while True:
            i, j, k = np.random.choice(len(self.ships), 3, replace=False)
            if self.ships[i].length + self.ships[j].length <= self.ships[k].length:
                return triplet_merge(i, j, k)

    def _ship_move(self) -> "State_Battleship":
        """1 to 4 ships were moved to a new location, selected uniformly at random, and accepted if the new configuration was legal"""
        while True:
            i, j, k, l = np.random.choice(len(self.ships), 4, replace=False)
            new_state = copy.deepcopy(self)
            new_state.ships[i].pos = Coord(random.randint(0, 9), random.randint(0, 9))
            new_state.ships[j].pos = Coord(random.randint(0, 9), random.randint(0, 9))
            new_state.ships[k].pos = Coord(random.randint(0, 9), random.randint(0, 9))
            new_state.ships[l].pos = Coord(random.randint(0, 9), random.randint(0, 9))
            return new_state

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
