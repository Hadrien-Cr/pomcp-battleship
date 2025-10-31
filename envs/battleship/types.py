from enum import Enum
import random
import pygame
import copy
import numpy as np

CELL_SIZE = 40
MARGIN = 10
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

    def __sub__(self, other):
        x = self.x - other.x
        y = self.y - other.y
        return Coord(x, y)

    def __eq__(self, value) -> bool:
        return self.x == value.x and self.y == value.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __str__(self):
        return "Coord({},{})".format(self.x, self.y)

    def _is_valid(self):
        return self.x >= 0 and self.y >= 0 and self.x < 10 and self.y < 10


class Compass(Enum):
    North = Coord(0, 1)
    East = Coord(1, 0)
    South = Coord(0, -1)
    West = Coord(-1, 0)
    NorthEast = Coord(1, 1)
    SouthEast = Coord(1, -1)
    SouthWest = Coord(-1, -1)
    NorthWest = Coord(-1, 1)
    Null = Coord(0, 0)

    @staticmethod
    def get_coord(idx) -> Coord:
        return list(Compass)[idx].value


def get_occupation_coords(pos: Coord, direction: Coord, length: int) -> list[Coord]:
    return [pos + i * direction for i in range(length)]


class Ship:
    def __init__(
        self,
        coord: Coord | None = None,
        direction: Coord | None = None,
        length: int | None = None,
    ):
        if coord is None:
            coord = Coord(random.randint(0, 9), random.randint(0, 9))
        if direction is None:
            direction = Compass.get_coord(random.randint(0, 3))
        if length is None:
            length = random.randint(1, 5)

        self.pos = coord
        self.direction = direction
        self.length = length

    def __repr__(self):
        return f"Ship({self.pos}, {self.direction}, {self.length})"

    def _is_valid(self) -> bool:
        return all(
            [
                coord._is_valid()
                for coord in get_occupation_coords(
                    self.pos, self.direction, self.length
                )
            ]
        )

    def __hash__(self) -> int:
        return hash((self.pos, self.direction, self.length))


class State_Battleship:
    def __init__(self, ships: list[Ship]) -> None:
        self.ships = ships

    def __repr__(self):
        return f"State_Battleship(ships={self.ships})"

    def __eq__(self, other):
        return isinstance(other, State_Battleship) and self.ships == other.ships

    def __hash__(self):
        return hash(tuple(self.ships))

    def is_occupied(self, coord) -> bool:
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

    def ship_adjacent(self, ship) -> bool:
        for coord in get_occupation_coords(ship.pos, ship.direction, ship.length):
            for other_ship in self.ships:
                if other_ship == ship:
                    continue

                for ship_coord in get_occupation_coords(
                    other_ship.pos, other_ship.direction, other_ship.length
                ):
                    for adj in range(8):
                        adj_pos = ship_coord + Compass.get_coord(adj)
                        if adj_pos == coord:
                            return True
        return False

    def _is_coherent_with_history(
        self,
        history: list,
    ) -> bool:
        if not self._is_valid():
            return False
        for a, o in history:
            if o.name == "hit" and not self.is_occupied(a.coord):
                return False
            elif o.name == "miss" and self.is_occupied(a.coord):
                return False
        return True

    def _is_valid(self) -> bool:
        for ship in self.ships:
            if not ship._is_valid() or self.ship_adjacent(ship):
                return False
        return True

    def _ship_swap(self) -> "State_Battleship":
        """
        2 ships of different sizes swapped location
        """
        while True:
            i, j = np.random.choice(len(self.ships), 2, replace=False)
            i, j = sorted((i, j))
            if all(
                [
                    coord._is_valid()
                    for coord in get_occupation_coords(
                        self.ships[j].pos, self.ships[j].direction, self.ships[i].length
                    )
                ]
            ) and all(
                [
                    coord._is_valid()
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
                new_state.ships[i].direction, new_state.ships[j].direction = (
                    self.ships[j].direction,
                    self.ships[i].direction,
                )
                return new_state

            else:
                d = self.ships[i].length - self.ships[j].length
                if all(
                    [
                        coord._is_valid()
                        for coord in get_occupation_coords(
                            self.ships[j].pos,
                            self.ships[j].direction,
                            self.ships[i].length,
                        )
                    ]
                ) and all(
                    [
                        coord._is_valid()
                        for coord in get_occupation_coords(
                            self.ships[i].pos + d * self.ships[i].direction,
                            self.ships[i].direction,
                            self.ships[j].length,
                        )
                    ]
                ):
                    new_state = copy.deepcopy(self)
                    new_state.ships[i].pos, new_state.ships[j].pos = (
                        self.ships[j].pos,
                        self.ships[i].pos + d * self.ships[i].direction,
                    )
                    new_state.ships[i].direction, new_state.ships[j].direction = (
                        self.ships[j].direction,
                        self.ships[i].direction,
                    )
                    return new_state
                if all(
                    [
                        coord._is_valid()
                        for coord in get_occupation_coords(
                            self.ships[i].pos,
                            self.ships[i].direction,
                            self.ships[j].length,
                        )
                    ]
                ) and all(
                    [
                        coord._is_valid()
                        for coord in get_occupation_coords(
                            self.ships[j].pos - d * self.ships[j].direction,
                            self.ships[j].direction,
                            self.ships[i].length,
                        )
                    ]
                ):
                    new_state = copy.deepcopy(self)
                    new_state.ships[i].pos, new_state.ships[j].pos = (
                        self.ships[i].pos,
                        self.ships[j].pos - d * self.ships[j].direction,
                    )
                    new_state.ships[i].direction, new_state.ships[j].direction = (
                        self.ships[i].direction,
                        self.ships[j].direction,
                    )
                    return new_state

    def _ship_merge(self) -> list["State_Battleship"]:
        """2 smaller ships were swapped into the location of 1 larger ship"""

        def triplet_merge(i, j, k) -> list["State_Battleship"]:
            new_state = copy.deepcopy(self)

            len_i = self.ships[i].length
            len_k = self.ships[k].length

            new_state.ships[i].pos = self.ships[k].pos
            new_state.ships[j].pos = (
                self.ships[k].pos + (len_i + 1) * self.ships[k].direction
            )

            new_state.ships[i].direction = self.ships[k].direction
            new_state.ships[j].direction = self.ships[k].direction

            outputs = []

            if all(
                [
                    coord._is_valid()
                    for coord in get_occupation_coords(
                        self.ships[i].pos, self.ships[i].direction, len_k
                    )
                ]
            ):
                new_state.ships[k].pos = self.ships[i].pos
                new_state.ships[k].direction = self.ships[i].direction
                outputs.append(new_state)

            if all(
                [
                    coord._is_valid()
                    for coord in get_occupation_coords(
                        self.ships[j].pos, self.ships[j].direction, len_k
                    )
                ]
            ):
                new_state.ships[k].pos = self.ships[j].pos
                new_state.ships[k].direction = self.ships[j].direction
                outputs.append(new_state)

            random_coord = Coord(
                random.randint(0, 9),
                random.randint(0, 9),
            )
            random_direction = Compass.get_coord(random.randint(0, 3))

            while not all(
                [
                    coord._is_valid()
                    for coord in get_occupation_coords(
                        random_coord, random_direction, len_k
                    )
                ]
            ):
                random_coord = Coord(
                    random.randint(0, 9),
                    random.randint(0, 9),
                )
                random_direction = Compass.get_coord(random.randint(0, 3))

            new_state.ships[k].pos = random_coord
            new_state.ships[k].direction = random_direction
            outputs.append(new_state)

            return outputs

        while True:
            i, j, k = np.random.choice(len(self.ships), 3, replace=False)
            if self.ships[i].length + self.ships[j].length < self.ships[k].length:
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
            if new_state._is_valid():
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

        font = pygame.font.SysFont(
            None, 24
        )  # Choose a font and size appropriate for your cell size

        # Draw column numbers
        for x in range(BOARD_SIZE):
            num_surf = font.render(str(x), True, pygame.Color("black"))
            num_rect = num_surf.get_rect(
                center=(
                    MARGIN + x * (CELL_SIZE + MARGIN) + CELL_SIZE // 2,
                    MARGIN // 2,  # half margin above grid
                )
            )
            self.screen.blit(num_surf, num_rect)

        # Draw row numbers
        for y in range(BOARD_SIZE):
            num_surf = font.render(str(y), True, pygame.Color("black"))
            num_rect = num_surf.get_rect(
                center=(
                    MARGIN // 2,  # half margin to left of grid
                    MARGIN + y * (CELL_SIZE + MARGIN) + CELL_SIZE // 2,
                )
            )
            self.screen.blit(num_surf, num_rect)

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


def generate_random_state() -> State_Battleship:
    state = State_Battleship([])

    for length in [5, 4, 3, 2, 2]:
        ship = Ship(length=length)

        while not ship._is_valid() or state.ship_adjacent(ship):
            ship = Ship(length=length)

        state.ships.append(ship)

    return state


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
