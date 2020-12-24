class Lane():
    def __init__(self, id: str, capacity: int, vehicles=0) -> None:
        self.id = id
        self.capacity = capacity
        self.vehicles = vehicles

    def get_capacity(self) -> int:
        return self.capacity

    def set_vehicles(self, vehicles) -> int:
        self.vehicles = vehicles

    def get_vehicles(self) -> int:
        return self.vehicles

    def get_id(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return "Lane[ id {} , capacity {} ]".format(
            self.id,
            self.capacity
        )
