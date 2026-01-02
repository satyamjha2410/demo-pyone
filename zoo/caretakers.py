class Keeper:
    # class variable
    role = "Animal Care Specialist"

    def __init__(self, keeper_name):
        self.keeper_name = keeper_name

    def feed_animal(self, animal_name):
        return f"{self.keeper_name} is feeding {animal_name}"
