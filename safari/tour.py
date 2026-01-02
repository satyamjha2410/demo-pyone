from zoo import Lion, Keeper

class SafariTour:
    # class variable
    location = "Serengeti"

    def __init__(self, tour_name):
        self.tour_name = tour_name

    def start_tour(self):
        lion = Lion("Simba", 5)
        keeper = Keeper("Ravi")

        lion_sound = lion.roar_sound()
        feeding = keeper.feed_animal(lion.name)

        return f"""
Tour: {self.tour_name}
Location: {self.location}
Animal: {lion.name} ({lion.species})
Sound: {lion_sound}
Action: {feeding}
"""
