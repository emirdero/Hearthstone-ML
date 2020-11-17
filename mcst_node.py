class MCSTNode():
    def __init__(self, game):
        self.children = []
        self.wins = 0
        self.number_of_simulations = 0
        self.state = game
        self.character = 0
        self.target = 0