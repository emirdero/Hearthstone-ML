from fireplace import cards, deck
from fireplace.player import Player
from fireplace.game import Game
from fireplace.exceptions import GameOver
from hearthstone.enums import CardClass, CardType
from fireplace.deck import Deck
import copy
import random
from model import CharacterSelectionModel
from mcst_node import MCSTNode
import torch

def has_lethal(player, health_of_opponent):
	total_attack_power = 0
	for character in player.characters:
		total_attack_power += character.atk
	if total_attack_power >= health_of_opponent:
		return True
	else:
		return False

def get_opponent(game):
	opponent = None 
	for game_player in game.players:
		if game_player != game.current_player:
			opponent = game_player
	return opponent
	

def who_wins(game):
	player = game.current_player
	opponent = get_opponent(game)
	
	if has_lethal(player, opponent.hero.health):
		return player

def play_turn_random(game):
	player = game.current_player
	while True:
		# iterate over our hand and play whatever is playable
		for card in player.hand:
			if card.is_playable() and random.random() < 0.5:
				target = None
				if card.must_choose_one:
					card = random.choice(card.choose_cards)
				if card.requires_target():
					target = random.choice(card.targets)
				card.play(target=target)

				if player.choice:
					choice = random.choice(player.choice.cards)
					player.choice.choose(choice)
				continue

		# Randomly attack with whatever can attack
		for character in player.characters:
			if character.can_attack():
				character.attack(random.choice(character.targets))
		break

	game.end_turn()
	return game

def mcst_simulation(game):
	root = MCSTNode()
	

def play_turn_mcst(game):
	player = game.current_player
	state = game_state(game)
	model = CharacterSelectionModel()

	result = model.forward(state)
	print(model.loss(result, 5))

	while True:
		# iterate over our hand and play whatever is playable
		for card in player.hand:
			if card.is_playable() and random.random() < 0.5:
				target = None
				if card.must_choose_one:
					card = random.choice(card.choose_cards)
				if card.requires_target():
					target = random.choice(card.targets)
				card.play(target=target)

				if player.choice:
					choice = random.choice(player.choice.cards)
					player.choice.choose(choice)
				continue

		# Randomly attack with whatever can attack
		for character in player.characters:
			if character.can_attack():
				character.attack(random.choice(character.targets))
		break

	game.end_turn()
	return game

def play_full_game(game):
	for player in game.players:
		print("Can mulligan %r" % (player.choice.cards))
		mull_count = random.randint(0, len(player.choice.cards))
		cards_to_mulligan = random.sample(player.choice.cards, mull_count)
		player.choice.choose(*cards_to_mulligan)

	while True:
		play_turn_mcst(game)

	return game

def simple_deck():
	cards = ["CS2_182", "CS2_200", "CS2_231", "CS2_168", "CS2_172", "CS2_120", "CS2_118", "CS2_186", "DRG_239", "NEW1_021", "EX1_096", "EX1_007", "CS2_119", "BT_728"]
	deck = []
	for card in cards:
		for i in range(0, 2):
			deck.append(card)
	return deck

def game_state(game):
	player = game.current_player
	opponent = get_opponent(game)
	characters = []
	player_entities = []

	for character in player.characters:
		characters.append(character.atk)
		characters.append(character._max_health - character.damage)
		player_entities.append(character)
	
	# Padding
	if len(characters) < 16:
		for i in range(0, 16-len(characters)):
			characters.append(0)

	for character in opponent.characters:
		characters.append(float(character.atk))
		characters.append(float(character._max_health - character.damage))

	# Padding
	if len(characters) < 32:
		for i in range(0, 32-len(characters)):
			characters.append(0)
	
	return torch.tensor(characters)

def setup_game():
	deck1 = simple_deck()
	deck2 = simple_deck() #random_draft(CardClass.WARRIOR)
	print(deck1)
	player1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
	player2 = Player("Player2", deck2, CardClass.MAGE.default_hero)

	game = Game(players=(player1, player2))
	game.start()

	return game

if __name__ == "__main__":
	cards.db.initialize()
	game = setup_game()
	try:
		play_full_game(game)
	except GameOver:
		print("Game completed normally.")