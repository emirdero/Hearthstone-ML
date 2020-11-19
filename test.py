from fireplace_modified.fireplace_modified import cards, deck
from fireplace_modified.fireplace_modified.player import Player
from fireplace_modified.fireplace_modified.game import Game
from fireplace_modified.fireplace_modified.exceptions import GameOver
from hearthstone.enums import CardClass, CardType
from fireplace_modified.fireplace_modified.deck import Deck
import copy
import random
from model import TargetSelectionModel
from mcst_node import MCSTNode
import torch
import itertools
import math

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
	

def current_player_wins(game):
	original_player = game.current_player
	player = game.current_player
	opponent = get_opponent(game)
	while not has_lethal(player, opponent.hero.health):
		game = play_turn_random(game)
		player = game.current_player
		opponent = get_opponent(game)
	return player == original_player

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
"""
	player = game.current_player
	usable_characters = []
	for character in player.characters:
		if character.can_attack():
			usable_characters.append(character)
	if len(usable_characters) == 0:
		return 0
"""

def get_characters(game):
	player = game.current_player
	characters = []
	for character in player.characters:
		characters.append(character)
	return characters

def get_usable_characters(game):
	player = game.current_player
	usable_characters = []
	for character in player.characters:
		if character.can_attack():
			usable_characters.append(character)
	return usable_characters

# Returns best target for given character
def mcst_simulation_best_move(game, character_index):
	root = MCSTNode(game)
	root.character = character_index

	temp_root_character_copy = next(itertools.islice(root.state.current_player.characters, character_index, None))
	targets = temp_root_character_copy.targets

	# if there is only one target, the choice is easy
	if len(targets) == 1:
		return 0
	
	(wins, simulations) = mcst_add_childern_root(root, targets)
	root.wins += wins
	root.number_of_simulations += simulations

	# run simulations
	current_node = root
	# remeber path for score update later on
	path = [root]
	for i in range(0, 1000):
		# If current node is leaf
		if len(current_node.children) == 0:
			(wins, simulations) = mcst_add_childern(current_node)
			# Update upwards
			for node in path:
				node.wins += wins
				node.number_of_simulations += simulations
			# Go back to top of tree
			current_node = root 
			path = [root]
		
		# else find best candidate
		else:
			best_so_far = None
			best_score = 0
			for child in current_node.children:
				score = child.wins / child.number_of_simulations + math.sqrt(2*math.log1p(child.number_of_simulations)/root.number_of_simulations)
				if score >= best_score:
					best_score = score
					best_so_far = child
			current_node = best_so_far
			path.append(current_node)

	best_score = 0
	best_child = None
	for child in root.children:
		score = child.wins / child.number_of_simulations
		if score >= best_score:
			best_score = score
			best_child = child
	return best_child.target

def mcst_add_childern_root(node, targets):
	total_wins = 0
	for j in range(-1, len(targets)):
		game_copy = copy.deepcopy(node.state)
		player_copy = game_copy.current_player
		character_copy = next(itertools.islice(player_copy.characters, node.character, None))
		child_node = MCSTNode(game_copy)
		targets_copy = character_copy.targets

		if character_copy.can_attack() and j >= 0:
			target_copy = next(itertools.islice(targets_copy, j, None))
			character_copy.attack(target_copy)
		
		avalible_characters = get_characters(game_copy)
		if len(avalible_characters) > node.character + 1:
			child_node.character = node.character + 1
		else:
			child_node.character = 0

		child_node.target = j
		if current_player_wins(copy.deepcopy(game_copy)):
			child_node.wins += 1
			total_wins += 1
		child_node.number_of_simulations = 1
		node.children.append(child_node)
	return (total_wins, len(targets))

def mcst_add_childern(node):
	# If there are no usable characters left we run a simulation another time
	if node.character == 0:
		win = current_player_wins(copy.deepcopy(node.state))
		node.number_of_simulations += 1
		if win:
			node.wins += 1
			return (1, 1)
		else:
			return (0, 1)

	temp_character_copy = next(itertools.islice(node.state.current_player.characters, node.character, None))
	targets = temp_character_copy.targets
	total_wins = 0
	for j in range(-1, len(targets)):
		game_copy = copy.deepcopy(node.state)
		player_copy = game_copy.current_player
		character_copy = next(itertools.islice(player_copy.characters, node.character, None))
		child_node = MCSTNode(game_copy)
		targets_copy = character_copy.targets

		if character_copy.can_attack() and j >= 0:
			target_copy = next(itertools.islice(targets_copy, j, None))
			character_copy.attack(target_copy)
		
		avalible_characters = get_characters(game_copy)
		character_index = 0
		if len(avalible_characters) > node.character + 1:
			character_index = node.character + 1

		child_node.character = character_index
		child_node.target = j
		if current_player_wins(copy.deepcopy(game_copy)):
			child_node.wins += 1
			total_wins += 1
		child_node.number_of_simulations = 1
		node.children.append(child_node)
	return (total_wins, len(targets))


def play_turn_mcst(game, model, optimizer):
	player = game.current_player
	state = game_state(game)

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
		for (i, character) in enumerate(player.characters):
			#turn_recap = []
			if character.can_attack():
				# find characters index
				character_index = 0
				for (j, current_character) in enumerate(player.characters):
					if character == current_character:
						character_index = j
				best_target = mcst_simulation_best_move(copy.deepcopy(game), character_index)
				#print("Best target: ", best_target)
				#if best_target == -1:
					#print(character, " dosen't attack")
				for (j, target) in enumerate(character.targets):
					if j == best_target:
						character.attack(target)
						#turn_recap.append([character, target])
						result = model.forward(state)
						# result is 0-9 and best_target is -1-8 so we add 1
						model.loss(result, best_target + 1).backward()
						optimizer.step()
						optimizer.zero_grad()
						break
			'''
			for action in  turn_recap:
				print(action[0], " attacks ", action[1])
				print()
			'''
		break

	game.end_turn()
	return game

def play_full_game(game, model, optimizer):
	for player in game.players:
		mull_count = random.randint(0, len(player.choice.cards))
		cards_to_mulligan = random.sample(player.choice.cards, mull_count)
		player.choice.choose(*cards_to_mulligan)

	while True:
		play_turn_mcst(game, model, optimizer)

	return game

def simple_deck():
	cards = ["CS2_182", "CS2_200", "CS2_231", "CS2_168", "CS2_172", "CS2_120", "CS2_118", "CS2_186", "DRG_239", "NEW1_021", "EX1_522", "EX1_007", "CS2_119", "BT_728"]
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
	
	return torch.tensor(characters, requires_grad=True)

def setup_game():
	deck1 = simple_deck()
	deck2 = simple_deck() #random_draft(CardClass.WARRIOR)
	player1 = Player("Player1", deck1, CardClass.MAGE.default_hero)
	player2 = Player("Player2", deck2, CardClass.MAGE.default_hero)

	game = Game(players=(player1, player2))
	game.start()

	return game

if __name__ == "__main__":
	model_path = "hs_state_dict_model.pt"
	cards.db.initialize()
	model = TargetSelectionModel()
	model.load_state_dict(torch.load(model_path))
	model.eval()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
	for i in range(0, 100):
		try:
			game = setup_game()
			play_full_game(game, model, optimizer)
		except GameOver:
			print("Game completed normally.")
			torch.save(model.state_dict(), model_path)