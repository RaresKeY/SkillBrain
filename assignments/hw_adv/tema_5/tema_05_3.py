# [x] III. Optional Topic (Advanced Level) – Blackjack
# Goal: build the Blackjack game engine using functions.

# The main program must call only 2 functions:
#   player = seat_player(name, budget)
#   run_blackjack(player)

# The rest should be organized through functions.
# The program must run step by step, stopping whenever user input is required
# (e.g., when deciding whether to “Hit” or “Hold”).

"""
## Flow

First: prompt AI as such:
    Explain blackjack rules in an organized manner
Second: Organize rules manually and make functions as you go

## Blackjack Rules:

Traditional: 1 deck 
Modern: 4-8 decks

Players: 1-7 players against dealer

[x] Objectives:
    [x] Get a hand closer to 21 than house
    [x] Do not go over 21

[x] Card Values:
    [x] Number Cards: 2-10 - face value
    [x] Face Cards: (J, Q, K): 10 points
    [x] Ace (A): 1 or 11 (whichever benefits the player's hand)

[x] Initial Deal:
    [x] Each player gets 2 cards face up
    [x] Dealer gets:
        - 1 card face up (the "upcard")
        - 1 card face down (the "hole card")
    [x] Order:
        - each player gets 1 face up card
        - dealer gets 1 card face up (the "upcard")
        - each player gets 1 face up card
        - dealer gets 1 card face down (the "hole card")

[x] Betting (before any cards are dealt)

[x] Player's Turn:
    Each Player decides:
    [x] - Hit: take another card
    [x] - Stand (Hold): keep current hand
    [x] - Double Down (optional rule): 
        double the bet, take one card only, then stand
    [ ] - Split (optional rule):
        if both cards are the same value, 
        split into two hands (requires another bet)
    Player continues until they 'Stand' or 'Bust'

[x] Dealer's Turn:
    - Dealer reveals his hole card
    - Must hit until reaching 17 or higher (rules vary)

[x] Resolution - Compare Hands:
    - Player Hand > Dealer Hand -> Player wins
    - Dealer Hand > Player Hand -> Dealer wins
    - Tie -> Push (no one wins, bet returned)
"""

import random
import math
from painting_on_water import vinput

# [x] top-level TODO: test play flow
# [x] top-level TODO: wrap input prompts with validations (generalized function)
# [ ] top-level TODO: re-make with Game class to hold deck, players, dealers and state, easier than passing dicts

# real randomness from the OS entropy pool
sys_rand = random.SystemRandom()

# useful lists
NUMBER_CARDS = [str(i) for i in range(2, 11)]
FACE_CARDS = ['J', 'Q', 'K']
ACE = ['A']

# one full deck
ORIGIN_DECK = NUMBER_CARDS * 4 + FACE_CARDS * 4 + ACE * 4

# debug stuff, verbosity
DEBUG = True

def check_hand(hand: list[str]) -> dict:
    """return 'bust' or card score if not bust"""

    # convert cards to score
    score = 0
    ace_count = 0
    for card in hand:
        if card in FACE_CARDS:
            score += 10
        elif card in ACE:
            ace_count += 1
        elif card in NUMBER_CARDS:
            score += int(card)

    # determine aces score
    score += ace_count # default to one
    for _ in range(ace_count):
        if score + 10 <= 21:
            score += 10 # change to 11 if beneficial
    
    return {"score": score, "busted": score > 21}


def hit(player: dict, deck: list, count: int = 1):
    """change player hand in place, remove from deck in place"""
    for _ in range(count):
        random_card = sys_rand.choice(deck)
        deck.remove(random_card)
        player["cards"].append(random_card)
        print(f"You hit {random_card}")


def initial_deal(players: list[dict], dealer: dict, deck: list):
    """order matters:
    - each player gets 1 face up card
    - dealer gets 1 card face up (the "upcard")
    - each player gets 1 face up card
    - dealer gets 1 card face down (the "hole card")
    """

    # first deal players face up
    for player in players:
        hit(player, deck)

    # second deal house face up
    hit(dealer, deck)

    # third deal players face up
    for player in players:
        hit(player, deck)

    # fourth deal house face down
    random_card = sys_rand.choice(deck)
    deck.remove(random_card)
    dealer["hole_card"] = random_card


def seat_player(name: str, budget: float, players: list):
    """add player to list using decided struct"""
    new_player = {
        "name": name,
        "money": budget,
        "bet": None, # (float)
        "cards": [],
        "score": None, # (int) card from check_hand()
        "stand": False, # indicates player turn is over
        "busted": False # check_hand()
    }
    players.append(new_player)


def seat_dealer(name: str):
    """organized way to get fresh dealer"""
    dealer = {
        "name": name,
        "money": math.inf,
        "bet": 0, # placeholder to avoid errors
        "cards": [],
        "score": None, # (int) card from check_hand()
        "hole_card": None,
        "stand": False, # indicates dealer turn is over
        "busted": False # check_hand()
    }
    return dealer


def run_action_player(action: str, player:dict, deck: list, debug=DEBUG):
    """specific player action logic, calls process_turn()"""
    # TODO: introduce AI mapping resolution

    mapped_actions = {'1': "hit",         "hit": "hit", 
                      '2': "stand",       "stand": "stand",
                      '3': "double_down", "double down": "double_down",
                      '4': "split",       "split": "split",
                      }

    if action in mapped_actions:
        action = mapped_actions[action.casefold()]
    else:
        return "not_mapped"

    if action == "hit":
        if debug: print("Hit!")
        hit(player, deck, 1)

    if action == "stand":
        if debug: print("Stand!")
        player["stand"] = True

    if action == "double_down":
        if debug: print("Double Down!")
        hit(player, deck, 1)
        player["bet"] = player["bet"] * 2
        player["stand"] = True
    
    process_turn(player)
    return
    # not implemented
    if action == "split":
        if debug: print("Split!")
        if len(player["cards"]) == 2 and player["cards"][0] == player["cards"][1]:
            print("POSSIBLE (NOT YET IMPLEMENTED)")


def player_bust(player, debug=DEBUG):
    """boilerplate bust code"""
    if debug:
        print(f"Player {player["name"]} busted with score {player["score"]}, ",
              f"losing {player["bet"]}$")
    player["busted"] = True
    player["stand"] = True
    player["money"] -= player["bet"]
    player["bet"] = None
    player["score"] = None
    player["cards"] = []


def player_win(player, debug=DEBUG):
    """boilerplate win code"""
    if debug:
        print(f"Player {player["name"]} won with score {player["score"]}, ",
              f"winning {player["bet"]}$")
    player["busted"] = False
    player["stand"] = False
    player["money"] += player["bet"]
    player["bet"] = None
    player["score"] = None
    player["cards"] = []


def player_lose(player, debug=DEBUG):
    """boilerplate lose code"""
    if debug:
        print(f"Player {player["name"]} lost with score {player["score"]}, ",
              f"losing {player["bet"]}$")
    player["busted"] = False
    player["stand"] = False
    player["money"] -= player["bet"]
    player["bet"] = None
    player["score"] = None
    player["cards"] = []


def player_reset(player, debug=DEBUG):
    """boilerplate reset code"""
    if debug:
        print(f"Player {player["name"]} reset")
    player["busted"] = False
    player["stand"] = False
    player["score"] = None
    player["cards"] = []


def process_turn(player, debug=DEBUG):
    """boilerplate processing code, safe to use anywhere and twice"""
    result = check_hand(player["cards"])
    player["score"] = result["score"]
    if result["busted"]:
        player_bust(player)
    else:
        if debug:
             # TODO: change flavour text for dealer
            print(f"Player {player["name"]} score: {player["score"]}")


def run_blackjack_console(players: list, dealer: dict, deck: list):
    """contains main game logic for one round"""

    # betting 
    for player in players:
        bet_amount = vinput(f"{player["name"]}, please make a bet({player["money"]}$): ",
                                     pattern=r'^\d+(\.\d+)?$', # integers, floats, decimal point
                                     condition=lambda t: float(t) <= player['money']).casefold().strip()
        bet_amount = float(bet_amount)
        player["bet"] = bet_amount if bet_amount > 0 and bet_amount <= player["money"] else 0
        print(f"Player {player["name"]} bet {player["bet"]}")

    # initial deal logic
    initial_deal(players, dealer, deck)

    # initial busted and score check
    for player in players:
        process_turn(player)

    remaining_players = [player for player in players if not player["busted"]]

    # reveal house's face-up card prior to player turn
    print(f"Dealer has {dealer["cards"][0]} as well as one hidden card")
    process_turn(dealer)
    print(f"Dealer's partial total: {dealer["score"]}")

    # game start, players turn
    for player in remaining_players:
        print(f"\nPlayer {player["name"]} you have:"
              f"\nCash: {player["money"]}"
              f"\nScore: {player["score"]}"
              f"\n    {player["cards"]}\n"
              )
        print("Available Actions:\n",
              "1.Hit\n2.Stand\n3.Double Down\n4.Split\n", sep='')
        
        while(not player["stand"]):
            player_action = vinput("Action: ", pattern=r"^(1|2|3|4|hit|stand|double down|split)$").lower().strip()
            run_action_player(player_action, player, deck) # processes turn as well

    remaining_players = [player for player in players if not player["busted"]]

    # dealers turn
    print("\n\nDealer's Turn\n")
    print(f"Dealer's hidden card is {dealer["hole_card"]}")
    dealer["cards"].append(dealer["hole_card"])
    process_turn(dealer, debug=False)
    print(f"Dealer has score {dealer["score"]}")
    print(dealer["cards"], '\n')

    if not dealer["busted"]:
        print(f"Dealer's total: {dealer["score"]}")

    while (dealer["score"] < 17 and not dealer["busted"]):
        hit(dealer, deck, 1)
        process_turn(dealer)

    # resolution
    if not dealer["busted"]:
        print(f"Dealer's total: {dealer["score"]}")
        for player in remaining_players:
            if player["score"] > dealer["score"]:
                player_win(player)
            elif player["score"] < dealer["score"]:
                player_lose(player)
            else:
                print(f"Player {player["name"]} tied with the house")
                player_reset(player)
    else:
        print("\nRemaining players win!")
        print([player["name"] + " " for player in remaining_players])
        for player in remaining_players:
            player_win(player)

    for player in players:
        player_reset(player)

    return


if __name__ == "__main__":
    # example one game with one round

    # get fresh deck
    modern_variant = False
    deck = ORIGIN_DECK * 8 if modern_variant else ORIGIN_DECK

    # get fresh players list
    players = []

    # example seat players
    seat_player("Rares", float(sys_rand.randint(2000, 8000)), players)
    seat_player("Raul", float(sys_rand.randint(2000, 8000)), players)
    seat_player("Cosmin", float(sys_rand.randint(2000, 8000)), players)

    # seat dealer
    new_dealer = seat_dealer("Dealer1")

    # shuffle deck
    sys_rand.shuffle(deck)

    # start round example
    # run_blackjack_console(players, new_dealer, deck)

    # keep the game rolling
    keep_on = True
    while keep_on:
        run_blackjack_console(players, new_dealer, deck)
        keep_on = True if vinput("Keep Playing?(yes/no): ", pattern=r'^(yes|no)$').lower().strip() == "yes" else False
        deck = ORIGIN_DECK * 8 if modern_variant else ORIGIN_DECK # fresh deck
        sys_rand.shuffle(deck) # shuffle
