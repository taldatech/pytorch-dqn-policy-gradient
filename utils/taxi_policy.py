import gym
import sys
from six import StringIO
from gym import utils

seed = 0
taxi_env = gym.make('Taxi-v2')
taxi_env.seed(seed)

MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : : : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]



"""
    passenger (not in taxi) and goal locations are one of 4 possibilities:
    R(ed)     code: 0 row: 0  col: 0
    G(reen)   code: 1 row: 0  col: 4
    Y(ellow)  code: 2 row: 4  col: 0
    B(lue)    code: 3 row: 4  col: 3 
"""
destidx_to_rowcol = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}


def decode_action(action):
    """
    decodes action number (int 0included-5 excluded) to word
    :param action: action to decode
    :return: action meaning 'str'
    """
    assert 0 <= action < 5, 'action  must be a number from 0 (included) to 5 (excluded)'
    return ["South", "North", "East", "West", "Pickup", "Dropoff"][action]


def encode(taxirow, taxicol, passloc, destidx):
    # (5) 5, 5, 4
    state = taxirow
    state *= 5
    state += taxicol
    state *= 5
    state += passloc
    state *= 4
    state += destidx
    return state


def decode(state):
    """
    decodes the taxi env state (int in [0,500) ) to:
    (taxirow, taxicol, passloc, destidx) tuple
    :param state: int in [0,500)
    :return: (taxirow, taxicol, passloc, destidx) tuple
    """
    destidx = state % 4
    state = state // 4
    passloc = state % 5
    state = state // 5
    taxicol = state % 5
    state = state // 5
    taxirow = state
    assert 0 <= state < 5
    return taxirow, taxicol, passloc, destidx


def test_dedode_encode():
    """
    the goal is to test decode and encode functions written above
    :return: None
    """
    for state in range(500):
        taxirow, taxicol, passloc, destidx = decode(state)
        encoded_state = encode(taxirow, taxicol, passloc, destidx)

        assert encoded_state == state, 'state does not equal encoded state!!'

        orig_decode = tuple(taxi_env.unwrapped.decode(state))
        assert taxirow == orig_decode[0]
        assert taxicol == orig_decode[1]
        assert passloc == orig_decode[2]
        assert destidx == orig_decode[3]

    print('test passed Hurray!!')


def go_to_location(taxi_row, taxi_col, destination, have_passenger):
    """
    returns the navigational action need in case dest != taxi location
    else return pick up or drop off action based on weather we have the passenger already or not
    :param taxi_row: taxi row [0 - 4 ]
    :param taxi_col: taxi col [0 - 4 ]
    :param destination: destination [0 - 3 ]
    :return: action [0 - 5 ]
    """

    MAP = [
        "+---------+",
        "|R: | : :G|",
        "| : : : : |",
        "| : : : : |",
        "| | : | : |",
        "|Y| : |B: |",
        "+---------+",
    ]
    encode_action = {"South": 0, "North": 1, "East": 2, "West": 3, "Pickup": 4, "Dropoff": 5}

    if (taxi_row, taxi_col) == destidx_to_rowcol[destination]:
        return encode_action['Dropoff'] if have_passenger else encode_action['Pickup']

    des_row, dest_col = destidx_to_rowcol[destination]

    if taxi_col == dest_col:
        return encode_action['North'] if taxi_row > des_row else encode_action['South']

    if taxi_row in [3, 4] and taxi_col in [0, 1, 2, 3]:
        return encode_action['North']

    if taxi_row == 0:
        if taxi_col == 1 and destination == 0:
            return encode_action['West']
        if taxi_col == 2 and destination == 1:
            return encode_action['East']
        return encode_action['South']

    return encode_action['West'] if taxi_col > dest_col else encode_action['East']


def optimal_human_policy(state):
    """
    this function represents the optimal human policy which i coded by hand
    :param state: state of the taxi environment , a number between 0(included)  and 500(excluded)
    :return:
    """

    taxirow, taxicol, passloc, destidx = decode(state)

    if passloc == 4:
        # passenger is in the taxi, we go straight to destination |action chosen includes drop_off
        return go_to_location(taxirow, taxicol, destidx, have_passenger=True)
    else:
        # we go to passenger location |action chosen includes pickup
        return go_to_location(taxirow, taxicol, passloc, have_passenger=False)

def render(self,state, mode='human'):
    outfile = StringIO() if mode == 'ansi' else sys.stdout

    out = self.desc.copy().tolist()
    out = [[c.decode('utf-8') for c in line] for line in out]
    taxirow, taxicol, passidx, destidx = decode(state)
    def ul(x): return "_" if x == " " else x
    if passidx < 4:
        out[1+taxirow][2*taxicol+1] = utils.colorize(out[1+taxirow][2*taxicol+1], 'yellow', highlight=True)
        pi, pj = self.locs[passidx]
        out[1+pi][2*pj+1] = utils.colorize(out[1+pi][2*pj+1], 'blue', bold=True)
    else: # passenger in taxi
        out[1+taxirow][2*taxicol+1] = utils.colorize(ul(out[1+taxirow][2*taxicol+1]), 'green', highlight=True)

    di, dj = self.locs[destidx]
    out[1+di][2*dj+1] = utils.colorize(out[1+di][2*dj+1], 'magenta')
    outfile.write("\n".join(["".join(row) for row in out])+"\n")
    if self.lastaction is not None:
        outfile.write("  ({})\n".format(["South", "North", "East", "West", "Pickup", "Dropoff"][self.lastaction]))
    else: outfile.write("\n")

    # No need to return anything for human
    if mode != 'human':
        return outfile
