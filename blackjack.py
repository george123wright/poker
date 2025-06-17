from functools import lru_cache
from collections import defaultdict

CARD_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_IDX   = {r: i for i, r in enumerate(CARD_RANKS)}
TEN_RANKS  = {'10', 'J', 'Q', 'K'}


def card_values(rank):
    """Return list of possible values for *rank*."""
    if rank in TEN_RANKS:
        return [10]
    if rank == 'A':
        return [1, 11]
    return [int(rank)]


def make_shoe(n_decks: int = 6):
    """Return an initial shoe as a *tuple* of counts, one per rank."""
    return tuple(4 * n_decks for _ in CARD_RANKS)


def remove_card(shoe: tuple, idx: int):
    """Return a *new* shoe with one card of index *idx* removed."""
    lst = list(shoe)
    lst[idx] -= 1
    return tuple(lst)


def total_cards(shoe: tuple) -> int:
    return sum(shoe)


def best_score(hand: tuple) -> int:
    """Best (highest ≤ 21) total, counting aces flexibly."""
    total, aces = 0, 0
    for r in hand:
        if r == 'A':
            aces += 1
        else:
            total += card_values(r)[0]
    total += aces  
    for _ in range(aces):
        if total + 10 <= 21:
            total += 10
    return total


def is_blackjack(hand: tuple) -> bool:
    return len(hand) == 2 and 'A' in hand and any(r in TEN_RANKS for r in hand)


def payoff(ps: int | str, ds: int | str, *, blackjack: bool = False, surrender: bool = False) -> float:
    """Return main‑bet payoff (player outcome vs dealer total)."""
    if surrender:
        return -0.5
    if blackjack:
        return 1.5
    if ds == 'bust':
        return 1.0
    if ps > ds:
        return 1.0
    if ps < ds:
        return -1.0
    return 0.0  


def draw_n(shoe: tuple, n: int):
    """Yield (seq, prob, new_shoe) for every *ordered* n‑card draw."""
    states = [((), 1.0, shoe)]
    for _ in range(n):
        nxt = []
        for seq, p, sh in states:
            tc = total_cards(sh)
            if tc == 0:
                continue
            for r in CARD_RANKS:
                idx = RANK_IDX[r]
                cnt = sh[idx]
                if cnt == 0:
                    continue
                pr = cnt / tc
                nxt.append((seq + (r,), p * pr, remove_card(sh, idx)))
        states = nxt
    return states


@lru_cache(maxsize=None)
def dealer_dist(dealer_start: tuple, shoe: tuple, hit_soft17: bool):
    """Return {total_or_'bust': prob} for the dealer's final hand."""

    sc = best_score(dealer_start)
    aces = dealer_start.count('A')
    hard_total = sum(card_values(r)[0] for r in dealer_start if r != 'A') + aces
    soft = aces > 0 and hard_total + 10 <= 21

    if sc > 21:
        return {'bust': 1.0}

    if sc >= 17 and not (soft and sc == 17 and hit_soft17):
        return {sc: 1.0}

    tc = total_cards(shoe)
    dist = defaultdict(float)
    for r in CARD_RANKS:
        idx = RANK_IDX[r]
        cnt = shoe[idx]
        if cnt == 0:
            continue
        pr = cnt / tc
        sh2 = remove_card(shoe, idx)
        for tot, p2 in dealer_dist(dealer_start + (r,), sh2, hit_soft17).items():
            dist[tot] += pr * p2
    return dict(dist)


def insurance_ev(hand: tuple, upcard: str, shoe: tuple, rules: dict) -> float:
    """EV of *accepting* insurance (half‑unit side‑bet) while standing."""
    if upcard != 'A' or not rules.get('can_insurance', False):
        return float('-inf')

    tc = total_cards(shoe)
    cnt_ten = sum(shoe[RANK_IDX[r]] for r in TEN_RANKS)
    p_bj = cnt_ten / tc 

    player_bj = is_blackjack(hand)
    main_if_bj = 0.0 if player_bj else -1.0
    main_ev = p_bj * main_if_bj + (1 - p_bj) * stand_ev(hand, upcard, shoe, rules)

    side_ev = p_bj * 1.0 + (1 - p_bj) * -0.5
    return main_ev + side_ev


def stand_ev(hand: tuple, upcard: str, shoe: tuple, rules: dict) -> float:
    """EV of **standing** on *hand* with dealer upcard *upcard* and *shoe*."""
    ps = best_score(hand)
    player_bj = is_blackjack(hand)
    hit_soft17 = rules.get('hit_soft17', False)
    use_peek = rules.get('use_peek', True)

    ev = 0.0
    tc = total_cards(shoe)

    if use_peek and upcard in TEN_RANKS | {'A'}:
        hole_rank = '10' if upcard == 'A' else 'A'
        idx_hole = RANK_IDX[hole_rank]
        cnt_hole = shoe[idx_hole]

        if cnt_hole:
            p_dealer_bj = cnt_hole / tc
            ev += p_dealer_bj * (0.0 if player_bj else -1.0)

            tc_no_bj = tc - cnt_hole
            for r in CARD_RANKS:
                idx = RANK_IDX[r]
                cnt = shoe[idx]
                if r == hole_rank or cnt == 0:
                    continue
                pr = cnt / tc_no_bj
                sh2 = remove_card(shoe, idx)
                for outcome, p2 in dealer_dist((upcard, r), sh2, hit_soft17).items():
                    ev += (1 - p_dealer_bj) * pr * p2 * \
                          payoff(ps, outcome, blackjack=player_bj)
            return ev

    for r in CARD_RANKS:
        idx = RANK_IDX[r]
        cnt = shoe[idx]
        if cnt == 0:
            continue
        pr = cnt / tc
        sh2 = remove_card(shoe, idx)
        for outcome, p2 in dealer_dist((upcard, r), sh2, hit_soft17).items():
            ev += pr * p2 * payoff(ps, outcome, blackjack=player_bj)

    return ev


HIT_PRUNE  = 1.0   
SPLIT_PRUNE = 4.5  

@lru_cache(maxsize=None)
def game_ev(shoe: tuple, upcard: str, hand: tuple, rules_key: frozenset, *, split_depth: int = 0) -> float:
    """Return the EV of playing *hand* optimally from this point on."""
    rules = dict(rules_key)

    if hand and best_score(hand) > 21:
        return -1.0

    tc = total_cards(shoe)

    if not hand:
        ev_acc = 0.0
        for h, p, sh2 in draw_n(shoe, 2):
            ev_acc += p * game_ev(sh2, upcard, h, rules_key, split_depth=split_depth)
        return ev_acc

    ev_st = stand_ev(hand, upcard, shoe, rules)
    best = ev_st

    if rules.get('can_double', True) and len(hand) == 2:
        ev_d = 0.0
        for r in CARD_RANKS:
            idx = RANK_IDX[r]
            if shoe[idx] == 0:
                continue
            pr = shoe[idx] / tc
            sh2 = remove_card(shoe, idx)
            h2 = hand + (r,)
            ev_after = -1.0 if best_score(h2) > 21 else stand_ev(h2, upcard, sh2, rules)
            ev_d += pr * ev_after
        ev_d *= 2
    else:
        ev_d = float('-inf')
    best = max(best, ev_d)

    ev_s = -0.5 if rules.get('can_surrender', False) and len(hand) == 2 else float('-inf')
    best = max(best, ev_s)

    ev_ins = insurance_ev(hand, upcard, shoe, rules)
    ev_even = 1.0 if (rules.get('can_even_money', True) and is_blackjack(hand) and upcard == 'A') else float('-inf')
    best = max(best, ev_ins, ev_even)

    ev_h = float('-inf')
    if rules.get('can_hit', True) and best < HIT_PRUNE:
        ev_h = 0.0
        for r in CARD_RANKS:
            idx = RANK_IDX[r]
            if shoe[idx] == 0:
                continue
            pr = shoe[idx] / tc
            sh2 = remove_card(shoe, idx)
            h2 = hand + (r,)
            if best_score(h2) > 21:
                ev_h += pr * -1.0
            else:
                ev_h += pr * game_ev(sh2, upcard, h2, rules_key, split_depth=split_depth)
        best = max(best, ev_h)

    ev_sp = float('-inf')
    if (rules.get('can_split', True) and len(hand) == 2 and hand[0] == hand[1] and
        split_depth < rules.get('max_splits', 3) and best < SPLIT_PRUNE):

        ev_sp = 0.0
        for r1, p1, sh1 in draw_n(shoe, 1):
            h1 = (hand[0],) + r1
            if hand[0] == 'A' and rules.get('split_aces_one_card', True):
                ev1 = stand_ev(h1, upcard, sh1, rules)
            else:
                ev1 = game_ev(sh1, upcard, h1, rules_key, split_depth=split_depth + 1)

            for r2, p2, sh2 in draw_n(sh1, 1):
                h2 = (hand[0],) + r2
                if hand[0] == 'A' and rules.get('split_aces_one_card', True):
                    ev2 = stand_ev(h2, upcard, sh2, rules)
                else:
                    ev2 = game_ev(sh2, upcard, h2, rules_key, split_depth=split_depth + 1)
                ev_sp += p1 * p2 * (ev1 + ev2)
        best = max(best, ev_sp)

    return best


def simulate_action(action: str, hand: tuple, shoe: tuple, upcard: str, rules_key: frozenset, *, split_depth: int = 0) -> float:
    """Return EV of *forcing* a specific action on *hand*."""
    rules = dict(rules_key)
    tc = total_cards(shoe)

    if action == 'stand':
        return stand_ev(hand, upcard, shoe, rules)

    if action == 'hit':
        ev = 0.0
        for r in CARD_RANKS:
            idx = RANK_IDX[r]
            if shoe[idx] == 0:
                continue
            pr = shoe[idx] / tc
            sh2 = remove_card(shoe, idx)
            h2 = hand + (r,)
            if best_score(h2) > 21:
                ev += pr * -1.0
            else:
                ev += pr * game_ev(sh2, upcard, h2, rules_key, split_depth=split_depth)
        return ev

    if action == 'double':
        if not (rules.get('can_double', True) and len(hand) == 2):
            return float('-inf')
        ev = 0.0
        for r in CARD_RANKS:
            idx = RANK_IDX[r]
            if shoe[idx] == 0:
                continue
            pr = shoe[idx] / tc
            sh2 = remove_card(shoe, idx)
            h2 = hand + (r,)
            ev_after = -1.0 if best_score(h2) > 21 else stand_ev(h2, upcard, sh2, rules)
            ev += pr * ev_after
        return 2 * ev

    if action == 'split':
        if not (rules.get('can_split', True) and len(hand) == 2 and hand[0] == hand[1] and
                split_depth < rules.get('max_splits', 3)):
            return float('-inf')
        ev_split = 0.0
        for seq1, p1, shoe1 in draw_n(shoe, 1):
            h1 = (hand[0],) + seq1
            if hand[0] == 'A' and rules.get('split_aces_one_card', True):
                ev1 = stand_ev(h1, upcard, shoe1, rules)
            else:
                ev1 = game_ev(shoe1, upcard, h1, rules_key, split_depth=split_depth + 1)
            for seq2, p2, shoe2 in draw_n(shoe1, 1):
                h2 = (hand[0],) + seq2
                if hand[0] == 'A' and rules.get('split_aces_one_card', True):
                    ev2 = stand_ev(h2, upcard, shoe2, rules)
                else:
                    ev2 = game_ev(shoe2, upcard, h2, rules_key, split_depth=split_depth + 1)
                ev_split += p1 * p2 * (ev1 + ev2)
        return ev_split

    if action == 'surrender':
        return -0.5 if rules.get('can_surrender', False) and len(hand) == 2 else float('-inf')

    if action == 'insurance':
        return insurance_ev(hand, upcard, shoe, rules)

    if action == 'even_money':
        return 1.0 if (rules.get('can_even_money', True) and is_blackjack(hand) and upcard == 'A') else float('-inf')

    return float('-inf')  


def optimal_action(shoe: tuple, upcard: str, hand: tuple, rules: dict):
    """Return (best_action, EV) for a single player hand."""
    rules_key = frozenset(rules.items())
    evs = {act: simulate_action(act, tuple(hand), shoe, upcard, rules_key) for act in
           ['stand', 'hit', 'double', 'split', 'surrender', 'insurance', 'even_money']}
    best_act = max(evs, key=evs.get)
    return best_act, evs[best_act]


if __name__ == '__main__':
    rules = {
        'can_double': True,
        'can_split': True,
        'split_aces_one_card': True,
        'max_splits': 3,
        'can_surrender': True,
        'can_insurance': True,
        'can_even_money': True,
        'can_hit': True,
        'use_peek': True,
        'hit_soft17': False,
    }

    shoe = make_shoe(6)
    shoe = remove_card(remove_card(shoe, RANK_IDX['A']), RANK_IDX['7'])
    shoe = remove_card(shoe, RANK_IDX['K'])

    act, ev = optimal_action(shoe, 'K', ('A', '7'), rules)
    print(f"Optimal action: {act},  EV = {ev:.4f}")
