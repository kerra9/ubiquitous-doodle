"""CLI entry point for the basketball simulator.

Usage:
    python -m basketball_sim [--seed SEED] [--quarters N] [--quiet]
"""

from __future__ import annotations

import argparse
import logging
import random
import sys

from basketball_sim.ai.coach_ai import CoachAI
from basketball_sim.ai.defensive_ai import BasicDefensiveAI
from basketball_sim.ai.offensive_ai import BasicOffensiveAI
from basketball_sim.core.engine import GameEngine
from basketball_sim.core.event_bus import EventBus
from basketball_sim.core.pipeline import ModifierPipeline
from basketball_sim.core.types import (
    EventType,
    GameEvent,
    GameState,
    Player,
    PlayerAttributes,
    PlayerTendencies,
    RulesConfig,
    ShootingProfile,
    TeamState,
)
from basketball_sim.data.loader import load_moves
from basketball_sim.resolvers.composite import CompositeResolver
from basketball_sim.modifiers.chemistry import chemistry_modifier
from basketball_sim.modifiers.coaching import coaching_modifier
from basketball_sim.modifiers.fatigue import fatigue_modifier
from basketball_sim.modifiers.history import history_modifier, reset_history
from basketball_sim.modifiers.psychology import psychology_modifier
from basketball_sim.modifiers.situational import situational_modifier
from basketball_sim.modifiers.tendencies import tendencies_modifier
from basketball_sim.narration.aggregator import EventAggregator
from basketball_sim.narration.enricher import ContextEnricher
from basketball_sim.narration.renderer import ProseRenderer, _format_clock
from basketball_sim.narration.stats_tracker import StatsTracker
from basketball_sim.narration.templates import TemplateSelector


# ---------------------------------------------------------------------------
# Roster generation (placeholder until real roster JSON exists)
# ---------------------------------------------------------------------------

_POSITIONS = ["PG", "SG", "SF", "PF", "C"]

_TEAM_A_NAMES = [
    "Marcus Cole", "Jaylen Wright", "DeAndre Harris", "Tyrone Mitchell", "Kwame Johnson",
    "Andre Williams", "Darius Brown", "Malik Thompson", "Chris Parker", "Jamal Robinson",
]

_TEAM_B_NAMES = [
    "Jordan Thomas", "Brandon Lewis", "Kevin Reed", "Anthony Davis Jr", "Demarcus White",
    "Terrence Hill", "Lamar Scott", "Rasheed Wallace Jr", "Devon Carter", "Isaiah Green",
]


def _make_player(pid: str, name: str, team: str, position: str, rng: random.Random) -> Player:
    """Generate a player with randomized but position-appropriate attributes."""
    # Position-based attribute tendencies
    is_guard = position in ("PG", "SG")
    is_big = position in ("PF", "C")

    bh = rng.randint(65, 92) if is_guard else rng.randint(40, 70)
    spd = rng.randint(70, 95) if is_guard else rng.randint(55, 78)
    tp = rng.randint(65, 90) if is_guard else rng.randint(50, 75)
    mr = rng.randint(65, 88)
    dl = rng.randint(70, 90) if is_guard else rng.randint(60, 82)
    dnk = rng.randint(40, 75) if is_guard else rng.randint(55, 90)
    pv = rng.randint(65, 92) if position == "PG" else rng.randint(50, 78)
    pa = rng.randint(65, 90) if position == "PG" else rng.randint(55, 80)
    pd = rng.randint(65, 90) if is_guard else rng.randint(50, 75)
    id_ = rng.randint(50, 75) if is_guard else rng.randint(65, 92)
    stl = rng.randint(60, 88) if is_guard else rng.randint(45, 70)
    blk = rng.randint(30, 55) if is_guard else rng.randint(55, 90)
    oreb = rng.randint(30, 55) if is_guard else rng.randint(55, 85)
    dreb = rng.randint(50, 72) if is_guard else rng.randint(65, 90)
    str_ = rng.randint(50, 75) if is_guard else rng.randint(65, 92)
    vert = rng.randint(60, 88) if is_guard else rng.randint(55, 82)
    stam = rng.randint(70, 92)
    iq = rng.randint(60, 90)

    attrs = PlayerAttributes(
        ball_handling=bh, speed=spd, three_point=tp, mid_range=mr,
        driving_layup=dl, dunk=dnk, passing_vision=pv, passing_accuracy=pa,
        perimeter_defense=pd, interior_defense=id_, steal=stl, block=blk,
        offensive_rebound=oreb, defensive_rebound=dreb, strength=str_,
        vertical=vert, stamina=stam, basketball_iq=iq,
    )

    tendencies = PlayerTendencies(
        iso_frequency=rng.uniform(0.15, 0.6),
        three_vs_midrange=rng.uniform(0.3, 0.8) if is_guard else rng.uniform(0.1, 0.5),
        pass_first_vs_score=rng.uniform(0.3, 0.8) if position == "PG" else rng.uniform(0.4, 0.9),
        clutch_usage=rng.uniform(0.3, 0.8),
        flashy_play_tendency=rng.uniform(0.1, 0.5),
        heat_check_tendency=rng.uniform(0.2, 0.6),
        off_ball_movement_quality=rng.uniform(0.3, 0.8),
    )

    moves = ["crossover", "hesitation", "jab_step"]
    if bh >= 70:
        moves.append("step_back")
    if bh >= 75:
        moves.extend(["spin_move", "behind_the_back"])

    return Player(
        player_id=pid,
        display_name=name,
        team_id=team,
        position=position,
        attributes=attrs,
        tendencies=tendencies,
        shooting=ShootingProfile(),
        move_repertoire=moves,
    )


def _build_team(team_id: str, team_name: str, names: list[str], rng: random.Random) -> TeamState:
    """Build a team with generated players."""
    players = []
    for i, name in enumerate(names):
        pos = _POSITIONS[i % 5]
        pid = f"{team_id}_p{i+1}"
        players.append(_make_player(pid, name, team_id, pos, rng))

    on_court = [p.player_id for p in players[:5]]
    return TeamState(team_id=team_id, name=team_name, players=players, on_court=on_court)


# ---------------------------------------------------------------------------
# Narration output handler
# ---------------------------------------------------------------------------

class NarrationOutput:
    """Subscribes to the event bus and produces narrated output.

    Filters out low-excitement routine events and only narrates
    beats that are worth describing in prose.
    """

    def __init__(
        self,
        aggregator: EventAggregator,
        enricher: ContextEnricher,
        selector: TemplateSelector,
        renderer: ProseRenderer,
        rng: random.Random,
        quiet: bool = False,
    ) -> None:
        self._aggregator = aggregator
        self._enricher = enricher
        self._selector = selector
        self._renderer = renderer
        self._rng = rng
        self._quiet = quiet
        self._last_clock: float = -1.0
        self._last_quarter: int = 0

    def handle_event(self, event: GameEvent) -> None:
        """Process an event through the narration pipeline."""
        beat = self._aggregator.process_event(event)
        if beat is None:
            return

        enriched = self._enricher.enrich(beat)

        # Filter out low-interest events to reduce narration spam
        if self._should_skip(enriched, beat):
            return

        template = self._selector.select(enriched, rng=self._rng)
        rendered = self._renderer.render(template, enriched)

        if rendered.text and not self._quiet:
            q = f"Q{beat.quarter}"
            clock = _format_clock(beat.game_clock)
            print(f"  {q} {clock:>5s} | {rendered.text}")

    def flush(self) -> None:
        """Flush remaining beats."""
        beat = self._aggregator.flush()
        if beat is not None:
            enriched = self._enricher.enrich(beat)
            if not self._should_skip(enriched, beat):
                template = self._selector.select(enriched, rng=self._rng)
                rendered = self._renderer.render(template, enriched)
                if rendered.text:
                    q = f"Q{beat.quarter}"
                    clock = _format_clock(beat.game_clock)
                    print(f"  {q} {clock:>5s} | {rendered.text}")

    def _should_skip(self, enriched, beat) -> bool:
        """Decide whether to skip narrating this beat."""
        # Always narrate game flow events
        if beat.primary_event_type in (
            EventType.GAME_START, EventType.GAME_END,
            EventType.QUARTER_START, EventType.QUARTER_END,
            EventType.TIMEOUT, EventType.SUBSTITUTION,
        ):
            return False

        # Always narrate scoring plays
        if beat.is_scoring_play:
            return False

        # Always narrate highlights
        if enriched.is_highlight:
            return False

        # Always narrate steals, blocks, turnovers
        if beat.primary_event_type in (
            EventType.STEAL, EventType.BLOCK, EventType.TURNOVER,
            EventType.SHOT_CLOCK_VIOLATION,
        ):
            return False

        # Always narrate shot attempts (even misses)
        if beat.primary_event_type in (EventType.SHOT_ATTEMPT, EventType.SHOT_MISSED):
            return False

        # Narrate rebounds
        if beat.primary_event_type == EventType.REBOUND:
            return False

        # Skip routine possession start/end
        if beat.primary_event_type in (EventType.POSSESSION_START, EventType.POSSESSION_END):
            return True

        # For dribble moves and passes, only narrate if interesting enough
        if enriched.excitement < 0.10:
            return True

        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Basketball Simulator")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--quarters", type=int, default=4, help="Number of quarters")
    parser.add_argument("--quiet", action="store_true", help="Suppress narration output")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    rng = random.Random(args.seed)

    # Build teams
    home = _build_team("storm", "Storm", _TEAM_A_NAMES, rng)
    away = _build_team("thunder", "Thunder", _TEAM_B_NAMES, rng)

    # Build all player name lookups
    player_names: dict[str, str] = {}
    team_names: dict[str, str] = {}
    for team in (home, away):
        team_names[team.team_id] = team.name
        for p in team.players:
            player_names[p.player_id] = p.display_name

    # Load move data
    moves = load_moves()

    # Set up event bus
    bus = EventBus()

    # Set up modifier pipeline with all realism layers
    pipeline = ModifierPipeline()
    pipeline.register(fatigue_modifier, "fatigue")
    pipeline.register(psychology_modifier, "psychology")
    pipeline.register(tendencies_modifier, "tendencies")
    pipeline.register(history_modifier, "history")
    pipeline.register(situational_modifier, "situational")
    pipeline.register(chemistry_modifier, "chemistry")
    pipeline.register(coaching_modifier, "coaching")

    # Set up AI
    offensive_ai = BasicOffensiveAI(move_registry=moves)
    defensive_ai = BasicDefensiveAI()

    # Set up narration pipeline
    aggregator = EventAggregator()
    enricher = ContextEnricher()
    selector = TemplateSelector()
    renderer = ProseRenderer(
        profile=selector.profile,
        player_names=player_names,
        team_names=team_names,
    )

    narration = NarrationOutput(
        aggregator=aggregator,
        enricher=enricher,
        selector=selector,
        renderer=renderer,
        rng=rng,
        quiet=args.quiet,
    )

    # Set up stats tracker
    stats = StatsTracker()
    stats.register_team(home.team_id, home.name)
    stats.register_team(away.team_id, away.name)
    for team in (home, away):
        for p in team.players:
            stats.register_player(p.player_id, team.team_id, p.display_name)

    # Subscribe to event bus
    bus.subscribe_all(narration.handle_event)
    bus.subscribe_all(stats.handle_event)

    # Reset per-game state
    reset_history()

    # Set up rules
    rules = RulesConfig(num_quarters=args.quarters)

    # Build composite resolver (replaces StubResolver with real Phase 2 resolvers)
    resolver = CompositeResolver(pipeline=pipeline, moves=moves)

    # Build engine
    engine = GameEngine(
        event_bus=bus,
        pipeline=pipeline,
        offensive_ai=offensive_ai,
        defensive_ai=defensive_ai,
        resolver=resolver,
        rules=rules,
    )

    # Build game state
    game = GameState(
        home_team=home,
        away_team=away,
        quarter=1,
        game_clock=rules.quarter_length,
        possession_team_id=home.team_id,
        score={"home": 0, "away": 0},
        rng=random.Random(args.seed),
    )

    # Print header
    print(f"\n{'=' * 60}")
    print(f"  {home.name} vs {away.name}")
    print(f"  Seed: {args.seed} | Quarters: {args.quarters}")
    print(f"{'=' * 60}\n")

    # Run the simulation
    final = engine.simulate_game(game)

    # Flush narration
    narration.flush()

    # Print final score
    print(f"\n{'=' * 60}")
    print(f"  FINAL SCORE")
    print(f"  {home.name}: {final.score['home']}  |  {away.name}: {final.score['away']}")
    print(f"{'=' * 60}")

    # Print box scores
    if not args.quiet:
        print(stats.format_box_scores())

    # Print engine stats
    print(f"\nEngine: {engine.stats.possessions_simulated} possessions, "
          f"{engine.stats.actions_resolved} actions in {engine.stats.total_time_seconds:.3f}s")
    print(f"  Avg time per possession: {engine.stats.avg_time_per_possession * 1000:.2f}ms")


if __name__ == "__main__":
    main()
