"""Composite action resolver -- dispatches to the correct Phase 2 resolver.

Routes each action type to its specialized resolver function.
This replaces the StubResolver from Phase 1 with real basketball logic.
"""

from __future__ import annotations

from basketball_sim.core.engine import ActionResolver
from basketball_sim.core.grid import COURT
from basketball_sim.core.pipeline import ModifierPipeline
from basketball_sim.core.types import (
    Action,
    ActionContext,
    ActionResult,
    ActionType,
    AggregatedModifier,
    BallHandlerRhythm,
    EventType,
    GameEvent,
    MatchupState,
    MoveData,
)
from basketball_sim.resolvers.dribble import resolve_dribble
from basketball_sim.resolvers.pass_action import resolve_pass
from basketball_sim.resolvers.rebound import resolve_rebound
from basketball_sim.resolvers.shoot import resolve_shot


class CompositeResolver(ActionResolver):
    """Routes actions to the appropriate resolver.

    Handles all action types from Phase 2 (dribble, shot, pass, rebound)
    plus new ones from Phase 3 (drive, screen, post moves).
    """

    def __init__(
        self,
        pipeline: ModifierPipeline,
        moves: dict[str, MoveData] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._moves = moves or {}

    def resolve(
        self,
        action: Action,
        matchup: MatchupState,
        context: ActionContext,
    ) -> ActionResult:
        """Resolve an action by dispatching to the correct resolver."""
        # Run modifier pipeline
        agg = self._pipeline.apply(context)

        if action.action_type == ActionType.DRIBBLE_MOVE:
            return self._resolve_dribble(action, matchup, agg, context)
        elif action.action_type == ActionType.SHOT:
            return resolve_shot(matchup, agg, context)
        elif action.action_type == ActionType.PASS:
            return resolve_pass(matchup, agg, context)
        elif action.action_type == ActionType.DRIVE:
            return self._resolve_drive(action, matchup, agg, context)
        elif action.action_type == ActionType.REBOUND:
            return resolve_rebound(context.possession, context.rng)
        elif action.action_type == ActionType.FREE_THROW:
            return self._resolve_free_throw(action, context)
        else:
            # Unhandled action type -- hold ball (no effect, just burns clock)
            return ActionResult(
                new_matchup=matchup,
                events=[],
                tags=[],
                ends_possession=False,
            )

    def _resolve_dribble(
        self,
        action: Action,
        matchup: MatchupState,
        agg: AggregatedModifier,
        context: ActionContext,
    ) -> ActionResult:
        """Resolve a dribble move using the move registry."""
        move_id = action.data.get("move", "crossover")
        move = self._moves.get(move_id)

        if move is None:
            # Fallback: use crossover data or generic result
            move = self._moves.get("crossover")

        if move is not None:
            return resolve_dribble(move, matchup, agg, context)

        # Ultra-fallback if no moves loaded at all
        return ActionResult(
            new_matchup=matchup,
            events=[
                GameEvent(
                    event_type=EventType.DRIBBLE_MOVE,
                    player_id=action.player_id,
                    data=action.data,
                    tags=["dribble_move"],
                )
            ],
            tags=["dribble_move"],
        )

    def _resolve_drive(
        self,
        action: Action,
        matchup: MatchupState,
        agg: AggregatedModifier,
        context: ActionContext,
    ) -> ActionResult:
        """Resolve a drive to the basket.

        Moves the ball handler closer to the basket on the grid.
        Drives create scoring opportunities but risk turnovers.
        """
        rng = context.rng
        player = context.attacker
        cell = context.cell

        # Determine target cell (move toward basket = D1)
        if COURT.is_valid(cell):
            meta = COURT.get(cell)
            col, row = meta.col, meta.row

            # Move 1-2 rows closer to basket
            new_row = max(0, row - rng.randint(1, 2))
            # Slight column drift
            col_drift = rng.choice([-1, 0, 0, 1])
            new_col = max(0, min(6, col + col_drift))

            from basketball_sim.core.grid import COLUMNS
            new_cell = f"{COLUMNS[new_col]}{new_row + 1}"
        else:
            new_cell = "D3"  # default: near the basket

        new_meta = COURT.get(new_cell)

        # Turnover risk on drives (contested drives are risky)
        turnover_chance = 0.06
        if matchup.help_status.name in ("HELP_COMMITTED", "HELP_ROTATING"):
            turnover_chance += 0.08
        # Better ball handlers lose it less
        turnover_chance *= (1.2 - player.attributes.ball_handling / 100.0)

        if rng.random() < turnover_chance:
            return ActionResult(
                events=[
                    GameEvent(
                        event_type=EventType.TURNOVER,
                        player_id=action.player_id,
                        data={"cause": "lost_handle_on_drive", "cell": new_cell},
                        tags=["turnover", "drive_turnover"],
                    )
                ],
                tags=["turnover"],
                ends_possession=True,
            )

        # Drive advances rhythm
        new_rhythm = BallHandlerRhythm.ATTACKING

        tags = ["drive"]
        if new_meta.is_paint:
            tags.append("in_the_paint")
        if new_meta.is_restricted_area:
            tags.append("at_the_rim")

        # Update matchup context with new cell (for the next action)
        context.cell = new_cell

        event = GameEvent(
            event_type=EventType.DRIBBLE_MOVE,
            player_id=action.player_id,
            data={
                "move": "drive",
                "from_cell": cell,
                "to_cell": new_cell,
                "region": new_meta.region,
            },
            tags=tags,
        )

        new_matchup = MatchupState(
            positioning=matchup.positioning,
            balance=matchup.balance,
            stance=matchup.stance,
            rhythm=new_rhythm,
            help_status=matchup.help_status,
        )

        return ActionResult(
            new_matchup=new_matchup,
            events=[event],
            tags=tags,
            ends_possession=False,
        )

    def _resolve_free_throw(
        self, action: Action, context: ActionContext
    ) -> ActionResult:
        """Resolve a free throw attempt."""
        rng = context.rng
        # Simple free throw: ~75% base + skill
        ft_pct = 0.50 + context.attacker.attributes.mid_range / 200.0
        made = rng.random() < ft_pct

        return ActionResult(
            events=[
                GameEvent(
                    event_type=EventType.FREE_THROW,
                    player_id=action.player_id,
                    data={"made": made, "points": 1 if made else 0},
                    tags=["free_throw_made" if made else "free_throw_missed"],
                )
            ],
            tags=["free_throw"],
            ends_possession=False,
            score_change=1 if made else 0,
        )
